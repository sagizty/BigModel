"""
WSI embedding dataset tools     Script  ver： Oct 16th 22:30

load a cropped dataset (ROI dataset):
    each WSI is a folder (slide_folder, name of slide_id),
    cropped tiles are inside with name like: 44004y_11136x.jpeg
->
embed to a tile_feature dataset
    each WSI is a folder (slide_folder, name of slide_id),
    all cropped tiles are embedded as one .h5 file:
        h5file['features'] is a list of numpy features, each feature (can be of multiple dims: dim1, dim2, ...)
                            for transformer embedding, the feature dim is [768]
        h5file['coords_yx'] is a list of coordinates, each item is a [Y, X], Y, X is slide_feature index in WSI

to embed the tiles, a model and its weights need to be set:
 we use Patch_embedding_model to achieve that

"""
import os
import sys
from pathlib import Path

# For convinience
this_file_dir = Path(__file__).resolve().parent
sys.path.append(str(this_file_dir.parent))
sys.path.append(str(this_file_dir.parent.parent))
sys.path.append(str(this_file_dir.parent.parent.parent))  # Go up 3 levels

# Prevent conflict with OpenMP
os.environ["MKL_THREADING_LAYER"] = "GNU"

import json
import yaml
import h5py
import numpy as np
import pandas as pd

import gc
import torch
import logging
import time
import timm
import random
import shutil
import argparse
import tempfile
from typing import Optional, List, Tuple, Union
from PIL import Image
from tqdm import tqdm
import torch.nn as nn
from torchvision import models
from torchvision import transforms
import multiprocessing
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from h5tools import hdf5_save_a_patch, hdf5_save_a_patch_coord
from Tiles_dataset import *

try:
    from ..ModelBase import Get_ROI_model
except:
    from PuzzleAI.ModelBase import Get_ROI_model


# tools for logging
def setup_logging(log_file_path):
    logging.basicConfig(filename=log_file_path, level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s')


# datasets for embedding step or loading embedded slide_level datasets

# dataset for loading the slides (cropped / embedded)
class Slide_loading_Dataset(Dataset):
    def __init__(self, root_path: str, possible_suffixes: tuple = ('.h5', '.pt', '.jpeg', '.jpg'),
                 stopping_folder_name_list: list = ['thumbnails', ]):
        '''
        This class is used to set up the slide_feature dataset for loading their slide_folder
        this can be used:
            1. for slide_feature level dataset (loading wsis from different folder and have a path list)
            2. for tile embedding to be GPU parallel to process WSIs from different location.

        Assertion: after tiling or embedding
        Each WSI has one slide_folder, the name of the folder is their 'slide_id'
        inside the slide_folder, there may be one of three kind of possible files:
            1. a .h5 file, representing embedded tiles for the WSI with ['features'] ['coords'] in h5 file
            2. a .pt file, representing embedded tiles in tensor of [N, feature_dim] format
            3. a series of .jpeg files, representing cropped ROI tiles

        Arguments:
        root_path: str
            The root path of the slide_folders, notice we don't know the dataset framework of them
            this code will go through all directories inside the root_path and therefore find each slide_folder

        possible_suffixes: tuple = ('.h5', '.pt', '.jpeg', '.jpg') the suffix of the tiles or embedded features

        stopping_folder_name_list: in searching, we stop searching for folders in stopping list
        '''
        self.root_path = root_path
        self.possible_suffixes = possible_suffixes

        # a dictionary mapping {"slide_id": absolute path of slide_folder}
        self.slide_paths = self.find_slide_paths_and_ids(stopping_folder_name_list=stopping_folder_name_list)
        self.slide_ids = list(self.slide_paths.keys())

    def find_slide_paths_and_ids(self, stopping_folder_name_list=['thumbnails']):
        """
        This operation is slow as there are many '.jpg' files in the slide_folder.
        Therefore, when it detects one slide_folder, all files inside should not be tested again.

        In searching, we stop searching for folders in the stopping list.

        For example: in searching xxx/thumbnails/xxx/xxx/xxx.jpg or xxx/thumbnails/xxx.jpeg,
        when we find 'thumbnails' (all path inside or inside the folders inside it, etc.
         should not be considered as valid slide_folder, therefore
        we should stop searching all directories under xxx/thumbnails
        """
        slide_paths = {}
        for dirpath, dirnames, _ in os.walk(self.root_path):
            # Remove directories in the stopping list from dirnames to avoid descending into them
            dirnames[:] = [d for d in dirnames if d not in stopping_folder_name_list]

            for dirname in dirnames:
                slide_folder_path = os.path.join(dirpath, dirname)
                # Check for the presence of .h5, .pt, or .jpg files and break early
                for fname in os.listdir(slide_folder_path):
                    if fname.endswith(self.possible_suffixes):
                        slide_id = dirname
                        slide_paths[slide_id] = Path(slide_folder_path)
                        break  # Break early once a valid file is found
        return slide_paths

    def __len__(self):
        return len(self.slide_paths)

    def __getitem__(self, idx):
        slide_folder = self.slide_paths[self.slide_ids[idx]]
        return slide_folder


# dataset for loading tiles from one cropped WSI_folder
class TileEncodingDataset(Dataset):
    """
    dataset for loading tiles from one cropped WSI_folder

    Arguments:
    ----------
    slide_folder: Path
        Path to a folder of WSI with tiled images inside.
    transform : torchvision.transforms.Compose, optional
        Transform to apply to each image.
    suffix : str, optional
        Suffix of the image files (default is '.jpeg').
    """

    def __init__(self, slide_folder: Path, transform=None, edge_size=224, suffix='.jpeg'):
        self.suffix = suffix
        self.image_paths = self._get_image_paths(slide_folder, suffix)

        # default_transform is only resize and to tensor
        default_transform = transforms.Compose([
            transforms.Resize(edge_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))])

        self.transform = transform or default_transform  # specified slide_feature image Transform can be assigned

    def _get_image_paths(self, slide_folder, suffix):
        """
        Helper function to get all image paths in the slide_folder with the given suffix
        """
        image_paths = [os.path.join(dp, f) for dp, _, filenames in os.walk(slide_folder)
                       for f in filenames if f.endswith(suffix)]
        return image_paths

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img_name = os.path.basename(img_path)
        # Extract y, x coordinates from the image name
        y, x = img_name.split(self.suffix)[0].split('_')  # EG: 44004y_11136x.jpeg
        y, x = int(y.replace('y', '')), int(x.replace('x', ''))
        patch_coord_yx_tensor = torch.tensor([y, x], dtype=torch.int32)
        # Load the image
        with open(img_path, "rb") as f:
            patch_image_tensor = Image.open(f).convert("RGB")  # Prepare tile in [H, W, C] int8 RGB
            if self.transform:
                patch_image_tensor = self.transform(patch_image_tensor)

        return patch_image_tensor, patch_coord_yx_tensor


def prefetch_dataloader_collate_fn(batch):
    """
    Custom collate function for PrefetchDataLoader.
    Since each batch contains (slide_id, slide_folder, dataloader), we return the first (and only) item in the batch.
    """
    return batch[0]

class PrefetchDataLoader_dataset(torch.utils.data.Dataset):
    def __init__(self, device_slide_folders,
                 dataset_builder=TileEncodingDataset, edge_size=224, embedding_batch_size=256,
                 embedding_num_workers=2, pin_memory=True):
        self.dataset_builder = dataset_builder
        self.edge_size = edge_size
        self.embedding_batch_size = embedding_batch_size
        self.embedding_num_workers = embedding_num_workers
        self.pin_memory = pin_memory
        self.device_slide_folders=device_slide_folders

    def __len__(self):
        return len(self.device_slide_folders)

    def __getitem__(self, idx):
        slide_id = self.device_slide_folders[idx][0]
        slide_folder = self.device_slide_folders[idx][1]

        # Create the dataset and dataloader
        tile_dataset = self.dataset_builder(slide_folder, edge_size=self.edge_size)
        tile_dataloader = DataLoader(tile_dataset, batch_size=self.embedding_batch_size,
                                     shuffle=False, num_workers=self.embedding_num_workers,
                                     drop_last=False, pin_memory=self.pin_memory)

        return slide_id, slide_folder, tile_dataloader


# functions for embedding the tiles from tiles_dataset
def embedding_one_slide_from_tiles(slide_folder: Union[str, Path],
                                   prefetch_loader,
                                   embedding_model_at_certain_GPU: torch.nn.Module,
                                   output_WSI_dataset_path: Optional[Union[str, Path]] = None,
                                   batch_size: int = 256,
                                   device: str = 'cuda',
                                   embedding_progress: bool = False,
                                   overwrite: bool = False) -> Optional[Tuple[str, str]]:
    """
    Embeds all tiles in a given slide_feature folder using a specified embedding model.

    This function processes each image tile in the slide_feature folder, extracts its features using
    the provided embedding model, and saves the features and their corresponding coordinates
    into an HDF5 file.

    Parameters:
    -----------
    slide_folder : str or Path
        Path to the folder containing tiled images of a whole slide_feature image (WSI).

    prefetch_loader: the prefetched tile_dataloader

    embedding_model_at_certain_GPU : torch.nn.Module
        Pretrained model to be used for extracting features from the image tiles.
    output_WSI_dataset_path: Path, optional
        Path to save the HDF5 file. If not specified, saves to the original slide_feature folder.
    batch_size : int, optional
        Number of image tiles to process in a batch (default is 256).
    shuffle : bool, optional
        Whether to shuffle the tiles before processing (default is False).
    num_workers : int, optional
        Number of subprocesses to use for data loading (default is 2).
    transform : torchvision.transforms.Compose, optional
        Transform to apply to each image tile (default is None).
    edge_size: int, optional
        Tile edge size (default is 224).
    suffix : str, optional
        Suffix of the image files (default is '.jpeg').
    device : str, optional
        Device to run the embedding model on (default is 'cuda').
    embedding_progress: bool, optional
        Whether to show a progress bar (default is False).
    overwrite: Whether to overwrite an existing output tiles dataset. If `True`,
        will delete previous file and recreate

    Returns:
    --------
    None or tuple
        Returns None if successful, or a tuple (slide_id, slide_folder) if an error occurs.
    """
    slide_id = os.path.basename(slide_folder)

    try:
        # Determine the output path for the HDF5 file
        if output_WSI_dataset_path is None:
            target_h5path = os.path.join(slide_folder, f'{slide_id}.h5')
        else:
            output_dir = os.path.join(output_WSI_dataset_path, slide_id)
            os.makedirs(output_dir, exist_ok=True)
            target_h5path = os.path.join(output_dir, f'{slide_id}.h5')

        # is_already_processed
        if os.path.exists(target_h5path):
            if overwrite:
                shutil.rmtree(target_h5path)
            else:
                logging.info(f">>> Skipping WSI: {slide_id} from {slide_folder} - h5 file already processed")
                return (slide_id, slide_folder)
        else:
            logging.info(f">>> Processing WSI: {slide_id} from {slide_folder}")

        since = time.time()
        embedding_model_at_certain_GPU.eval()  # Set model to evaluation mode
        # Process each batch of image tiles
        for data_iter_step, (patch_image_tensor, patch_coord_yx_tensor) \
                in tqdm(enumerate(prefetch_loader),
                        disable=not embedding_progress,
                        total=len(prefetch_loader),
                        unit="batch",
                        desc=f'Embedding slide_feature {slide_id} on batch of {batch_size} tiles'):
            patch_image_tensor = patch_image_tensor.to(device)  # Move tensor to device
            with torch.no_grad():  # No need for gradient computation during embedding
                patch_feature_tensor = embedding_model_at_certain_GPU(patch_image_tensor)  # Extract features

            # Save the features and coordinates to the HDF5 file
            for idx in range(patch_feature_tensor.shape[0]):
                hdf5_save_a_patch(target_h5path, patch_feature_tensor[idx].cpu().numpy(), patch_type='features')
                hdf5_save_a_patch_coord(target_h5path,
                                        coord_y=patch_coord_yx_tensor[idx][0].item(),
                                        coord_x=patch_coord_yx_tensor[idx][1].item())

        time_elapsed = time.time() - since
        logging.info(f'slide_id: {slide_id}, embedding completed in {time_elapsed:.2f} seconds')

    except Exception as e:
        logging.error(f"Error processing slide_feature {slide_id}: {e}")
        return (slide_id, slide_folder)  # Return error information

    return None  # Return None if successful


# Function to embed slides for a specific device
def embed_at_device(device, model_name, edge_size, model_weight_path, device_slide_folders,
                    PrefetchDataLoader_num_workers, num_workers, output_WSI_dataset_path,
                    batch_size, overwrite, output_queue):
    """
    this func is for one GPU to embed multiple wsi folders

    the key design is calling a dataloader to pop tile dataloaders in parallel
    (with PrefetchDataLoader_num_workers for parallel)
    """
    # Setup logging in each subprocess
    log_file_path = Path(output_WSI_dataset_path) / f'log_{device}.log'
    setup_logging(log_file_path)

    # build Patch_embedding_model
    embedding_model = Get_ROI_model.get_model(num_classes=0, edge_size=edge_size,
                                              model_idx=model_name, pretrained_backbone=model_weight_path)
    compiled_model = torch.compile(embedding_model)
    embedding_model_at_certain_GPU = compiled_model.to(device)

    # build a Prefetch dataloader to pop tile dataloders
    embedding_num_workers = (num_workers - PrefetchDataLoader_num_workers) // PrefetchDataLoader_num_workers
    PrefetchDataLoaders_dataset = PrefetchDataLoader_dataset(device_slide_folders,
                                                             dataset_builder=TileEncodingDataset,
                                                             edge_size=edge_size, embedding_batch_size=batch_size,
                                                             embedding_num_workers=embedding_num_workers,
                                                             pin_memory=True)

    PrefetchDataLoader = DataLoader(PrefetchDataLoaders_dataset,
                                    batch_size=1,  # every time return one dataloader
                                    shuffle=False, num_workers=PrefetchDataLoader_num_workers,
                                    drop_last=False, pin_memory=False,
                                    collate_fn=prefetch_dataloader_collate_fn)  # to support pop dataloader

    error_wsi_infor_list_at_device = []
    # iterating through Prefetched tile_dataloders
    for slide_id, slide_folder, prefetch_loader in tqdm(PrefetchDataLoader,
                                                        desc=f'Embedding slides on GPU:{device}', unit="wsi"):

        logging.info(f'Processing {len(prefetch_loader)} tiles from slide_feature {slide_id}')

        error_wsi_infor = embedding_one_slide_from_tiles(
            slide_folder, prefetch_loader, embedding_model_at_certain_GPU, output_WSI_dataset_path,
            batch_size=batch_size, device=device, embedding_progress=False, overwrite=overwrite)

        if error_wsi_infor:
            error_wsi_infor_list_at_device.append(error_wsi_infor)
    output_queue.put(error_wsi_infor_list_at_device)


def embedding_all_slides_from_tiles_dataset(input_tile_WSI_dataset_path, output_WSI_dataset_path,
                                            model_name, model_weight_path, PrefetchDataLoader_num_workers=2,
                                            batch_size=256, edge_size=224,
                                            overwrite=False):
    """
    Embeds all slides in the given root_path using the specified model and saves the embeddings.
    the embedding is running parallel per GPU


    Arguments:
    ----------
    input_raw_WSI_dataset_path : str
        Path to the root directory containing slide_feature folders.
    output_WSI_dataset_path : str
        Path to the directory where the embedded slides will be saved.

    model_name : str
        Name of the model to be used for embedding.
    model_weight_path : str
        Path to the pretrained model weights.

    PrefetchDataLoader_num_workers: int, default 2, the preloading fetching loder parallel num

    batch_size : int, optional
        Number of image tiles to process in a batch (default is 256).
    edge_size : int, optional
        Size of the edge of the image tiles (default is 224).

    overwrite: Whether to overwrite an existing output tiles dataset. If `True`,
        will delete previous file and recreate
    returns:
    ----------
    a list of (slide_id, slide_folder) if the slide_feature encounter error in the embedding process
    """
    if not os.path.exists(output_WSI_dataset_path):
        os.makedirs(output_WSI_dataset_path)

    # Configure logging
    main_log_file = Path(output_WSI_dataset_path) / 'wsi_tile_embedding.log'
    logging.basicConfig(filename=main_log_file, level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s')

    multiprocessing.set_start_method('spawn', force=True)

    since = time.time()
    logging.info(f'Embedding all_slides_from_tiles_dataset at {input_tile_WSI_dataset_path}')
    logging.info(f'Embedding output dataset folder at {output_WSI_dataset_path}')

    # List of available devices (GPUs), if no GPU is available, use 'cpu'
    if torch.cuda.is_available():
        device_list = [f'cuda:{i}' for i in range(torch.cuda.device_count())]
    else:
        device_list = ['cpu']

    num_workers = max(1, (multiprocessing.cpu_count() - len(device_list)) // len(device_list))
    # Number of CPU cores per GPU

    slide_dataset = Slide_loading_Dataset(input_tile_WSI_dataset_path)
    slide_path_dict = slide_dataset.slide_paths  # fixme change to num works for parallel loading slides?

    # Split slide_feature paths among available devices
    slide_folders = list(slide_path_dict.items())
    random.shuffle(slide_folders)  # Randomly shuffle the slides
    split_slide_folders = [slide_folders[i::len(device_list)] for i in range(len(device_list))]

    # Use multiprocessing to parallelly process slides on each device
    processes = []
    output_queue = multiprocessing.Queue()

    for device_index, device_slide_folders in enumerate(split_slide_folders):
        device = device_list[device_index]
        p = multiprocessing.Process(target=embed_at_device,
                                    args=(device, model_name, edge_size, model_weight_path, device_slide_folders,
                                          PrefetchDataLoader_num_workers, num_workers, output_WSI_dataset_path,
                                          batch_size, overwrite, output_queue))
        p.start()
        processes.append(p)

    # Join processes to ensure all embeddings are completed
    device_combined_error_wsi_infor_list = []
    for p in processes:
        p.join()
        device_combined_error_wsi_infor_list.extend(output_queue.get())

    error_wsi_infor_list = []
    for error_info in device_combined_error_wsi_infor_list:
        if error_info:
            logging.error(f"Error embedding slide_feature: {error_info}")
            error_wsi_infor_list.append(error_info)

    # Merge logs from subprocesses into the main log file
    for device_index, device in enumerate(device_list):
        log_file_path = Path(output_WSI_dataset_path) / f'log_{device}.log'
        if log_file_path.exists():
            with open(log_file_path, 'r') as log_file:
                log_content = log_file.read()
            with open(main_log_file, 'a') as main_log:
                main_log.write(log_content)
            os.remove(log_file_path)

    time_elapsed = time.time() - since
    logging.info(f'Embedding for all slides completed in {time_elapsed:.2f} seconds')
    logging.info(f'Embedding output dataset folder at {output_WSI_dataset_path}')
    logging.info(f'error_wsi_infor_list is {error_wsi_infor_list}')

    # return the combined error_wsi_infor_list from all gpus (skip None in the list)
    return error_wsi_infor_list


# functions for cropping and embedding the tiles from original slides
def crop_and_embed_one_slide(sample: Dict["SlideKey", Any],
                             embedding_model: torch.nn.Module,
                             output_dir: Path, thumbnail_dir: Optional[Path] = None,
                             batch_size: int = 32,
                             shuffle: bool = False,
                             num_workers: int = 1,
                             transform: Optional[transforms.Compose] = None,
                             device: str = 'cuda',
                             margin: int = 0, tile_size: int = 224, target_mpp: float = 0.5,
                             foreground_threshold: Optional[float] = None, occupancy_threshold: float = 0.1,
                             pixel_std_threshold: int = 5, extreme_value_portion_th: float = 0.5,
                             chunk_scale_in_tiles: int = 0,
                             tile_progress: bool = False,
                             image_key: str = "slide_image_path",
                             ROI_image_key='tile_image_path',
                             overwrite: bool = False) -> str:
    """Load and process a slide_feature, saving tile images and information to a CSV file.

    sample: dict
        Slide information dictionary, returned by the input slide_feature dataset.
    embedding_model: torch.nn.Module
        Pretrained model to be used for extracting features from the image tiles.
    task_settings_path: Path
        Root directory for the output dataset; outputs for a single slide_feature will be saved inside `task_settings_path/slide_id/`.
    thumbnail_dir: Optional[Path], optional
        Root directory for all thumbnails.
    batch_size: int, optional
        Number of image tiles to process in a batch (default is 256).
    shuffle: bool, optional
        Whether to shuffle the tiles before processing (default is False).
    num_workers: int, optional
        Number of subprocesses to use for data loading (default is 1 for not multiple processing).
    transform: torchvision.transforms.Compose, optional
        Transform to apply to each image tile (default is None).
    device: str, optional
        Device to run the embedding model on (default is 'cuda').
    margin: int, optional
        Margin around the foreground bounding box, in pixels at lowest resolution.
    tile_size: int, optional
        Lateral dimensions of each tile, in pixels (default is 224).
    target_mpp: float, optional
        Target microns per pixel for the slide_feature (default is 0.5).
    foreground_threshold: Optional[float], optional
        Luminance threshold (0 to 255) to determine if a pixel is foreground.
    occupancy_threshold: float, optional
        Threshold (between 0 and 1) to determine empty tiles to discard (default is 0.1).
    pixel_std_threshold: int, optional
        The threshold for the pixel variance at one ROI to say this ROI image is too 'empty'.
    extreme_value_portion_th: float, optional
        The threshold for the ratio of the pixels being 0 of one ROI, to say this ROI image is too 'empty'.
    chunk_scale_in_tiles: int, optional
        To speed up the I/O for loading the WSI regions.
    tile_progress: bool, optional
        Whether to display a progress bar in the terminal.
    image_key: str, optional
        Image key in the input and output dictionaries (default is 'slide_image_path').
    ROI_image_key: str, optional
        ROI Image key in the input and output dictionaries (default is 'tile_image_path').
    overwrite: bool, optional
        Whether to overwrite existing files (default is False).

    Returns:
    --------
    slide_feature id if the process raise error, else None
    """
    # STEP 0: set up path and log files
    slide_id: str = sample["slide_id"]
    slide_image_path = Path(sample[image_key])
    slide_folder = os.path.split(slide_image_path)[0]
    # Determine the output path for the HDF5 file
    os.makedirs(output_dir, exist_ok=True)
    output_slide_folder = os.path.join(output_dir, slide_id)
    os.makedirs(output_slide_folder, exist_ok=True)
    target_h5path = os.path.join(output_slide_folder, f'{slide_id}.h5')

    thumbnail_dir = Path(thumbnail_dir or output_dir)
    thumbnail_dir.mkdir(parents=True, exist_ok=True)

    # is_already_processed
    if os.path.exists(target_h5path):
        if overwrite:
            shutil.rmtree(output_slide_folder)
            os.makedirs(output_slide_folder, exist_ok=True)
        else:
            logging.info(f">>> Skipping WSI: {slide_id} from {slide_folder} "
                         f"- h5 file already processed at output_slide_folder {output_slide_folder}")
            return None  # Return None if successful

    try:
        # STEP 1: take the WSI and get the ROIs (valid tissue regions)
        logging.info(f"Loading slide_feature {slide_id} ...\nFile: {slide_image_path}")

        # take the valid tissue regions (ROIs) of the WSI (with monai and OpenSlide loader)
        loader = Loader_for_get_one_WSI_sample(WSIReader(backend="OpenSlide"), image_key=image_key,
                                               target_mpp=target_mpp, margin=margin,
                                               foreground_threshold=foreground_threshold,
                                               thumbnail_dir=thumbnail_dir)
        WSI_image_obj, loaded_ROI_samples = loader(sample)

        # process in a temp file
        with tempfile.TemporaryDirectory() as output_tiles_dir:
            # STEP 2: prepare log files:
            # prepare a csv log file for ROI tiles
            keys_to_save = ("slide_id", ROI_image_key, "tile_id", "label",
                            "tile_y", "tile_x", "occupancy")
            # Decode the slide_feature metadata (if got)
            slide_metadata: Dict[str, Any] = loaded_ROI_samples[0]["metadata"]
            metadata_keys = tuple("slide_" + key for key in slide_metadata)
            # print('metadata_keys',metadata_keys)
            csv_columns: Tuple[str, ...] = (*keys_to_save, *metadata_keys)
            # print('csv_columns',csv_columns)

            # build a recording file to log the processed files
            dataset_csv_path = output_tiles_dir / "dataset.csv"
            dataset_csv_file = dataset_csv_path.open('w')
            dataset_csv_file.write(','.join(csv_columns) + '\n')  # write CSV header

            failed_tiles_csv_path = output_tiles_dir / "failed_tiles.csv"
            failed_tiles_file = failed_tiles_csv_path.open('w')
            failed_tiles_file.write('tile_id' + '\n')  # write CSV header

            # STEP 3: Tile (crop) the WSI into ROI tiles (patches), save into output_tiles_dir
            logging.info(f"Tiling slide_feature {slide_id} ...")
            # each ROI_sample in loaded_WSI_samples is a valid ROI region
            n_failed_tiles = 0

            for index, ROI_sample in enumerate(loaded_ROI_samples):
                # The estimated luminance (foreground threshold) for WSI is applied to ROI here to filter the tiles
                tile_info_list, n_failed_tile = extract_valid_tiles(slide_image_path, ROI_sample,
                                                                    Path(output_tiles_dir),
                                                                    tile_size=tile_size,
                                                                    foreground_threshold=ROI_sample[
                                                                        "foreground_threshold"],
                                                                    occupancy_threshold=occupancy_threshold,
                                                                    pixel_std_threshold=pixel_std_threshold,
                                                                    extreme_value_portion_th=extreme_value_portion_th,
                                                                    chunk_scale_in_tiles=chunk_scale_in_tiles,
                                                                    tile_progress=tile_progress,
                                                                    ROI_image_key=ROI_image_key,
                                                                    num_workers=num_workers,
                                                                    log_file_elements=(dataset_csv_file,
                                                                                       failed_tiles_file,
                                                                                       keys_to_save,
                                                                                       metadata_keys))

                # STEP 4: visualize the tile location overlay to WSI
                visualize_tile_locations(ROI_sample, thumbnail_dir / (slide_image_path.name
                                                                      + "_roi_" + str(index) + "_tiles.jpeg"),
                                         tile_info_list, image_key=image_key)
                n_failed_tiles += n_failed_tile
            # close csv logging
            dataset_csv_file.close()
            failed_tiles_file.close()

            # STEP 5 : embedding on-fly
            # Create the dataset and dataloader
            tile_dataset = TileEncodingDataset(output_tiles_dir, transform=transform, edge_size=tile_size)
            tile_dataloader = DataLoader(tile_dataset, batch_size=batch_size, shuffle=shuffle,
                                         num_workers=num_workers, drop_last=False)

            since = time.time()

            logging.info(f'Embedding {len(tile_dataset)} tiles from slide_feature {slide_id}')
            embedding_model.eval()

            # Process each batch of image tiles
            for data_iter_step, (patch_image_tensor, patch_coord_yx_tensor) \
                    in tqdm(enumerate(tile_dataloader),
                            disable=not tile_progress,
                            total=len(tile_dataloader),
                            unit="batch",
                            desc=f'Embedding slide_feature {slide_id} on batch of {batch_size} tiles'):
                patch_image_tensor = patch_image_tensor.to(device)  # Move tensor to device
                with torch.no_grad():  # No need for gradient computation during embedding
                    patch_feature_tensor = embedding_model(patch_image_tensor)  # Extract features

                # Save the features and coordinates to the HDF5 file
                for idx in range(patch_feature_tensor.shape[0]):
                    hdf5_save_a_patch(target_h5path, patch_feature_tensor[idx].cpu().numpy(),
                                      patch_type='features')
                    hdf5_save_a_patch_coord(target_h5path,
                                            coord_y=patch_coord_yx_tensor[idx][0].item(),
                                            coord_x=patch_coord_yx_tensor[idx][1].item())

            time_elapsed = time.time() - since
            logging.info(f'slide_id: {slide_id}, embedding completed in {time_elapsed:.2f} seconds')

    except Exception as e:
        logging.error(f"Error processing slide_feature {slide_id}: {e}")
        return slide_id  # Return error WSI information

    else:
        if n_failed_tiles > 0:
            # what we want to do with slides that have some failed tiles? for now, just drop?
            logging.warning(f"{slide_id} is incomplete. {n_failed_tiles} tiles failed in reading.")

        logging.info(f"Finished processing slide_feature {slide_id}")

        # Explicitly delete the large objects
        del WSI_image_obj
        del loaded_ROI_samples
        del loader

        # Force garbage collection
        gc.collect()

        return None  # Return None if successful


def crop_and_embed_slides_at_device(device, model_name, model_weight_path, slide_folders, output_queue,
                                    output_WSI_dataset_path, batch_size, edge_size, overwrite, tile_progress,
                                    target_mpp, parallel=False, num_workers=1):
    # Setup logging in each subprocess
    log_file_path = Path(output_WSI_dataset_path) / f'log_{device}.log'
    setup_logging(log_file_path)

    # Initialize CUDA in the subprocess
    embedding_model = Get_ROI_model.get_model(num_classes=0, edge_size=edge_size,
                                              model_idx=model_name, pretrained_backbone=model_weight_path)
    # Patch_embedding_model(model_name=model_name, edge_size=edge_size, model_weight_path=model_weight_path)
    embedding_model = torch.compile(embedding_model)
    embedding_model = embedding_model.to(device)
    embedding_model.eval()

    error_wsi_infor_list_at_device = []

    for sample in tqdm(slide_folders, desc=f'Embedding slides on GPU:{device}', unit="wsi"):
        error_wsi_infor = None
        if parallel:
            # cpu_pool_size_for_each_device
            pass
            # TODO: Implement parallel processing for slides assigned to one GPU
        else:
            error_wsi_infor = crop_and_embed_one_slide(sample, embedding_model,
                                                       output_dir=output_WSI_dataset_path,
                                                       thumbnail_dir=output_WSI_dataset_path,
                                                       batch_size=batch_size, shuffle=False,
                                                       num_workers=1, transform=None, tile_size=edge_size,
                                                       target_mpp=target_mpp, device=device,
                                                       chunk_scale_in_tiles=4,
                                                       tile_progress=tile_progress, overwrite=overwrite)
        if error_wsi_infor:
            error_wsi_infor_list_at_device.append(error_wsi_infor)
    output_queue.put(error_wsi_infor_list_at_device)


def embedding_all_slides_from_slides(input_raw_WSI_dataset_path: Union[str, Path],
                                     output_WSI_dataset_path: Union[str, Path],
                                     model_name: str, model_weight_path: str, target_mpp: float,
                                     batch_size: int = 32, edge_size: int = 224,
                                     overwrite: bool = False, parallel: bool = False) -> List[Optional[str]]:
    """
    Embed all slides from the given input dataset path and save the outputs to the specified output path.

    Parameters
    ----------
    input_raw_WSI_dataset_path : Union[str, Path]
        Path to the input dataset containing the tiles of WSIs.
    output_WSI_dataset_path : Union[str, Path]
        Path to save the output embedded tiles.
    model_name : str
        Name of the pre-trained model to use for embedding.
    model_weight_path : str
        Path to the pre-trained model weights.
    target_mpp : float
        0.5 for prov-gigapath
    batch_size : int, optional
        Batch size for processing tiles (default is 32).
    edge_size : int, optional
        Edge size for the tiles (default is 224).
    overwrite : bool, optional
        Whether to overwrite existing files (default is False).
    parallel : bool, optional
        Whether to run the embedding in parallel across multiple devices (default is False).

    Returns
    -------
    List[Optional[str]]
        List of slide_feature IDs that encountered errors during processing.
    """
    if not os.path.exists(output_WSI_dataset_path):
        os.makedirs(output_WSI_dataset_path)

    # Configure logging
    main_log_file = Path(output_WSI_dataset_path) / 'wsi_tile_embedding.log'
    logging.basicConfig(filename=main_log_file, level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s')

    multiprocessing.set_start_method('spawn', force=True)

    since = time.time()
    logging.info(f'Cropping and Embedding all_slides_from_tiles_dataset at {input_raw_WSI_dataset_path}')
    logging.info(f'Cropping and Embedding output dataset folder at {output_WSI_dataset_path}')

    tile_progress = not parallel

    device_list = [f'cuda:{i}' for i in range(torch.cuda.device_count())] if torch.cuda.is_available() else ['cpu']
    slide_folders = prepare_slides_sample_list(slide_root=input_raw_WSI_dataset_path)
    random.shuffle(slide_folders)
    split_slide_folders = [slide_folders[i::len(device_list)] for i in range(len(device_list))]

    processes = []
    output_queue = multiprocessing.Queue()
    num_workers = (multiprocessing.cpu_count() - len(device_list)) // len(device_list)

    for device_index, device_slide_folders in enumerate(split_slide_folders):
        device = device_list[device_index]
        p = multiprocessing.Process(target=crop_and_embed_slides_at_device,
                                    args=(device, model_name, model_weight_path, device_slide_folders,
                                          output_queue, output_WSI_dataset_path, batch_size, edge_size,
                                          overwrite, tile_progress, target_mpp, parallel, num_workers))
        p.start()
        processes.append(p)

    device_combined_error_wsi_infor_list = []
    for p in processes:
        p.join()
        device_combined_error_wsi_infor_list.extend(output_queue.get())

    error_wsi_infor_list = []
    for error_info in device_combined_error_wsi_infor_list:
        if error_info:
            logging.error(f"Error embedding slide_feature: {error_info}")
            error_wsi_infor_list.append(error_info)

    # Merge logs from subprocesses into the main log file
    for device_index, device in enumerate(device_list):
        log_file_path = Path(output_WSI_dataset_path) / f'log_{device}.log'
        if log_file_path.exists():
            with open(log_file_path, 'r') as log_file:
                log_content = log_file.read()
            with open(main_log_file, 'a') as main_log:
                main_log.write(log_content)
            os.remove(log_file_path)

    time_elapsed = time.time() - since
    logging.info(f'Cropping and Embedding for all slides completed in {time_elapsed:.2f} seconds')
    logging.info(f'Cropping and Embedding output dataset folder at {output_WSI_dataset_path}')
    logging.info(f'error_wsi_infor_list is {error_wsi_infor_list}')

    return error_wsi_infor_list


def main(args):
    # TODO: add args.target_mpp

    if args.crop_on_fly:
        embedding_all_slides_from_slides(
            input_raw_WSI_dataset_path=args.WSI_dataset_path,
            output_WSI_dataset_path=args.embedded_WSI_dataset_path,
            model_name=args.model_name,
            model_weight_path=args.model_weight_path,
            target_mpp=args.target_mpp,
            batch_size=args.batch_size, edge_size=args.edge_size,
            overwrite=args.overwrite, parallel=True)
    else:
        embedding_all_slides_from_tiles_dataset(
            input_tile_WSI_dataset_path=args.WSI_dataset_path,
            output_WSI_dataset_path=args.embedded_WSI_dataset_path,
            model_name=args.model_name,
            model_weight_path=args.model_weight_path,
            PrefetchDataLoader_num_workers=args.PrefetchDataLoader_num_workers,
            batch_size=args.batch_size, edge_size=args.edge_size,
            overwrite=args.overwrite)  # , parallel=True


def get_args_parser():
    """Input parameters
    """
    parser = argparse.ArgumentParser(description='Build split and task configs.')

    parser.add_argument('--WSI_dataset_path', type=str,
                        default='/data/hdd_1/BigModel/qupath_tiles_datasets',
                        help='Root path for the datasets')
    parser.add_argument('--embedded_WSI_dataset_path', type=str,
                        default='/data/hdd_1/BigModel/qupath_embedded_datasets/',
                        help='Root path for the datasets')

    parser.add_argument('--model_name', type=str, default='gigapath',
                        help='name of the embedding model')
    parser.add_argument('--model_weight_path', type=str, default=None,
                        help='path of the embedding model weight')

    parser.add_argument('--edge_size', type=int, default=224,
                        help='edge size of tile')

    parser.add_argument('--batch_size', type=int, default=256,
                        help='batch_size of embedding model')

    parser.add_argument('--target_mpp', type=float, default=0.5,
                        help='target_mpp')

    parser.add_argument('--PrefetchDataLoader_num_workers', type=int, default=10,
                        help='PrefetchDataLoader_num_workers')

    parser.add_argument('--overwrite', action='store_true',
                        help='overwrite previous embedding at the path')

    parser.add_argument('--crop_on_fly', action='store_true',
                        help='overwrite previous embedding at the path')

    return parser


if __name__ == '__main__':
    '''
    # for processing tiles dataset
    # demo with one sample
    
    dataset = Slide_loading_Dataset(root_path='/data/hdd_1/BigModel/sampled_tiles_datasets')
    slide_folder = dataset.slide_paths[dataset.labeled_slide_names[0]]
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    embedding_model = Patch_embedding_model(model_name='ViT', pretrained_weight='timm')
    embedding_model_at_certain_GPU = embedding_model.to(device)

    embedding_one_slide_from_tiles(slide_folder, embedding_model_at_certain_GPU,
                                   output_WSI_dataset_path='/data/hdd_1/BigModel/sampled_embedded_datasets',
                                   batch_size=256, shuffle=False, num_workers=20,
                                   transform=None, suffix='.jpeg', device=device, embedding_progress=True)
                                   
    embedding_all_slides_from_tiles_dataset(input_raw_WSI_dataset_path='/data/hdd_1/BigModel/tiles_datasets',
                                            output_WSI_dataset_path='/data/hdd_1/BigModel/embedded_datasets',
                                            model_name='gigapath', model_weight_path='timm', batch_size=256,
                                            edge_size=224,
                                            overwrite=True)                               
    # demo with multiple sample
    embedding_all_slides_from_tiles_dataset(input_raw_WSI_dataset_path='/data/hdd_1/BigModel/sampled_tiles_datasets/',
                                            output_WSI_dataset_path='/data/hdd_1/BigModel/sampled_embedded_datasets/',
                                            model_name='gigapath', model_weight_path=None, batch_size=256,
                                            edge_size=224, overwrite=True)
    '''

    '''
    # for processing slide_feature dataset directly from slides
    # demo with one sample
    slide_folders = prepare_slides_sample_list(slide_root='/data/hdd_1/ai4dd/metadata/TCGA-READ/raw_data_sample')
    sample = slide_folders[3]
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    output_WSI_dataset_path = '/data/hdd_1/BigModel/sampled_embedded_datasets'

    embedding_model = Patch_embedding_model(model_name='ViT', pretrained_weight='timm')
    embedding_model_at_certain_GPU = embedding_model.to(device)

    error_wsi_infor = crop_and_embed_one_slide(sample, embedding_model_at_certain_GPU,
                                               task_settings_path=output_WSI_dataset_path,
                                               thumbnail_dir=output_WSI_dataset_path,
                                               batch_size=32, shuffle=False,
                                               num_workers=10, transform=None,
                                               device=device,
                                               chunk_scale_in_tiles=4,
                                               tile_progress=True, overwrite=True)
    
    # demo with multiple sample
    embedding_all_slides_from_slides(input_raw_WSI_dataset_path='/data/hdd_1/ai4dd/metadata/TCGA-READ/raw_data_sample',
                                     output_WSI_dataset_path='/data/hdd_1/BigModel/sampled_embedded_datasets',
                                     model_name='gigapath', model_weight_path='timm', overwrite=True, parallel=False)
    '''

    parser = get_args_parser()
    args = parser.parse_args()
    main(args)
