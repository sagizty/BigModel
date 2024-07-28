"""
WSI embedding dataset tools   Script  ver： July 28th 10:00


"""
import sys
import os

# Add the parent directory to the system path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'ModelBase')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'ModelBase', 'ROI_models')))
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'ModelBase', 'gigapath')))

import h5py
import torch
import logging
import time
import random
import shutil
import pandas as pd
import numpy as np
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

try:
    from ..ModelBase.ROI_models.VPT_ViT_modules import build_ViT_or_VPT
    from ..ModelBase.gigapath.Inference_pipeline import load_tile_slide_encoder
except:
    try:
        from VPT_ViT_modules import build_ViT_or_VPT
        from Inference_pipeline import load_tile_slide_encoder
    except ModuleNotFoundError:
        sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'ModelBase', 'ROI_models')))
        # sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'ModelBase', 'gigapath')))
        from VPT_ViT_modules import build_ViT_or_VPT
        # from Inference_pipeline import load_tile_slide_encoder


# datasets for embedding step or loading embedded slide_level datasets

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

        self.transform = transform or default_transform  # specified patch image Transform can be assigned

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
        patch_coord_yx_tensor = torch.tensor([y, x], dtype=torch.int8)
        # Load the image
        with open(img_path, "rb") as f:
            patch_image_tensor = Image.open(f).convert("RGB")  # Prepare tile in [H, W, C] int8 RGB
            if self.transform:
                patch_image_tensor = self.transform(patch_image_tensor)

        return patch_image_tensor, patch_coord_yx_tensor


# dataset for loading the slides (cropped / embedded)
class Slide_loading_Dataset(Dataset):
    def __init__(self, root_path: str, possible_suffixes: tuple = ('.h5', '.pt', '.jpeg', '.jpg'),
                 stopping_folder_name_list: list = ['thumbnails', ]):
        '''
        This class is used to set up the slide dataset for loading their slide_folder
        this can be used:
            1. for slide level dataset (loading wsis from different folder and have a path list)
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

        possible_suffixes: tuple = ('.h5', '.pt', '.jpeg', '.jpg') recording the settings

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


# todo Embedded slide dataset for WSI tasks
class SlideDataset(Slide_loading_Dataset):
    def __init__(self,
                 data_df: pd.DataFrame,
                 root_path: str,
                 splits: list,
                 task_config: dict,
                 slide_id_key='slide_id',
                 split_target_key='pat_id',
                 **kwargs):
        '''
        The slide dataset class for retrieving the slide sample for different tasks

        Arguments:
        ----------
        data_df: pd.DataFrame
            The dataframe that contains the slide sample, from a csv file of slide and labels
        root_path: str
            The root path of the tile embeddings

        task_config_path: dict
            The task configuration dictionary
                task configuration is a json file with the following structure:
                        'setting': ['task_name_1', 'task_name_2',...] a list of target task
                        'label_dict': a list of encoding rules for MTL fixme check with previous design

        splits: list
            The list of patient_ids/slide_ids as the split lists to build dataset

        slide_id_key: str
            The key that contains the slide id
        split_target_key: str
            The key that specifies the column name for taking the splits
        '''
        super(SlideDataset, self).__init__(**kwargs)

        self.root_path = root_path

        self.task_cfg = task_config

        self.split_target_key = split_target_key
        self.slide_id_key = slide_id_key

        # get slides that have tile encodings
        self.embedded_slide_paths = {}
        valid_slide_ids = self.get_valid_slides(self.root_path, data_df[self.slide_id_key].values)
        # filter out slides that do not have tile encodings
        data_df = data_df[data_df[self.slide_id_key].isin(valid_slide_ids)]
        # set up the task (fixme 'label' is for example should change to certain specific task name)
        self.setup_task_data(data_df, splits, task_config.get('setting', 'label'))

        # load from settings or set default value
        self.max_tiles = task_config.get('max_tiles', 1000)
        self.shuffle_tiles = task_config.get('shuffle_tiles', False)
        print('Dataset has been initialized!')

    def get_valid_slides(self, root_path: str, slide_ids: list) -> list:
        '''This function is used to get the slides that have tile encodings stored in the tile directory'''
        valid_slide_ids = []
        for slide_id in slide_ids:

            if 'pt_files' in root_path.split('/')[-1]:
                embedded_slide_file = slide_id.replace(".svs", "") + '.pt'
            else:
                embedded_slide_file = slide_id.replace(".svs", "") + '.h5'

            embedded_slide_path = os.path.join(self.slide_paths[slide_id], embedded_slide_file)
            if not os.path.exists(embedded_slide_path):
                print('Missing: ', embedded_slide_path)
            else:
                # add to valid list
                valid_slide_ids.append(slide_id)
                self.embedded_slide_paths[slide_id] = embedded_slide_path

        return valid_slide_ids

    def prepare_single_task_data_list(self, df: pd.DataFrame, splits: list, task_name_list=['label', ]):
        '''Prepare the sample for single task'''
        task_name = task_name_list[0]

        # set up the label_dict
        label_dict_settings = self.task_cfg.get('label_dict', {})  # one-encoding dict
        assert label_dict_settings, 'No label_dict found in the task configuration'
        label_dict = label_dict_settings[task_name]

        # set up the mappings
        assert task_name in df.columns, 'No label column found in the dataframe'
        df[task_name] = df[task_name].map(label_dict)
        n_classes = len(label_dict)  # if 1, its regression task

        # get the corresponding splits
        assert self.split_target_key in df.columns, 'No {} column found in the dataframe'.format(self.split_target_key)
        df = df[df[self.split_target_key].isin(splits)]
        slide_ids = df[self.slide_id_key].to_list()
        if n_classes == 1:
            slide_labels = df[[task_name]].to_numpy().astype(int)  # manual long-int encoding in df[['label']]
        else:
            slide_labels = df[[task_name]].to_numpy()
        return df, slide_ids, slide_labels, n_classes

    # fixme change to Multiple task (currently its multiple-labeled single task)
    def prepare_multi_label_CLS_data_list(self, df: pd.DataFrame, splits: list, task_name_list: list):
        '''Prepare the sample for multi-label classification'''
        # todo make this func ready in later days
        # set up the label_dict
        label_dict = self.task_cfg.get('label_dict', {})
        assert label_dict, 'No label_dict found in the task configuration'
        # Prepare mutation sample
        label_keys = label_dict.keys()
        # sort key using values
        label_keys = sorted(label_keys, key=lambda x: label_dict[x])
        n_classes = len(label_dict)

        # get the corresponding splits
        assert self.split_target_key in df.columns, 'No {} column found in the dataframe'.format(self.split_target_key)
        df = df[df[self.split_target_key].isin(splits)]
        slide_ids = df[self.slide_id_key].to_list()
        labels = df[label_keys].to_numpy().astype(int)

        return df, slide_ids, labels, n_classes

    def setup_task_data(self, df: pd.DataFrame, splits: list, task_name_list: list):
        '''Prepare the sample for single task setting or single task setting'''
        # Prepare slide sample
        if len(task_name_list) == 1:
            prepare_data_func = self.prepare_single_task_data_list
        elif len(task_name_list) > 1:
            prepare_data_func = self.prepare_multi_label_CLS_data_list
        else:
            raise ValueError('Invalid task: {}'.format(task_name_list))
        self.slide_data, self.slide_ids, self.labels, self.n_classes = prepare_data_func(df, splits, task_name_list)
        # reset the embedded_slide_paths dict
        self.embedded_slide_paths = self.embedded_slide_paths[self.slide_ids]

    def shuffle_data_pairs(self, images: torch.Tensor, coords: torch.Tensor) -> tuple:
        '''Shuffle the serialized images and coordinates'''
        indices = torch.randperm(len(images))
        images_ = images[indices]
        coords_ = coords[indices]
        return images_, coords_

    def read_assets_from_h5(self, h5_path: str) -> tuple:
        '''Read the assets from the h5 file'''
        assets = {}
        attrs = {}
        with h5py.File(h5_path, 'r') as f:
            for key in f.keys():
                assets[key] = f[key][:]
                if f[key].attrs is not None:
                    attrs[key] = dict(f[key].attrs)
        return assets, attrs

    def get_slide_name_from_path(self, sld: str) -> str:
        '''Get the slide name from the slide path'''
        slide_name = os.path.basename(sld).split('.h5')[0]
        return slide_name

    def get_embedded_data_dict(self, embedding_file_path: str) -> dict:
        """Get the image_features from the path"""
        # fixme future abandon .pt?
        if '.pt' in embedding_file_path:
            image_features = torch.load(embedding_file_path)
            coords_yx = 0
        elif '.h5' in embedding_file_path:
            assets, _ = self.read_assets_from_h5(embedding_file_path)
            image_features = torch.from_numpy(assets['features'])
            coords_yx = torch.from_numpy(assets['coords_yx'])

            # if shuffle the sample
            if self.shuffle_tiles:
                image_features, coords_yx = self.shuffle_data_pairs(image_features, coords_yx)

            if image_features.size(0) > self.max_tiles:
                image_features = image_features[:self.max_tiles, :]
            if coords_yx.size(0) > self.max_tiles:
                coords_yx = coords_yx[:self.max_tiles, :]

        # set the input dict
        data_dict = {'image_features': image_features,
                     'image_features_lens': image_features.size(0),
                     'pad_mask': 0,  # It may be used for some model design
                     'coords_yx': coords_yx}
        return data_dict

    def get_one_embedded_sample(self, idx: int) -> dict:
        '''Get one sample from the dataset'''
        # get the slide id
        slide_id = self.slide_ids[idx]
        # get the slide path
        embedded_slide_path = self.embedded_slide_paths[slide_id]

        # get the slide tile embeddings
        data_dict = self.get_embedded_data_dict(embedded_slide_path)
        # get the slide label
        label = torch.from_numpy(self.labels[idx])
        # set the sample dict
        sample = {'image_features': data_dict['image_features'],
                  'image_features_lens': data_dict['image_features_lens'],
                  'pad_mask': data_dict['pad_mask'],
                  'coords_yx': data_dict['coords_yx'],
                  'slide_id': slide_id,
                  'labels': label}
        return sample

    def get_embedded_sample_with_try(self, idx, n_try=3):
        '''Get the sample with n_try,
        fixme this handels missing/ failed sample, but not nicely'''
        for _ in range(n_try):
            try:
                one_embedded_sample = self.get_one_embedded_sample(idx)
                return one_embedded_sample
            except:
                print('Error in getting the sample, try another index')
                idx = np.random.randint(0, len(self.slide_data))
        print('Error in getting one sample with n_try: ', n_try)
        raise  # Error in getting one sample with n_try

    def __len__(self):
        return len(self.slide_data)

    def __getitem__(self, idx):
        slide_level_sample = self.get_embedded_sample_with_try(idx)
        return slide_level_sample


# class for setting up embedding model
class Patch_embedding_model(nn.Module):
    """
    each tile is embedded to a token based on pre-trained model

    [batch, 3, edge, edge] -> [batch, D]

    """

    def __init__(self, model_name='18', edge_size=224, pretrained_weight=None, prompt_state_dict=None):

        super(Patch_embedding_model, self).__init__()
        self.edge_size = edge_size

        if pretrained_weight is not None and pretrained_weight != 'timm' and pretrained_weight != 'Timm':
            print("feature extractor backbone using weight at :", pretrained_weight)
            pretrained_weight = torch.load(pretrained_weight)
            pretrained_backbone = False
        else:
            print("feature extractor backbone weight is using timm")
            # default set the embedding to Timm
            pretrained_backbone = True
            pretrained_weight = None

        if type(model_name) is str:
            if model_name == '18':
                backbone = models.resnet18(pretrained=pretrained_backbone)
                backbone.fc = nn.Identity()
                self.backbone = backbone
                if pretrained_weight is not None:
                    self.backbone.load_state_dict(pretrained_weight, False)

            elif model_name == '34':
                backbone = models.resnet34(pretrained=pretrained_backbone)
                backbone.fc = nn.Identity()
                self.backbone = backbone
                if pretrained_weight is not None:
                    self.backbone.load_state_dict(pretrained_weight, False)

            elif model_name == '50':
                backbone = models.resnet50(pretrained=pretrained_backbone)
                backbone.fc = nn.Identity()
                self.backbone = backbone
                if pretrained_weight is not None:
                    self.backbone.load_state_dict(pretrained_weight, False)

            # VPT feature embedding
            elif model_name == 'VPT' or model_name == 'vpt':
                self.backbone = build_ViT_or_VPT(
                    num_classes=0,  # set to feature extractor model, output is CLS token
                    edge_size=224, model_idx='ViT', patch_size=16,
                    Prompt_Token_num=20, VPT_type="Deep",
                    prompt_state_dict=prompt_state_dict,
                    base_state_dict=pretrained_weight or 'timm')

            elif model_name == 'ViT' or model_name == 'vit':
                self.backbone = build_ViT_or_VPT(
                    num_classes=0,  # set to feature extractor model, output is CLS token
                    edge_size=224, model_idx='ViT', patch_size=16,
                    VPT_type=None, base_state_dict=pretrained_weight or 'timm')

            # GTP feature embedding resnet
            # Ref: Y. Zheng et al., “A Graph-Transformer for Whole Slide Image Classification,”
            elif model_name == 'gigapath':
                # fixme if failed, use your own hugging face token and register for the project gigapath
                os.environ["HF_TOKEN"] = "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"
                if pretrained_weight is not None:
                    tile_encoder = timm.create_model("hf_hub:prov-gigapath/prov-gigapath", pretrained=False,
                                                     checkpoint_path=pretrained_weight)
                else:
                    tile_encoder = timm.create_model("hf_hub:prov-gigapath/prov-gigapath", pretrained=True)
                print("Tile encoder param #", sum(p.numel() for p in tile_encoder.parameters()))
                self.backbone = tile_encoder

            else:
                raise NotImplementedError("This function is not yet implemented, model_name: " + str(model_name))
        else:
            self.backbone = model_name  # put for future input as a model

        # set to eval model, we don't need gradient for the fixed embedding
        self.backbone.eval()

    def forward(self, x):
        assert isinstance(self.backbone, nn.Module)
        # [batch, 3, edge, edge] -> [batch, self.embed_dim]
        x = self.backbone(x)
        return x


# functions for the tile embedding and prepare dataset for slide-level tasks
def embedding_one_slide(slide_folder: Union[str, Path],
                        embedding_model_at_certain_GPU: torch.nn.Module,
                        output_WSI_dataset_path: Optional[Union[str, Path]] = None,
                        batch_size: int = 256,
                        shuffle: bool = False,
                        num_workers: int = 2,
                        transform: Optional[transforms.Compose] = None,
                        edge_size: int = 224,
                        suffix: str = '.jpeg',
                        device: str = 'cuda',
                        embedding_progress: bool = False,
                        overwrite: bool = False) -> Optional[Tuple[str, str]]:
    """
    Embeds all tiles in a given slide folder using a specified embedding model.

    This function processes each image tile in the slide folder, extracts its features using
    the provided embedding model, and saves the features and their corresponding coordinates
    into an HDF5 file.

    Parameters:
    -----------
    slide_folder : str or Path
        Path to the folder containing tiled images of a whole slide image (WSI).
    embedding_model_at_certain_GPU : torch.nn.Module
        Pretrained model to be used for extracting features from the image tiles.
    output_WSI_dataset_path: Path, optional
        Path to save the HDF5 file. If not specified, saves to the original slide folder.
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

        # Create the dataset and dataloader
        tile_dataset = TileEncodingDataset(slide_folder, transform=transform, edge_size=edge_size, suffix=suffix)
        tile_dataloader = DataLoader(tile_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
                                     drop_last=False)

        since = time.time()
        embedding_model_at_certain_GPU.eval()  # Set model to evaluation mode

        logging.info(f'Processing {len(tile_dataset)} tiles from slide {slide_id}')

        # Process each batch of image tiles
        for data_iter_step, (patch_image_tensor, patch_coord_yx_tensor) \
                in tqdm(enumerate(tile_dataloader),
                        disable=not embedding_progress,
                        total=len(tile_dataloader),
                        unit="batch",
                        desc=f'Embedding slide {slide_id} on batch of {batch_size} tiles'):
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
        logging.error(f"Error processing slide {slide_id}: {e}")
        return (slide_id, slide_folder)  # Return error information

    return None  # Return None if successful


def embedding_all_slides(input_tile_WSI_dataset_path, output_WSI_dataset_path,
                         model_name, model_weight_path, batch_size=256, edge_size=224,
                         overwrite=False):
    """
    Embeds all slides in the given root_path using the specified model and saves the embeddings.
    the embedding is running parallel per GPU


    Arguments:
    ----------
    input_tile_WSI_dataset_path : str
        Path to the root directory containing slide folders.
    output_WSI_dataset_path : str
        Path to the directory where the embedded slides will be saved.

    model_name : str
        Name of the model to be used for embedding.
    model_weight_path : str
        Path to the pretrained model weights.

    batch_size : int, optional
        Number of image tiles to process in a batch (default is 256).
    edge_size : int, optional
        Size of the edge of the image tiles (default is 224).

    overwrite: Whether to overwrite an existing output tiles dataset. If `True`,
        will delete previous file and recreate
    returns:
    ----------
    a list of (slide_id, slide_folder) if the slide encounter error in the embedding process
    """
    if not os.path.exists(output_WSI_dataset_path):
        os.makedirs(output_WSI_dataset_path)

    slide_dataset = Slide_loading_Dataset(input_tile_WSI_dataset_path)
    slide_path_dict = slide_dataset.slide_paths

    # List of available devices (GPUs), if no GPU is available, use 'cpu'
    if torch.cuda.is_available():
        device_list = [f'cuda:{i}' for i in range(torch.cuda.device_count())]
    else:
        device_list = ['cpu']

    num_workers = max(1, multiprocessing.cpu_count() // len(device_list))  # Number of CPU cores per GPU

    embedding_model = Patch_embedding_model(model_name=model_name, edge_size=edge_size,
                                            pretrained_weight=model_weight_path)
    embedding_model_list = [embedding_model.to(device) for device in device_list]

    # Function to embed slides for a specific device
    def embed_at_device(device_index, device_list, embedding_model_list, slide_folders, output_queue):
        embedding_model_at_certain_GPU = embedding_model_list[device_index]
        error_wsi_infor_list_at_device = []
        for slide_id, slide_folder in tqdm(slide_folders,
                                           desc=f'Embedding slides on GPU:{device_list[device_index]}', unit="wsi"):
            error_wsi_infor = embedding_one_slide(
                slide_folder, embedding_model_at_certain_GPU, output_WSI_dataset_path,
                batch_size=batch_size, device=device_list[device_index], num_workers=num_workers,
                embedding_progress=False, overwrite=overwrite
            )
            if error_wsi_infor:
                error_wsi_infor_list_at_device.append(error_wsi_infor)
        output_queue.put(error_wsi_infor_list_at_device)

    # Split slide paths among available devices
    slide_folders = list(slide_path_dict.items())
    random.shuffle(slide_folders)  # Randomly shuffle the slides
    split_slide_folders = [slide_folders[i::len(device_list)] for i in range(len(device_list))]

    # Use multiprocessing to parallelly process slides on each device
    processes = []
    output_queue = multiprocessing.Queue()

    for device_index, device_slide_folders in enumerate(split_slide_folders):
        p = multiprocessing.Process(target=embed_at_device,
                                    args=(device_index, device_list, embedding_model_list, device_slide_folders,
                                          output_queue))
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
            logging.error(f"Error embedding slide: {error_info}")
            error_wsi_infor_list.append(error_info)

    # return the combined error_wsi_infor_list from all gpus (skip None in the list)
    return error_wsi_infor_list


if __name__ == '__main__':
    # Configure logging
    logging.basicConfig(filename='wsi_tile_embedding.log', level=logging.DEBUG,
                        format='%(asctime)s %(levelname)s:%(message)s')

    '''
    # demo with one sample 
    
    dataset = Slide_loading_Dataset(root_path='/data/hdd_1/BigModel/sampled_datasets')
    slide_folder = dataset.slide_paths[dataset.slide_ids[0]]
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    embedding_model = Patch_embedding_model(model_name='ViT', pretrained_weight='timm')
    embedding_model_at_certain_GPU = embedding_model.to(device)

    embedding_one_slide(slide_folder, embedding_model_at_certain_GPU,
                        output_WSI_dataset_path='/data/hdd_1/BigModel/sampled_embedded_datasets',
                        batch_size=256, shuffle=False, num_workers=20,
                        transform=None, suffix='.jpeg', device=device, embedding_progress=True)
    '''
    # demo with multiple sample
    embedding_all_slides(input_tile_WSI_dataset_path='/data/hdd_1/BigModel/sampled_tiles_datasets',
                         output_WSI_dataset_path='/data/hdd_1/BigModel/sampled_embedded_datasets',
                         model_name='ViT', model_weight_path='timm', batch_size=256, edge_size=224,
                         overwrite=True)
