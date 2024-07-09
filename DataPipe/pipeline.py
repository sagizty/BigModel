# --------------------------------------------------------
# Pipeline for processing the data

# type A is for (ROI+WSI approaches)
# type B is for (Cell+ROI+WSI approaches)
# --------------------------------------------------------
import os
import torch
import numpy as np
import pandas as pd
from PIL import Image
from pathlib import Path
from typing import List, Tuple, Union
from torch.utils.data import Dataset, DataLoader
import functools
import logging
import shutil
import tempfile
import traceback
import warnings
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import PIL
from matplotlib import collections, patches, pyplot as plt
from monai.data import Dataset
from monai.data.wsi_reader import WSIReader
from openslide import OpenSlide
from tqdm import tqdm

from .WSI_tools import *
from .Segmentation_and_filtering_tools import *

def generate_tiles(slide_image: np.ndarray, tile_size: int, foreground_threshold: float,
                   occupancy_threshold: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Split the foreground of an input slide image into tiles.

    :param slide_image: The RGB image array in (C, H, W) format.
    :param tile_size: Lateral dimensions of each tile, in pixels.
    :param foreground_threshold: Luminance threshold (0 to 255) to determine tile occupancy.
    here the foreground_threshold value is automatically get from WSI level estimation, and applied to its ROI
    :param occupancy_threshold: Threshold (between 0 and 1) to determine empty tiles to discard.
    :return: A tuple containing the image tiles (N, C, H, W), tile coordinates (N, 2), occupancies
    (N,), and total number of discarded empty tiles.
    """
    # STEP 1: crop WSI to generate tile sequence
    image_tiles, tile_locations = tile_array_2d(slide_image, tile_size=tile_size, constant_values=255)
    # log WSI/ROI information
    logging.info(f"image_tiles.shape: {image_tiles.shape}, dtype: {image_tiles.dtype}")
    logging.info(f"Tiled {slide_image.shape} to {image_tiles.shape}")

    # STEP 2: Filtering the ROIs with foreground ratio
    # generate the foreground mask for each ROI in the sequence, based on the value of luminance at each pixel
    foreground_mask, _ = segment_foreground(image_tiles, foreground_threshold)
    # filtering the ROI tiles
    selected, occupancies = ROI_occupancy_filtering(foreground_mask, occupancy_threshold)
    # log ROI selection infor
    n_discarded = (~selected).sum()
    logging.info(f"Percentage tiles discarded: {n_discarded / len(selected) * 100:.2f}")

    # STEP 3: Filtering the ROIs with variance and empty values
    # FIXME: this uses too much memory, and its hacky design needs attention
    empty_tile_bool_mask = check_empty_tiles(image_tiles)
    selected = selected & (~empty_tile_bool_mask)
    n_discarded = (~selected).sum()
    logging.info(f"Percentage tiles discarded after filtering empty tiles: {n_discarded / len(selected) * 100:.2f}")

    # log selection infor of WSI locations
    logging.info(f"Before filtering: min y: {tile_locations[:, 0].min()}, max y: {tile_locations[:, 0].max()}, min x: {tile_locations[:, 1].min()}, max x: {tile_locations[:, 1].max()}")

    image_tiles = image_tiles[selected]
    tile_locations = tile_locations[selected]
    occupancies = occupancies[selected]

    if len(tile_locations) == 0:
        logging.warn("No tiles selected")
    else:
        logging.info(f"After filtering: min y: {tile_locations[:, 0].min()}, max y: {tile_locations[:, 0].max()}, min x: {tile_locations[:, 1].min()}, max x: {tile_locations[:, 1].max()}")

    return image_tiles, tile_locations, occupancies, n_discarded



def process_one_slide(sample: Dict["SlideKey", Any], level: int, margin: int, tile_size: int,
                      foreground_threshold: Optional[float], occupancy_threshold: float, output_dir: Path,
                      thumbnail_dir: Path,
                      tile_progress: bool = False) -> str:
    """Load and process a slide, saving tile images and information to a CSV file.

    :param sample: Slide information dictionary, returned by the input slide dataset.
    :param level: Magnification level at which to process the slide.
    :param margin: Margin around the foreground bounding box, in pixels at lowest resolution.
    :param tile_size: Lateral dimensions of each tile, in pixels.
    :param foreground_threshold: Luminance threshold (0 to 255) to determine tile occupancy.
    If `None` (default), an optimal threshold will be estimated automatically.
    :param occupancy_threshold: Threshold (between 0 and 1) to determine empty tiles to discard.
    :param output_dir: Root directory for the output dataset; outputs for a single slide will be
    saved inside `output_dir/slide_id/`.
    :param tile_progress: Whether to display a progress bar in the terminal.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    thumbnail_dir.mkdir(parents=True, exist_ok=True)
    slide_metadata: Dict[str, Any] = sample["metadata"]
    keys_to_save = ("slide_id", "tile_id", "image", "label",
                    "tile_x", "tile_y", "occupancy")
    metadata_keys = tuple("slide_" + key for key in slide_metadata)
    csv_columns: Tuple[str, ...] = (*keys_to_save, *metadata_keys)
    print(csv_columns)
    slide_id: str = sample["slide_id"]
    rel_slide_dir = Path(slide_id)
    output_tiles_dir = output_dir / rel_slide_dir
    logging.info(f">>> Slide dir {output_tiles_dir}")
    if is_already_processed(output_tiles_dir):
        logging.info(f">>> Skipping {output_tiles_dir} - already processed")
        return output_tiles_dir

    else:
        output_tiles_dir.mkdir(parents=True, exist_ok=True)
        dataset_csv_path = output_tiles_dir / "dataset.csv"
        dataset_csv_file = dataset_csv_path.open('w')
        dataset_csv_file.write(','.join(csv_columns) + '\n')  # write CSV header

        n_failed_tiles = 0
        failed_tiles_csv_path = output_tiles_dir / "failed_tiles.csv"
        failed_tiles_file = failed_tiles_csv_path.open('w')
        failed_tiles_file.write('tile_id' + '\n')

        slide_image_path = Path(sample["image"])
        logging.info(f"Loading slide {slide_id} ...\nFile: {slide_image_path}")

        # Somehow it's very slow on Datarbicks
        # hack: copy the slide file to a temporary directory
        tmp_dir = tempfile.TemporaryDirectory()
        tmp_slide_image_path = Path(tmp_dir.name) / slide_image_path.name
        logging.info(f">>> Copying {slide_image_path} to {tmp_slide_image_path}")
        shutil.copy(slide_image_path, tmp_slide_image_path)
        sample["image"] = tmp_slide_image_path
        logging.info(f">>> Finished copying {slide_image_path} to {tmp_slide_image_path}")

        # Save original slide thumbnail
        save_thumbnail(slide_image_path, thumbnail_dir / (slide_image_path.name + "_original.png"))

        loader = LoadROId(WSIReader(backend="OpenSlide"), level=level, margin=margin,
                          foreground_threshold=foreground_threshold)
        sample = loader(sample)  # load 'image' from disk

        # Save ROI thumbnail
        slide_image = sample["image"]
        plt.figure()
        plt.imshow(slide_image.transpose(1, 2, 0))
        plt.savefig(thumbnail_dir / (slide_image_path.name + "_roi.png"))
        plt.close()
        logging.info(f"Saving thumbnail {thumbnail_dir / (slide_image_path.name + '_roi.png')}, shape {slide_image.shape}")

        logging.info(f"Tiling slide {slide_id} ...")
        image_tiles, rel_tile_locations, occupancies, _ = \
            generate_tiles(sample["image"], tile_size, sample["foreground_threshold"], occupancy_threshold)
        # It estimated luminance (foreground threshold) for whole WSI, and it is applied to ROI here

        # origin in level-0 coordinate
        # location in the current level coordiante
        # tile_locations in level-0 coordinate
        tile_locations = (sample["scale"] * rel_tile_locations
                            + sample["origin"]).astype(int)  # noqa: W503

        n_tiles = image_tiles.shape[0]
        logging.info(f"{n_tiles} tiles found")

        tile_info_list = []

        logging.info(f"Saving tiles for slide {slide_id} ...")
        for i in tqdm(range(n_tiles), f"Tiles ({slide_id[:6]}…)", unit="img", disable=not tile_progress):
            try:
                tile_info = get_tile_info(sample, occupancies[i], tile_locations[i], rel_slide_dir)
                tile_info_list.append(tile_info)

                save_image(image_tiles[i], output_dir / tile_info["image"])
                dataset_row = format_csv_row(tile_info, keys_to_save, metadata_keys)
                dataset_csv_file.write(dataset_row + '\n')
            except Exception as e:
                n_failed_tiles += 1
                descriptor = get_tile_descriptor(tile_locations[i])
                failed_tiles_file.write(descriptor + '\n')
                traceback.print_exc()
                warnings.warn(f"An error occurred while saving tile "
                                f"{get_tile_id(slide_id, tile_locations[i])}: {e}")

        dataset_csv_file.close()
        failed_tiles_file.close()

        # tile location overlay
        visualize_tile_locations(sample, thumbnail_dir / (slide_image_path.name + "_roi_tiles.png"), tile_info_list, tile_size, origin_offset=sample["origin"])

        if n_failed_tiles > 0:
            # TODO what we want to do with slides that have some failed tiles?
            logging.warning(f"{slide_id} is incomplete. {n_failed_tiles} tiles failed.")

        logging.info(f"Finished processing slide {slide_id}")

        return output_tiles_dir

# for pretraining
def main(slides_dataset: "SlidesDataset", root_output_dir: Union[str, Path],
         level: int, tile_size: int, margin: int, foreground_threshold: Optional[float],
         occupancy_threshold: float, parallel: bool = False, overwrite: bool = False,
         n_slides: Optional[int] = None) -> None:
    """Process a slides dataset to produce many folders of tiles dataset.

    :param slides_dataset: Input tiles dataset object.
    :param root_output_dir: The root directory of the output tiles dataset.
    :param level: Magnification level at which to process the slide.
    :param tile_size: Lateral dimensions of each tile, in pixels.
    :param margin: Margin around the foreground bounding box, in pixels at lowest resolution.
    :param foreground_threshold: Luminance threshold (0 to 255) to determine tile occupancy.
    If `None` (default), an optimal threshold will be estimated automatically.
    :param occupancy_threshold: Threshold (between 0 and 1) to determine empty tiles to discard.
    :param parallel: Whether slides should be processed in parallel with multiprocessing.
    :param overwrite: Whether to overwrite an existing output tiles dataset. If `True`, will delete
    and recreate `root_output_dir`, otherwise will resume by skipping already processed slides.
    :param n_slides: If given, limit the total number of slides for debugging.
    """

    # Ignoring some types here because mypy is getting confused with the MONAI Dataset class
    # to select a subsample use keyword n_slides
    dataset = Dataset(slides_dataset)[:n_slides]  # type: ignore

    # make sure all slide files exist in the image dir
    for sample in dataset:
        image_path = Path(sample["image_path"])
        assert image_path.exists(), f"{image_path} doesn't exist"

    output_dir = Path(root_output_dir)
    logging.info(f"Creating dataset of level-{level} {tile_size}x{tile_size} "
                 f"{slides_dataset.__class__.__name__} tiles at: {output_dir}")

    if overwrite and output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=not overwrite)
    thumbnail_dir = output_dir / "thumbnails"
    thumbnail_dir.mkdir(exist_ok=True)
    logging.info(f"Thumbnail directory: {thumbnail_dir}")

    func = functools.partial(process_one_slide, level=level, margin=margin, tile_size=tile_size,
                             foreground_threshold=foreground_threshold,
                             occupancy_threshold=occupancy_threshold, output_dir=output_dir,
                             thumbnail_dir=thumbnail_dir,
                             tile_progress=not parallel)

    if parallel:
        import multiprocessing

        pool = multiprocessing.Pool()
        map_func = pool.imap_unordered  # type: ignore
    else:
        map_func = map  # type: ignore

    list(tqdm(map_func(func, dataset), desc="Slides", unit="img", total=len(dataset)))  # type: ignore

    if parallel:
        pool.close()

    logging.info("Merging slide files in a single file")
    merge_dataset_csv_files(output_dir)


# for inference
def prepare_single_slide_dataset(slide_file: str = '', save_dir: str = '', level: int = 0, tile_size: int = 256):
    """
    This function is used to tile a single slide and save the tiles to a directory.
    -------------------------------------------------------------------------------
    Warnings: pixman 0.38 has a known bug, which produces partial broken images.
    Make sure to use a different version of pixman.
    -------------------------------------------------------------------------------

    Arguments:
    ----------
    slide_file : str
        The path to the slide file.
    save_dir : str
        The directory to save the tiles.
    level : int
        The magnification level to use for tiling. level=0 is the highest magnification level.
    tile_size : int
        The size of the tiles.
    """
    slide_id = os.path.basename(slide_file)
    # slide_sample = {"image": slide_file, "slide_id": slide_id, "metadata": {'TP53': 1, 'Diagnosis': 'Lung Cancer'}}
    slide_sample = {"image": slide_file, "slide_id": slide_id, "metadata": {}}

    save_dir = Path(save_dir)
    if save_dir.exists():
        print(f"Warning: Directory {save_dir} already exists. ")

    print(f"Processing slide {slide_file} at level {level} with tile size {tile_size}. Saving to {save_dir}.")

    slide_dir = process_one_slide(
        slide_sample,
        level=level,
        margin=0,
        tile_size=tile_size,
        foreground_threshold=None,
        occupancy_threshold=0.1,
        output_dir=save_dir / "output",
        thumbnail_dir=save_dir / "thumbnails",
        tile_progress=True,
    )

    dataset_csv_path = slide_dir / "dataset.csv"
    dataset_df = pd.read_csv(dataset_csv_path)
    assert len(dataset_df) > 0
    failed_csv_path = slide_dir / "failed_tiles.csv"
    failed_df = pd.read_csv(failed_csv_path)
    assert len(failed_df) == 0

    print(f"Slide {slide_file} has been tiled. {len(dataset_df)} tiles saved to {slide_dir}.")

class TileEncodingDataset(Dataset):
    """
    Do encoding for tiles

    Arguments:
    ----------
    image_paths : List[str]
        List of image paths, each image is named with its coordinates
        Example: ['images/256x_256y.png', 'images/256x_512y.png']
    transform : torchvision.transforms.Compose
        Transform to apply to each image
    """

    def __init__(self, image_paths: List[str], transform=None):
        self.transform = transform
        self.image_paths = image_paths

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img_name = os.path.basename(img_path)
        # get x, y coordinates from the image name
        x, y = img_name.split('.png')[0].split('_')
        x, y = int(x.replace('x', '')), int(y.replace('y', ''))
        # load the image
        with open(img_path, "rb") as f:
            img = Image.open(f).convert("RGB")
            if self.transform:
                img = self.transform(img)
        return {'img': torch.from_numpy(np.array(img)),
                'coords': torch.from_numpy(np.array([x, y])).float()}

