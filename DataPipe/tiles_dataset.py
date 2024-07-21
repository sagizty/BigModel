"""
# Pipeline for processing the sample

# type A is for (ROI+WSI approaches)
# type B is for (Cell+ROI+WSI approaches)



Terminology:
1. WSI / slide: the whole scanned slide
2. ROI: the region of interest. usually it means the region with tissue or a small part of slide,
notice it has two meanings therefor it may be confusing.
3. Tile (as noun) / Patch / ROI: a small part taken from WSI,
usually a non-overlapping griding can generate a series of them for each WSI.
4. Tile (as verb) / Crop: the non-overlapping griding methods to break the WSI into many small patches.
5. ROI sequence / tile sequence / patch sequence: all patches taken from one WSI.

In WSI object, the data is stored in multiple scale, it is called levels (from high to low)
it corresponds to low resolution/magnification (highest level) to high resolution/magnification (lowest level)
for example:
1. you can see all tissues on your screen with 1x magnification,
now its lowest magnification and highest level (level-8), now the resolution for whole slide is super low
2. you can only one cell on your screen with 100x magnification,
now its highest magnification and lowest level (level-0), now the resolution for whole slide is super high

"""

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

from WSI_tools import *
from Segmentation_and_filtering_tools import *

def process_one_slide_to_tiles(sample: Dict["SlideKey", Any], level: int, margin: int, tile_size: int,
                               foreground_threshold: Optional[float], occupancy_threshold: float, output_dir: Path,
                               thumbnail_dir: Path, tile_progress: bool = False, image_key="image") -> str:
    """Load and process a slide, saving tile images and information to a CSV file.

    :param sample: Slide information dictionary, returned by the input slide dataset.

    :param level: Magnification level at which to process the slide.
    :param margin: Margin around the foreground bounding box, in pixels at lowest resolution.
    :param tile_size: Lateral dimensions of each tile, in pixels.

    :param foreground_threshold: Luminance threshold (0 to 255) to determine if one pixel is foreground
    then the pixels can be used for checking tile occupancy. If `None` (default),
    an optimal threshold will be estimated automatically.

    :param occupancy_threshold: Threshold (between 0 and 1) to determine empty tiles to discard.

    :param output_dir: Root directory for the output dataset; outputs for a single slide will be
    saved inside `output_dir/slide_id/`.

    :param tile_progress: Whether to display a progress bar in the terminal.
    :param image_key: Image key in the input and output dictionaries. default is 'image'


    """
    # STEP 0: set up path and log files
    output_dir.mkdir(parents=True, exist_ok=True)
    thumbnail_dir.mkdir(parents=True, exist_ok=True)

    slide_id: str = sample["slide_id"]
    rel_slide_dir = Path(slide_id)
    output_tiles_dir = output_dir / rel_slide_dir

    logging.info(f">>> Slide dir {output_tiles_dir}")
    if is_already_processed(output_tiles_dir):
        logging.info(f">>> Skipping {output_tiles_dir} - already processed")
        return output_tiles_dir

    else:
        output_tiles_dir.mkdir(parents=True, exist_ok=True)

        # prepare a csv log file for ROI tiles
        keys_to_save = ("slide_id", "tile_id", "image", "label",
                        "tile_y", "tile_x", "occupancy")
        # Decode the slide metadata (if got)
        slide_metadata: Dict[str, Any] = sample["metadata"]
        metadata_keys = tuple("slide_" + key for key in slide_metadata)
        # print('metadata_keys',metadata_keys)
        csv_columns: Tuple[str, ...] = (*keys_to_save, *metadata_keys)
        # print('csv_columns',csv_columns)

        # build a recording file to log the processed files
        dataset_csv_path = output_tiles_dir / "dataset.csv"
        dataset_csv_file = dataset_csv_path.open('w')
        dataset_csv_file.write(','.join(csv_columns) + '\n')  # write CSV header

        n_failed_tiles = 0
        failed_tiles_csv_path = output_tiles_dir / "failed_tiles.csv"
        failed_tiles_file = failed_tiles_csv_path.open('w')
        failed_tiles_file.write('tile_id' + '\n')  # write CSV header

        # STEP 1: take the WSI and get the ROIs (valid tissue regions)
        slide_image_path = Path(sample["image"])
        logging.info(f"Loading slide {slide_id} ...\nFile: {slide_image_path}")

        '''
        # fixme : gigapath design a temp file path to duplicate the WSI during cropping, we may not need
        # Somehow it's very slow on Datarbicks
        # hack: copy the slide file to a temporary directory
        # import tempfile  # a repo to make temp path and will automatically delete
        
        tmp_dir = tempfile.TemporaryDirectory()
        tmp_slide_image_path = Path(tmp_dir.name) / slide_image_path.name
        logging.info(f">>> Copying {slide_image_path} to {tmp_slide_image_path}")
        shutil.copy(slide_image_path, tmp_slide_image_path)
        sample["image"] = tmp_slide_image_path
        logging.info(f">>> Finished copying {slide_image_path} to {tmp_slide_image_path}")
        '''

        # take the valid tissue regions (ROIs) of the WSI (with monai and OpenSlide loader)
        loader = Loader_for_one_WSI_image(WSIReader(backend="OpenSlide"), level=level, margin=margin,
                                          foreground_threshold=foreground_threshold)
        loaded_WSI_sample = loader(sample)  # load 'image' from disk and composed it into 'sample'

        # Save original slide thumbnail
        save_openslide_thumbnail(slide_image_path, thumbnail_dir / (slide_image_path.name + "_original.png"))

        # Save ROI thumbnail
        slide_image = loaded_WSI_sample[image_key]  # CHW format
        plt.figure()
        plt.imshow(slide_image.transpose(1, 2, 0))
        plt.savefig(thumbnail_dir / (slide_image_path.name + "_roi.png"))
        plt.close()
        logging.info(f"Saving thumbnail {thumbnail_dir / (slide_image_path.name + '_roi.png')}, "
                     f"shape {slide_image.shape}")

        # STEP 2: Tile (crop) the WSI into ROI tiles (patches)
        logging.info(f"Tiling slide {slide_id} ...")
        # The estimated luminance (foreground threshold) for whole WSI is applied to ROI here to filter the tiles
        image_tiles, rel_tile_locations, occupancies, _ = \
            generate_tiles(loaded_WSI_sample[image_key], tile_size,
                           loaded_WSI_sample["foreground_threshold"], occupancy_threshold)

        # 'origin' is the level-0 coordinate (the highest magnification)
        # location is the current level coordinate (due to the mpp rescaling)
        # we need tile_locations at level-0 coordinate to locate and crop:
        tile_locations = (loaded_WSI_sample["scale"] * rel_tile_locations + loaded_WSI_sample["origin"]).astype(int)

        # tile_locations coordinates (N, 2), HW(YX) ordering

        n_tiles = image_tiles.shape[0]
        logging.info(f"{n_tiles} tiles found")

        # make a list to record the Tile information dictionary for valid tiles
        tile_info_list = []

        # STEP 3: save ROI tiles (patches) and prepare logs
        logging.info(f"Saving tiles for slide {slide_id} ...")
        for i in tqdm(range(n_tiles), f"Tiles ({slide_id[:6]}…)", unit="img", disable=not tile_progress):
            try:
                # save tile to the location
                tile_info = get_tile_info_dict(loaded_WSI_sample, occupancies[i], tile_locations[i], rel_slide_dir)
                save_chw_image(image_tiles[i], output_dir / tile_info["image"])
            except Exception as e:
                # sometimes certain tiles is broken (for pixel failure) and we need to skip
                n_failed_tiles += 1
                descriptor = get_tile_descriptor(tile_locations[i])
                # we write down these failed tiles into the log
                failed_tiles_file.write(descriptor + '\n')
                traceback.print_exc()
                warnings.warn(f"An error occurred while saving tile "
                              f"{get_tile_id(slide_id, tile_locations[i])}: {e}")
            else:
                # record the tile information into tile_info_list
                tile_info_list.append(tile_info)
                dataset_row = format_csv_row(tile_info, keys_to_save, metadata_keys)
                dataset_csv_file.write(dataset_row + '\n')
        # close csv logging
        dataset_csv_file.close()
        failed_tiles_file.close()

        # STEP 4: visualize the tile location overlay to WSI
        visualize_tile_locations(loaded_WSI_sample, thumbnail_dir / (slide_image_path.name + "_roi_tiles.png"),
                                 tile_info_list, tile_size, origin_offset=loaded_WSI_sample["origin"])

        '''
        # put back all tiles for visualization 
        assemble_img_array, offset = assemble_tiles_2d(image_tiles, tile_locations)
        assemble_img_array = downsample_chw_numpy_image(assemble_img_array)
        visualize_CHW_numpy_image(assemble_img_array, thumbnail_dir / (slide_image_path.name + "_roi_recompose.png"))
        '''

        if n_failed_tiles > 0:
            # what we want to do with slides that have some failed tiles? for now, just drop?
            logging.warning(f"{slide_id} is incomplete. {n_failed_tiles} tiles failed.")

        logging.info(f"Finished processing slide {slide_id}")

        return output_tiles_dir


# todo it should be designed for better pretraining management
def prepare_slides_dataset(slide_root, slide_suffixes=['.svs', '.ndpi'], metadata_file_paths=None):
    """
    Make a slides_dataset list, each element inside is a
    slide_sample = {"image": slide_file, "slide_id": slide_id, "metadata": {}}

    Later it will be warped like: dataset = torch.utils.data.Dataset(slides_dataset)

    Args:
        slide_root: a root folder of multiple WSIs, notice each WSI may be organized in different folder format
        this code will go through all files in the slide_root and find the WSI files with suffix
        in the slide_suffixes list

        slide_suffixes: list, the possible suffixes for WSI image file.

        metadata_file_paths: list, the paths to multiple metadata files, this code will go through the files and
        load them as pd dataframe, if the information for a certain WSI (slide_id) can be found in certain pd dataframe,
        the information will be kept in dictionary in "metadata"

    Returns: slides_dataset
    """
    slides_dataset = []

    # Traverse the slide_root directory to find all files with the given suffixes
    for root, _, files in os.walk(slide_root):
        for file in files:
            if any(file.endswith(suffix) for suffix in slide_suffixes):
                slide_file = Path(root) / file
                # Assuming the slide_id is the filename [0:12] without the suffix
                slide_id = slide_file.stem
                patient_id = slide_id[0:12]  # fixme TCGA format only

                # Initialize the slide_sample dictionary
                slide_sample = {"image": str(slide_file), "slide_id": slide_id, "metadata": {}}

                # If metadata file paths are provided, load metadata
                if metadata_file_paths:
                    for metadata_file_path in metadata_file_paths:
                        metadata_df = pd.read_csv(metadata_file_path)

                        # Check if the slide_id is in the metadata dataframe
                        if patient_id in metadata_df['patient_id'].values:  # fixme TCGA format only 'patient_id'
                            # Get the metadata row for this slide_id and convert it to a dictionary
                            # fixme TCGA format only
                            metadata = metadata_df.loc[metadata_df['patient_id']
                                                       == patient_id].to_dict(orient='records')[0]
                            slide_sample["metadata"].update(metadata)

                # Append the slide_sample to the slides_dataset
                slides_dataset.append(slide_sample)

    return slides_dataset


# Process multiple WSIs for pretraining
def prepare_tiles_dataset_for_all_slides(slides_dataset: "SlidesDataset", root_output_dir: Union[str, Path],
                                         level: int, tile_size: int, margin: int,
                                         foreground_threshold: Optional[float] = None,
                                         occupancy_threshold: float = 0.1, parallel: bool = False,
                                         n_processes: Optional[int] = 1,
                                         overwrite: bool = False,
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
    :param n_processes: If given, limit the total number of slides for multiprocessing

    :param overwrite: Whether to overwrite an existing output tiles dataset. If `True`, will delete
    and recreate `root_output_dir`, otherwise will resume by skipping already processed slides.
    :param n_slides: If given, limit the total number of slides for debugging.
    """

    # Ignoring some types here because mypy is getting confused with the MONAI Dataset class
    # to select a sub-set of samples use keyword n_slides
    dataset = Dataset(slides_dataset)[:n_slides]  # type: ignore

    # make sure all slide files exist in the image dir
    for sample in dataset:
        image_path = Path(sample["image"])
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

    func = functools.partial(process_one_slide_to_tiles, level=level, margin=margin, tile_size=tile_size,
                             foreground_threshold=foreground_threshold,
                             occupancy_threshold=occupancy_threshold, output_dir=output_dir,
                             thumbnail_dir=thumbnail_dir,
                             tile_progress=not parallel)

    if parallel:
        import multiprocessing

        pool = multiprocessing.Pool(processes=n_processes)
        map_func = pool.imap_unordered  # type: ignore
    else:
        map_func = map  # type: ignore

    output_tilesets_dirs = list(
        tqdm(map_func(func, dataset), desc="Slides", unit="img", total=len(dataset)))  # type: ignore

    if parallel:
        pool.close()

    logging.info("Merging slide files into a single file")
    merge_dataset_csv_files(output_dir)


# for inference
def prepare_tiles_dataset_for_single_slide(slide_file: str = '', save_dir: str = '', level: int = 0,
                                           tile_size: int = 256):
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

    slide_dir = process_one_slide_to_tiles(
        slide_sample,
        level=level,
        margin=0,
        tile_size=tile_size,
        foreground_threshold=None,  # None to use automatic illuminance estimation
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


if __name__ == '__main__':
    slides_dataset = prepare_slides_dataset(slide_root='/data/hdd_1/ai4dd/metadata/TCGA-READ/raw_data_sample',
                                            metadata_file_paths=[
                                                '/data/hdd_1/ai4dd/metadata/240418-combined_labels.csv', ])
    prepare_tiles_dataset_for_all_slides(slides_dataset, root_output_dir='/data/ssd_1/BigModel/tiles_datasets',
                                         level=0, tile_size=224, margin=0, overwrite=True, parallel=False)
