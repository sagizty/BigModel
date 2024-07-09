#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#
# Original: https://github.com/microsoft/hi-ml/blob/main/hi-ml-cpath/src/health_cpath/preprocessing/loading.py
#  ------------------------------------------------------------------------------------------

import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import PIL
import skimage.filters
from monai.config.type_definitions import KeysCollection
from monai.data.wsi_reader import WSIReader
from monai.transforms.transform import MapTransform
from openslide import OpenSlide

# ROI/WSI Segmentation tools
def get_luminance(image_or_images: np.ndarray) -> np.ndarray:
    """Compute a grayscale version of the input image_or_images.

    :param image_or_images: The RGB image array in (*, C, H, W) format.
    :return: The single-channel luminance array as (*, H, W).
    """
    # TODO: Consider more sophisticated luminance calculation if necessary
    return image_or_images.mean(axis=-3, dtype=np.float16)  # type: ignore


def segment_foreground(image_or_images: np.ndarray, threshold: Optional[float] = None) \
        -> Tuple[np.ndarray, float]:
    """Segment the given slide by thresholding its luminance.

    :param image_or_images: The RGB image array in (*, C, H, W) format.
    :param threshold: Pixels with luminance below this value will be considered foreground.
    If `None` (default), an optimal threshold will be estimated automatically using Otsu's method.

    :return: the boolean output array of foreground_mask and the threshold used.
    foreground_mask is (*, H, W) boolean array indicating whether the pixel is foreground or not

    """
    luminance = get_luminance(image_or_images)  # -> (*, H, W)
    if threshold is None:
        threshold = skimage.filters.threshold_otsu(luminance)
    logging.info(f"Otsu threshold from luminance: {threshold}")
    return luminance < threshold, threshold


# ROI filtering tools
def ROI_occupancy_filtering(foreground_mask: np.ndarray, occupancy_threshold: float) \
        -> Tuple[np.ndarray, np.ndarray]:
    """
    Process the occupancy mask for ROI sequence and make selection array
    Idea is to exclude tiles that are mostly background based on estimated occupancy.

    :param foreground_mask: Boolean array of shape (*, H, W). * is the number of ROIs for this WSI

    :param occupancy_threshold: filtering threshold, ROIs with a lower occupancy will be discarded.

    :return: selections, occupancies
        selection: boolean arrays of shape (*,), or scalars if `foreground_mask` is for one single tile.
                    It contains which tiles were selected
        occupancies: float arrays of shape (*,), or scalars if `foreground_mask` is for one single tile.
                    It is the estimated occupancies for each ROI.
    """
    if occupancy_threshold < 0. or occupancy_threshold > 1.:
        raise ValueError("Tile occupancy threshold must be between 0 and 1")

    occupancy = foreground_mask.mean(axis=(-2, -1), dtype=np.float16)
    # boolean arrays of shape (*,), or scalars if `foreground_mask` is for one single tile.
    selections = (occupancy > occupancy_threshold).squeeze()
    # float arrays of shape (*,), or scalars if `foreground_mask` is for one single tile.
    occupancies = occupancy.squeeze()

    return selections, occupancies   # type: ignore


def check_empty_tiles(tiles: np.ndarray, pixel_std_threshold: int = 5,
                      extreme_value_portion_th: float = 0.5) -> np.ndarray:
    """
    Determine if a ROI image (tile) is empty.

    fixme: Hacky. This is depends on the value variance and ratio of pixels being 0

    :param tiles: The tile array in (N, C, H, W) format.
    :param pixel_std_threshold: The threshold for the pixel variance at one ROI
                                to say this ROI image is too 'empty'
    :param extreme_value_portion_th: The threshold for the ratio of the pixels being 0 of one ROI,
                                    to say this ROI image is too 'empty'

    :return: Boolean array of shape (N,). an union of the two filtering methods
    """
    # calculate standard deviation of rgb ROI image series
    N, C, H, W = tiles.shape
    flattened_tiles = tiles.reshape(N, C, H * W)  # -> N,C,pixels

    std_rgb = flattened_tiles[:, :, :].std(axis=2)  # [N,C] value is std across all locations in each channel
    std_rgb_mean = std_rgb.mean(axis=1)  # [N] value is std across all channel and all locations
    # [N] boolean array, value to say which ROIs are 'empty' (by having low variance)
    low_std_mask = std_rgb_mean < pixel_std_threshold

    # count 0 pixel values
    extreme_value_count = ((flattened_tiles == 0)).sum(axis=2)  # -> [N,C], value is zero_pixels number
    # -> [N,C], value is zero_pixels ratio for all locations in each channel
    extreme_value_proportion = extreme_value_count / (H * W)
    # [N] value is zero_pixels ratio across all channel and all locations
    extreme_value_proportion_mean = extreme_value_proportion.max(axis=1)
    # [N] boolean array, value to say which ROIs are 'empty' (by having alot of empty values)
    extreme_value_mask = extreme_value_proportion_mean > extreme_value_portion_th

    # the conclusion for selection is the union of the two filtering methods
    return low_std_mask | extreme_value_mask


# General image tools
def save_image(array_chw: np.ndarray, path: Path) -> PIL.Image:
    """
    Save an image array in (C, H, W) format to disk.

    :param array_chw: numpy image in CHW RGB format
    :param path: path of saving a .png ROI
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    array_hwc = np.moveaxis(array_chw, 0, -1).astype(np.uint8).squeeze()
    pil_image = PIL.Image.fromarray(array_hwc)
    pil_image.convert('RGB').save(path)
    return pil_image


