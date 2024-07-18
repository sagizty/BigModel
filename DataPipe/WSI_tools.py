"""
a WSI is a scanned whole slide image of some tissue region and many blank background

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


this code consists many tools to
0.load WSI object
1.select the valid region (with tissue instead of empty)
2.tile(crop) the wsi into ROI sequences

we use
from monai.data.wsi_reader import WSIReader(backend="OpenSlide")
it returns slide shape in (CHW) not (CWH)!
However
slide = openslide.OpenSlide(wsi_path) return slide in (WHC)
"""

import openslide
import cv2
from typing import Any, Dict, Iterable, Optional, Sequence
import pandas as pd
import PIL
from matplotlib import collections, patches, pyplot as plt
from monai.data import Dataset
from monai.data.wsi_reader import WSIReader
from tqdm import tqdm

from box_utils import get_bounding_box, Box
from Segmentation_and_filtering_tools import *


# this is used to process WSI to get slide-level (for OpenSlide) at a target mpp
def find_level_for_target_mpp(slide_path, target_mpp):
    """
    Find the level in the slide that mostly corresponds to the target MPP.
    NOTE: Prov-GigaPath is trained with 0.5 mpp preprocessed slides

    Parameters:
    slide_path (str): Path to the slide file.
    target_mpp (float): Target microns per pixel (MPP).

    Returns:
    int: Level number that corresponds to the target MPP or None if not found.
    """
    slide = openslide.OpenSlide(slide_path)

    print(slide.properties)

    # Retrieve resolution information from properties
    y_resolution = float(slide.properties.get('tiff.YResolution'))
    x_resolution = float(slide.properties.get('tiff.XResolution'))
    resolution_unit = slide.properties.get('tiff.ResolutionUnit')

    # Convert resolution to microns per pixel (MPP)
    if resolution_unit == 'centimeter':
        mpp_y = 10000 / y_resolution
        mpp_x = 10000 / x_resolution
    else:
        print("Resolution unit is not in centimeters. Adjust the calculation accordingly.")
        return None

    # Check if MPP information is available
    if not mpp_x or not mpp_y:
        print("Could not calculate MPP due to missing or invalid resolution information.")
        return None

    # Iterate through each level and calculate MPP
    for level in range(slide.level_count):
        # Calculate MPP for the current level
        level_mpp_y = mpp_y * slide.level_downsamples[level]
        level_mpp_x = mpp_x * slide.level_downsamples[level]

        # Check if this level's MPP is close to the target MPP
        if abs(level_mpp_x - target_mpp) < 0.1 and abs(level_mpp_y - target_mpp) < 0.1:
            print(f"Level {level} corresponds to approximately {target_mpp} MPP.")
            return level

    print(f"No level corresponds to approximately {target_mpp} MPP.")
    return None


# WSI reading tools
# fixme abandoned!
# MONAI's convention is that dictionary transforms have a 'd' suffix in the class name
class ReadImaged(MapTransform):
    """Basic transform to read image files.


    """

    def __init__(self, reader: WSIReader, keys: KeysCollection,
                 allow_missing_keys: bool = False, **kwargs: Any) -> None:
        """

        Args:
            reader: A MONAI `WSIReader` using OpenSlide backend.
            keys: Image keys to decode the dictionary of OpenSlide

            allow_missing_keys:
            **kwargs:
        """
        super().__init__(keys, allow_missing_keys=allow_missing_keys)
        self.reader = reader
        self.kwargs = kwargs

    def __call__(self, data: Dict) -> Dict:
        for key in self.keys:
            if key in data or not self.allow_missing_keys:
                data[key] = self.reader.read(data[key], **self.kwargs)
        return data


# fixme abandoned! Temporary workaround for MONAI bug (https://github.com/Project-MONAI/MONAI/pull/3417/files)
def _get_image_size(img, level=None, location=(0, 0), backend="openslide"):
    """
    This is a manual workaround for a MONAI bug (https://github.com/Project-MONAI/MONAI/issues/3415)
    fixed in a later PR (https://github.com/Project-MONAI/MONAI/pull/3417).

    Args:
        img: The WSI image object returned by `reader.read(<image_file>)`.
        level: Index of the desired magnification level as defined in the `slide_obj` headers.
        location: the top left corner, should be (0, 0)
        backend: method used to open WSI

    Returns:

    """
    max_size = []
    downsampling_factor = []
    if backend == "openslide":
        downsampling_factor = img.level_downsamples[level]
        max_size = img.level_dimensions[level][::-1]
    elif backend == "cucim":
        downsampling_factor = img.resolutions["level_downsamples"][level]
        max_size = img.resolutions["level_dimensions"][level][::-1]
    elif backend == "tifffile":
        level0_size = img.pages[0].shape[:2]
        max_size = img.pages[level].shape[:2]
        downsampling_factor = np.mean([level0_size[i] / max_size[i] for i in range(len(max_size))])

    # scale and subtract the top left corner of the patch from maximum size
    level_location = [round(location[i] / downsampling_factor) for i in range(len(location))]
    size = [max_size[i] - level_location[i] for i in range(len(max_size))]

    return size


# fixme abandoned! Temporary workaround for MONAI bug (https://github.com/Project-MONAI/MONAI/pull/3417/files)
def load_slide_at_level(reader: WSIReader, slide_obj: OpenSlide, level: int) -> np.ndarray:
    """Load full slide array at the given magnification level.

    This is a manual workaround for a MONAI bug (https://github.com/Project-MONAI/MONAI/issues/3415)
    fixed in a later PR (https://github.com/Project-MONAI/MONAI/pull/3417).

    :param reader: A MONAI `WSIReader` using OpenSlide backend.
    :param slide_obj: The OpenSlide image object returned by `reader.read(<image_file>)`.
    :param level: Index of the desired magnification level as defined in the `slide_obj` headers.

    :return: The loaded image array in (C, H, W) format.
    """
    size = _get_image_size(slide_obj, level=level)
    # get the whole WSI and resize to 'size' to zoom to defined magnification level
    img_data, meta_data = reader.get_data(slide_obj, size=size, level=level)

    return img_data, meta_data


# The pipeline class for WSI loading and region selection:
class Loader_for_one_WSI_with_valid_regions(MapTransform):
    """
    This process is a pipeline warps openslide and other functions:
    Load a pathology whole slide image (WSI), and crop the valid regions (with foreground bounding box).

    This process operates on 'Slide information dictionary'!!!!


    replacing the file paths in `image_key` with the respective loaded arrays, in (C, H, W) format.
    Also adds the following meta-sample entries:
    - 'location' (tuple): top-right coordinates of the bounding box
    - 'size' (tuple): width and height of the bounding box
    - 'level' (int): chosen magnification level
    - 'scale' (float): corresponding scale, loaded from the file
    """

    def __init__(self, reader: WSIReader, image_key: str = "image", level: int = 0,
                 margin: int = 0, foreground_threshold: Optional[float] = None) -> None:
        """
        :param reader: An instance of MONAI's `WSIReader`.
        :param image_key: Image key in the input and output dictionaries. default is 'image'

        :param level: Magnification level to load from the raw multi-scale files.
        :param margin: Amount in pixels by which to enlarge the estimated bounding box for cropping.

        :param foreground_threshold: Pixels with luminance below this value will be considered as foreground.
        If `None` (default), an optimal threshold will be estimated automatically using Otsu's method.
        and it will be passed to the process of each specific WSI/ROI

        """
        super().__init__([image_key], allow_missing_keys=False)
        self.reader = reader
        self.image_key = image_key
        self.level = level
        self.margin = margin
        self.foreground_threshold = foreground_threshold  # default None for automatic estimation

    # WSI segmentation (high level)
    # + calculate luminance estimation for the threshold (identifying pixel as foreground)
    def WSI_region_detection(self, slide_obj: OpenSlide) -> Box:
        # Estimate bounding box at the lowest resolution/magnification (i.e. highest level)
        highest_level = slide_obj.level_count - 1
        # Load full slide array at the given magnification level. in [C,H,W]
        slide, meta_data = self.reader.get_data(slide_obj, level=highest_level)
        logging.info(f"img: {slide.dtype} {slide.shape}, metadata: {meta_data}")

        # old monai has a bug, and people use Temporary workaround (see line 94 fixmes)
        # Nowadays there's no need to do such a detour! if there is still bug, use manual detour with:
        #  slide, meta_data = load_slide_at_level(self.reader, slide_obj, level=highest_level)

        if slide_obj.level_count == 1:
            # in this case, the sample is not nicely organized into multiple level,
            # the whole slide is stored only once at a very high magnification (eg 100x).
            logging.warning(f"Only one WSI level found. "
                            f"segment_foregound at high magnification for whole slide will use a lot of memory.")

        # if self.foreground_threshold is None, a threshold will be estimated with skimage Otsu's method.
        # foreground_mask is (1, H, W) boolean array indicating whether the pixel is foreground or not
        foreground_mask, Luminance_threshold = segment_foreground(slide, self.foreground_threshold)
        # threshold: Pixels with luminance below this value will be considered as foreground.

        scale = slide_obj.level_downsamples[highest_level]
        bbox = scale * get_bounding_box(foreground_mask).add_margin(self.margin)
        return bbox, Luminance_threshold

    def __call__(self, sample: Dict) -> Dict:
        """
        This process open WSI and compose image data and other information
        into the 'Slide information dictionary'

        :param sample: 'Slide information dictionary', dict describing image metadata.

        Example:
        {'image_id': ['1ca999adbbc948e69783686e5b5414e4'],
        'image': ['/tmp/datasets/PANDA/train_images/1ca999adbbc948e69783686e5b5414e4.tiff'],
         'mask': ['/tmp/datasets/PANDA/train_label_masks/1ca999adbbc948e69783686e5b5414e4_mask.tiff'],
         'data_provider': ['karolinska'],
         'isup_grade': tensor([0]),
         'gleason_score': ['0+0']}

        Returns:
        sample (but compose the WSI information)

        Example:
        {'image_id': ['1ca999adbbc948e69783686e5b5414e4'],
        'image': ['/tmp/datasets/PANDA/train_images/1ca999adbbc948e69783686e5b5414e4.tiff'],
         'mask': ['/tmp/datasets/PANDA/train_label_masks/1ca999adbbc948e69783686e5b5414e4_mask.tiff'],
         'data_provider': ['karolinska'],
         'isup_grade': tensor([0]),
         'gleason_score': ['0+0'],

         'image_key(default is 'image')': (1,C,H,W) array for the WSI's valid tissue region
         'location': valid tissue region on the level
         'size': scaled size of valid tissue region
         'level': converted magnification level
         "origin": the absolute location on level-0 of the valid tissue region
         "scale": the scale factor of magnification level to level-0
         "foreground_threshold": luminance estimation for the threshold (identifying pixel as foreground)
         }

        """
        # STEP 1: Open WSI
        logging.info(f"Loader_for_one_WSI_with_valid_regions: read {sample[self.image_key]}")
        WSI_image_obj: OpenSlide = self.reader.read(sample[self.image_key])

        # STEP 2: Select the valid regions on the WSI, in one bbox
        logging.info("Loader_for_one_WSI_with_valid_regions: get bbox")
        # get WSI bbox's location and the foreground_threshold for Luminance estimation
        level0_bbox, Luminance_threshold = self.WSI_region_detection(WSI_image_obj)
        logging.info(f"Loader_for_one_WSI_with_valid_regions: level0_bbox: {level0_bbox}")

        # STEP 3: Calibrate the location for OpenSlide
        # OpenSlide takes absolute location coordinates in the level 0 reference frame,
        # but relative region size in pixels at the chosen level
        scale = WSI_image_obj.level_downsamples[self.level]
        scaled_bbox = level0_bbox / scale
        # fixme: notice in Monai reader.get_data: order of location/size arguments is YX (HW)
        original_location = (level0_bbox.y, level0_bbox.x)
        get_data_kwargs = dict(location=original_location,
                               size=(scaled_bbox.h, scaled_bbox.w),
                               level=self.level)

        # STEP 4: take the valid region from WSI
        img_data, _ = self.reader.get_data(WSI_image_obj, **get_data_kwargs)  # type: ignore
        # img_data is [C,H,W]
        logging.info(f"img_data: {img_data.dtype} {img_data.shape}")
        # compose the WSI information for following steps
        sample[self.image_key] = img_data
        sample.update(get_data_kwargs)
        sample["origin"] = original_location  # style in YX / HW
        sample["scale"] = scale
        sample["foreground_threshold"] = Luminance_threshold

        WSI_image_obj.close()
        return sample


# WSI tilling tools
def get_1d_padding(length: int, tile_size: int) -> Tuple[int, int]:
    """Computes symmetric padding for `length` to be divisible by `tile_size`."""
    pad = (tile_size - length % tile_size) % tile_size
    return (pad // 2, pad - pad // 2)


def pad_for_tiling_2d(array: np.ndarray, tile_size: int, channels_first: Optional[bool] = True,
                      **pad_kwargs: Any) -> Tuple[np.ndarray, np.ndarray]:
    """Symmetrically pads a 2D `array` such that both dimensions are divisible by `tile_size`.

    :param array: 2D image array.
    :param tile_size: Width/height of each tile in pixels.
    :param channels_first: Whether `array` is in CHW (`True`, default) or HWC (`False`) layout.
    :param pad_kwargs: Keyword arguments to be passed to `np.pad()` (e.g. `constant_values=0`).
    :return: A tuple containing:
        - `padded_array`: Resulting array, in the same CHW/HWC layout as the input.
        - `offset`: HW (YX) offset introduced by the padding. Add this to coordinates relative to the
        original array to obtain indices for the padded array.
    """
    height, width = array.shape[1:] if channels_first else array.shape[:-1]

    padding_h = get_1d_padding(height, tile_size)
    padding_w = get_1d_padding(width, tile_size)
    padding = [padding_h, padding_w]

    channels_axis = 0 if channels_first else 2
    padding.insert(channels_axis, (0, 0))  # zero padding on channels axis
    padded_array = np.pad(array, padding, **pad_kwargs)
    offset = (padding_h[0], padding_w[0])

    return padded_array, np.array(offset)


def tile_array_2d(array: np.ndarray, tile_size: int, channels_first: Optional[bool] = True,
                  **pad_kwargs: Any) -> Tuple[np.ndarray, np.ndarray]:
    """
    Split an image array into square non-overlapping tiles.

    The array will be padded symmetrically if its dimensions are not exact multiples of `tile_size`.

    :param array: Image array.
    :param tile_size: Width/height of each tile in pixels.
    :param pad_kwargs: Keyword arguments to be passed to `np.pad()` (e.g. `constant_values=0`).
    :param channels_first: Whether `array` is in CHW (`True`, default) or HWC (`False`) layout.

    :return: A tuple containing:
        - `tiles`: A batch of tiles in NCHW layout.
        - `coords`: YX coordinates of each tile, in the same order.
    """
    padded_array, (offset_h, offset_w) = pad_for_tiling_2d(array, tile_size, channels_first, **pad_kwargs)

    if channels_first:
        channels, height, width = padded_array.shape
    else:
        height, width, channels = padded_array.shape
    n_tiles_h = height // tile_size
    n_tiles_w = width // tile_size

    if channels_first:
        intermediate_shape = (channels, n_tiles_h, tile_size, n_tiles_w, tile_size)
        axis_order = (1, 3, 0, 2, 4)  # (n_tiles_h, n_tiles_w, channels, tile_size, tile_size)
        output_shape = (n_tiles_h * n_tiles_w, channels, tile_size, tile_size)
    else:
        intermediate_shape = (n_tiles_h, tile_size, n_tiles_w, tile_size, channels)
        axis_order = (0, 2, 1, 3, 4)  # (n_tiles_h, n_tiles_w, tile_size, tile_size, channels)
        output_shape = (n_tiles_h * n_tiles_w, tile_size, tile_size, channels)

    tiles = padded_array.reshape(intermediate_shape)  # Split width and height axes
    tiles = tiles.transpose(axis_order)
    tiles = tiles.reshape(output_shape)  # Flatten tile batch dimension

    # Compute top-left coordinates of every tile, relative to the original array's origin
    coords_h = tile_size * np.arange(n_tiles_h) - offset_h
    coords_w = tile_size * np.arange(n_tiles_w) - offset_w
    # Shape: (n_tiles_h * n_tiles_w, 2), ordering W,H due to np.meshgrid(coords_w, coords_h, indexing='xy')
    coords_wh = np.stack(np.meshgrid(coords_w, coords_h, indexing='xy'), axis=-1).reshape(-1, 2)
    # Swap columns to get [H, W] ordering
    coords_hw = coords_wh[:, [1, 0]]
    return tiles, coords_hw


def assemble_tiles_2d(tiles: np.ndarray, coords: np.ndarray, fill_value: Optional[int] = 255,
                      channels_first: Optional[bool] = True) -> Tuple[np.ndarray, np.ndarray]:
    """Assembles a 2D array from sequences of tiles and coordinates.

    :param tiles: Stack of tiles with batch dimension first.
    :param coords: YX tile coordinates, assumed to be spaced by multiples of `tile_size` (shape: [N, 2]).
    :param tile_size: Size of each tile; must be >0.
    :param fill_value: Value to assign to empty elements (default: 255 for white).
    :param channels_first: Whether each tile is in CHW (`True`, default) or HWC (`False`) layout.
    :return: A tuple containing:
        - `array`: The reassembled 2D array with the smallest dimensions to contain all given tiles.
        - `offset`: The lowest YX coordinates.
        - `offset`: YX offset introduced by the assembly.
        Added this as tile coordinates to obtain indices for the assembled array.
    """
    if coords.shape[0] != tiles.shape[0]:
        raise ValueError(f"Tile coordinates and values must have the same length, "
                         f"got {coords.shape[0]} and {tiles.shape[0]}")

    if channels_first:
        n_tiles, channels, tile_size, _ = tiles.shape
    else:
        n_tiles, tile_size, _, channels = tiles.shape
    tile_ys, tile_xs = coords.T

    y_min, y_max = min(tile_ys), max(tile_ys + tile_size)
    x_min, x_max = min(tile_xs), max(tile_xs + tile_size)
    height = y_max - y_min
    width = x_max - x_min

    output_shape = (channels, height, width) if channels_first else (height, width, channels)
    array = np.full(output_shape, fill_value, dtype=tiles.dtype)

    offset = np.array([-y_min, -x_min])  # todo, I don't understand what is this
    for idx in range(n_tiles):
        row = coords[idx, 0] + offset[0]  # y axis
        col = coords[idx, 1] + offset[1]  # x axis
        if channels_first:
            array[:, row:row + tile_size, col:col + tile_size] = tiles[idx]
        else:
            array[row:row + tile_size, col:col + tile_size, :] = tiles[idx]

    return array, offset


def downsample_chw_numpy_image(image: np.ndarray, scale_factor: Optional[float] = None) -> np.ndarray:
    """Downsamples a CHW numpy array while maintaining the same aspect ratio.

    :param image: The input image in CHW format.
    :param scale_factor: The factor by which to downsample the image (e.g., 0.5 for half size).
    :return: The downsampled image in CHW format.
    """
    channels, height, width = image.shape
    scale_factor = scale_factor or 2000.0/max(height, width)

    if not (0 < scale_factor <= 1):
        raise ValueError("Scale factor must be in the range (0, 1].")

    # Calculate new dimensions
    new_height = int(height * scale_factor)
    new_width = int(width * scale_factor)

    # Initialize an array for the downsampled image
    downsampled_image = np.zeros((channels, new_height, new_width), dtype=image.dtype)

    # Downsample each channel
    for c in range(channels):
        downsampled_image[c] = cv2.resize(image[c], (new_width, new_height), interpolation=cv2.INTER_AREA)

    return downsampled_image


# Other tools
def get_tile_descriptor(tile_location: Sequence[int]) -> str:
    """Format the YX tile coordinates into a tile descriptor.

    :param tile_location: Sequence[int] get the encoded tile coordinates
    :return: a string describing the tile location like 01234y_05678x

    """
    return f"{tile_location[0]:05d}y_{tile_location[1]:05d}x"


def get_tile_id(slide_id: str, tile_location: Sequence[int]) -> str:
    """
    Format the slide ID and YX tile coordinates into a unique tile ID.

    :param slide_id: id name of the WSI
    :param tile_location: Sequence[int] get the encoded tile coordinates

    :return: a string describing the tile WSI_name and location like WSIname_01234y_05678x
    """
    return f"{slide_id}.{get_tile_descriptor(tile_location)}"


def get_tile_info_dict(sample: Dict["SlideKey", Any], occupancy: float, tile_location: Sequence[int],
                       rel_slide_dir: Path) -> Dict["TileKey", Any]:
    """Map slide information and tiling outputs into tile-specific information dictionary.

    :param sample: Slide dictionary.
    :param occupancy: Estimated tile foreground occuppancy.
    :param tile_location: Tile YX coordinates.
    :param rel_slide_dir: Directory where tiles are saved, relative to dataset root.

    :return: Tile information dictionary.
    """
    slide_id = sample["slide_id"]
    descriptor = get_tile_descriptor(tile_location)
    rel_image_path = f"{rel_slide_dir}/{descriptor}.png"

    tile_info = {
        "slide_id": slide_id,
        "tile_id": get_tile_id(slide_id, tile_location),
        "image": rel_image_path,
        "label": sample.get("label", None),
        "tile_y": tile_location[0],
        "tile_x": tile_location[1],
        "occupancy": occupancy,
        "metadata": {"slide_" + key: value for key, value in sample["metadata"].items()}
    }

    return tile_info


def format_csv_row(tile_info: Dict["TileKey", Any], keys_to_save: Iterable["TileKey"],
                   metadata_keys: Iterable[str]) -> str:
    """Format tile information dictionary as a row to write to a dataset CSV tile.

    :param tile_info: Tile information dictionary.
    :param keys_to_save: Which main keys to include in the row, and in which order.
    :param metadata_keys: Likewise for metadata keys.
    :return: The formatted CSV row.
    """
    tile_slide_metadata = tile_info.pop("metadata")

    fields = [str(tile_info[key]) for key in keys_to_save]
    fields.extend(str(tile_slide_metadata[key]) for key in metadata_keys)
    dataset_row = ','.join(fields)

    return dataset_row


def load_image_dict(sample: dict, level: int, margin: int, foreground_threshold: Optional[float] = None) -> Dict[
    "SlideKey", Any]:
    """
    Load image from metadata dictionary
    :param sample: dict describing image metadata. Example:
        {'image_id': ['1ca999adbbc948e69783686e5b5414e4'],
        'image': ['/tmp/datasets/PANDA/train_images/1ca999adbbc948e69783686e5b5414e4.tiff'],
         'mask': ['/tmp/datasets/PANDA/train_label_masks/1ca999adbbc948e69783686e5b5414e4_mask.tiff'],
         'data_provider': ['karolinska'],
         'isup_grade': tensor([0]),
         'gleason_score': ['0+0']}

    :param level: level of resolution to be loaded
    :param margin: margin to be included

    :param foreground_threshold: Pixels with luminance below this value will be considered as foreground.
        If `None` (default), an optimal threshold will be estimated automatically using Otsu's method.
        and it will be passed to the process of each specific WSI/ROI

    :return: a dict containing the image sample and metadata
    """
    loader = Loader_for_one_WSI_with_valid_regions(WSIReader(backend="OpenSlide"), level=level, margin=margin,
                                                   foreground_threshold=foreground_threshold)
    WSI_img = loader(sample)

    return WSI_img


def save_thumbnail(slide_path, output_path, size_target=1024):
    with OpenSlide(str(slide_path)) as openslide_obj:
        scale = size_target / max(openslide_obj.dimensions)
        thumbnail = openslide_obj.get_thumbnail([int(m * scale) for m in openslide_obj.dimensions])
        thumbnail.save(output_path)
        logging.info(f"Saving thumbnail {output_path}, shape {thumbnail.size}")


def visualize_tile_locations(slide_sample, output_path, tile_info_list, tile_size, origin_offset):
    """

    Args:
        slide_sample: Slide information dictionary, returned by the input slide dataset.
        output_path: Root directory for the output visualizations
        tile_info_list: a list to record the Tile information dictionary for valid tiles
        tile_size: tile size
        origin_offset: absolute level-0 location of the top-left pixel on the valid tissue regions

    Returns:

    """
    # check slide_image size. should be thumbnail size?
    slide_image = slide_sample["image"]
    downscale_factor = slide_sample["scale"]

    fig, ax = plt.subplots()
    ax.imshow(slide_image.transpose(1, 2, 0))
    rects = []
    for tile_info in tile_info_list:
        # change level-0 coordinate to the current level coordinate
        # tile location is in the level-0 image coordinate, while the slide image is after selecting ROI
        # additionally, the axis ordering is converted to xy to use patches.Rectangle here
        xy = ((tile_info["tile_x"] - origin_offset[1]) / downscale_factor,
              (tile_info["tile_y"] - origin_offset[0]) / downscale_factor)
        rects.append(patches.Rectangle(xy, tile_size, tile_size))

    # paint black to the location of tiles
    pc = collections.PatchCollection(rects, match_original=True, alpha=0.5, edgecolor="black")
    pc.set_array(np.array([100] * len(tile_info_list)))
    ax.add_collection(pc)
    fig.savefig(output_path)
    plt.close()


def visualize_CHW_numpy_image(image_array, output_path):
    # Convert from (channels, height, width) to (height, width, channels)
    image_array = np.transpose(image_array, (1, 2, 0))

    # Ensure the values are in the range [0, 255] and convert to uint8
    if image_array.max() <= 1.0:  # Assuming the input array is in range [0, 1]
        image_array = (image_array * 255).astype(np.uint8)
    else:  # Assuming the input array is already in range [0, 255]
        image_array = image_array.astype(np.uint8)

    # Display the image using Matplotlib
    fig, ax = plt.subplots()
    ax.imshow(image_array)
    fig.savefig(output_path)
    plt.close()


def is_already_processed(output_tiles_dir):
    """
    check whether the slide has been processed with all output log files

    Args:
        output_tiles_dir: output folder for tiles (on WSI name)

    Returns:

    """
    if not output_tiles_dir.exists():
        return False

    if len(list(output_tiles_dir.glob("*.png"))) == 0:
        return False

    dataset_csv_path = output_tiles_dir / "dataset.csv"
    try:
        df = pd.read_csv(dataset_csv_path)
    except:
        return False

    return len(df) > 0


def merge_dataset_csv_files(dataset_dir: Path) -> Path:
    """Combines all "*/dataset.csv" files into a single "dataset.csv" file in the given directory."""
    full_csv = dataset_dir / "dataset.csv"
    # TODO change how we retrieve these filenames, probably because mounted, the operation is slow
    #  and it seems to find many more files
    # print("List of files")
    # print([str(file) + '\n' for file in dataset_dir.glob("*/dataset.csv")])
    with full_csv.open('w') as full_csv_file:
        # full_csv_file.write(','.join(CSV_COLUMNS) + '\n')  # write CSV header
        first_file = True
        for slide_csv in tqdm(dataset_dir.glob("*/dataset.csv"), desc="Merging dataset.csv", unit='file'):
            logging.info(f"Merging slide {slide_csv}")
            content = slide_csv.read_text()
            if not first_file:
                content = content[content.index('\n') + 1:]  # discard header row for all but the first file
            full_csv_file.write(content)
            first_file = False
    return full_csv
