from tiles_dataset import *


def process_one_slide_to_tiles(sample: Dict["SlideKey", Any],
                               margin: int, tile_size: int,
                               foreground_threshold: Optional[float], occupancy_threshold: float,
                               output_dir: Path, thumbnail_dir: Path,
                               tile_progress: bool = False, image_key: str = "image") -> str:
    """Load and process a slide, saving tile images and information to a CSV file.

    :param sample: Slide information dictionary, returned by the input slide dataset.

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

        # STEP 1: take the WSI and get the ROIs (valid tissue regions)
        slide_image_path = Path(sample["image"])
        logging.info(f"Loading slide {slide_id} ...\nFile: {slide_image_path}")

        # take the valid tissue regions (ROIs) of the WSI (with monai and OpenSlide loader)
        loader = Loader_for_get_one_WSI_sample(WSIReader(backend="OpenSlide"), image_key=image_key,
                                               target_mpp=0.5, margin=margin,
                                               foreground_threshold=foreground_threshold,
                                               thumbnail_dir=thumbnail_dir)
        WSI_image_obj, loaded_WSI_sample = loader(sample)  # load 'image' from disk and composed it into 'sample'

        # STEP 2: Tile (crop) the WSI into ROI tiles (patches), save into h5
        logging.info(f"Tiling slide {slide_id} ...")

        # The estimated luminance (foreground threshold) for whole WSI is applied to ROI here to filter the tiles
        tile_info_list, n_failed_tiles = extract_valid_tiles(WSI_image_obj, loaded_WSI_sample, output_tiles_dir,
                                                             tile_size=tile_size,
                                                             foreground_threshold=loaded_WSI_sample[
                                                                 "foreground_threshold"],
                                                             occupancy_threshold=occupancy_threshold,
                                                             tile_progress=tile_progress)

        # STEP 3: visualize the tile location overlay to WSI
        new_visualize_tile_locations(loaded_WSI_sample, thumbnail_dir / (slide_image_path.name + "_roi_tiles.png"),
                                     tile_info_list, image_key=image_key)

        if n_failed_tiles > 0:
            # what we want to do with slides that have some failed tiles? for now, just drop?
            logging.warning(f"{slide_id} is incomplete. {n_failed_tiles} tiles failed in reading.")

        logging.info(f"Finished processing slide {slide_id}")

        return output_tiles_dir


# for inference
def prepare_tiles_dataset_for_single_slide(slide_file: str = '', save_dir: str = '', tile_size: int = 256):
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
    tile_size : int
        The size of the tiles.
    """
    slide_id = os.path.basename(slide_file)
    # slide_sample = {"image": slide_file, "slide_id": slide_id, "metadata": {'TP53': 1, 'Diagnosis': 'Lung Cancer'}}
    slide_sample = {"image": slide_file, "slide_id": slide_id, "metadata": {}}

    save_dir = Path(save_dir)
    if save_dir.exists():
        print(f"Warning: Directory {save_dir} already exists. ")

    print(f"Processing slide {slide_file} with tile size {tile_size}. Saving to {save_dir}.")

    slide_dir = process_one_slide_to_tiles(
        slide_sample,
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
    prepare_tiles_dataset_for_single_slide(
        slide_file='/data/hdd_1/ai4dd/metadata/TCGA-READ/raw_data_sample/TCGA-AF-2693-01Z-00-DX1.620a9998-65df-4024-b719-f2384b4e7d36.svs/TCGA-AF-2693-01Z-00-DX1.620a9998-65df-4024-b719-f2384b4e7d36.svs',
        save_dir='/data/ssd_1/BigModel/tiles_datasets', tile_size=224)
    prepare_tiles_dataset_for_single_slide(
        slide_file='/data/hdd_1/ai4dd/metadata/TCGA-READ/raw_data_sample/TCGA-AF-3400-01Z-00-DX1.b5c266c7-b465-4478-8058-cdf3f35dff38.svs/TCGA-AF-3400-01Z-00-DX1.b5c266c7-b465-4478-8058-cdf3f35dff38.svs',
        save_dir='/data/ssd_1/BigModel/tiles_datasets', tile_size=224)
    prepare_tiles_dataset_for_single_slide(
        slide_file='/data/hdd_1/ai4dd/metadata/TCGA-READ/raw_data_sample/TCGA-AF-3913-01Z-00-DX1.52725653-b597-4f1f-ac1f-80333e74ea1a.svs/TCGA-AF-3913-01Z-00-DX1.52725653-b597-4f1f-ac1f-80333e74ea1a.svs',
        save_dir='/data/ssd_1/BigModel/tiles_datasets', tile_size=224)
