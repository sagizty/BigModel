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
from .create_tiles_dataset import process_slide


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


def tile_one_slide(slide_file: str = '', save_dir: str = '', level: int = 0, tile_size: int = 256):
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

    slide_dir = process_slide(
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
