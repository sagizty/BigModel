"""
Build the pseudo bulk dataset with the Cropped tiles       Oct 16th 23:00

This code generate a label h5 file into each WSI_folder (entities as slide_id)
the value of
"""
import pandas as pd
import os
from pathlib import Path
from tqdm import tqdm
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image


def read_locations_from_tiles_for_one_slide(slide_folder,suffix='.jpeg'):
    '''
    we have a lot of cropped tiles in the folder at slide_folder_path, we read their names to obtain the
    tile_location of each tile named as ("{tile_location[0]:05d}y_{tile_location[1]:05d}x"+ suffix)
    '''
    # read the tiles
    image_paths = [os.path.join(dp, f) for dp, _, filenames in os.walk(slide_folder)
                   for f in filenames if f.endswith(suffix)]
    tile_locations = []

    for img_path  in image_paths:
        img_name = os.path.basename(img_path)
        # Extract y, x coordinates from the image name
        y, x = img_name.split(suffix)[0].split('_')  # EG: 44004y_11136x.jpeg
        y, x = int(y.replace('y', '')), int(x.replace('x', ''))
        tile_locations.append((y,x))

    return tile_locations


def load_gene_description_csv(WSI_root,WSI_name):
    """
    WSI_root: this is the root folder of original Xenium WSI dataset folder, the labels are inside
    """
    # HE WSI from local SO data
    xenium_dir = Path(os.path.join(WSI_root, WSI_name, 'outs'))

    # read in Xenium (fixme Tianyi write random nonsense here)
    gene_description_csv = sc.read_10x_h5(xenium_dir / 'cell_feature_matrix.h5')
    # fixme assume this adata is a processed pandas dataframe, later we locate the gene values from it

    return gene_description_csv


def aggregate_pseudo_bulk_for_one_tile(gene_description_csv, tile_location, tile_size, mpp_info):
    """
    todo this code should calculate the aggregated genes for the given location
    """

    # todo gene_location_corrected and gene_box_corrected should be based on mpp_info, xenium infor (maybe resolution)
    gene_location_corrected = (y, x)   # fixme: notice the tile_location is encoded as (y, x), so here is also y,x
    gene_box_corrected = (y+tile_size, x + tile_size)
    # todo read gene_description_csv with the values inside the gene_box
    psl_write_some_reading_codes_here(gene_description_csv, gene_location_corrected,gene_box_corrected)

    # encode the value in a dictionary
    pseudo_bulk_for_one_tile ={'gene_name_1':aggregated_value1, 'gene_name_2':aggregated_value2,...}

    return pseudo_bulk_for_one_tile


def aggregate_pseudo_bulk_for_one_slide(WSI_root, slide_folder, suffix='.jpeg'):
    """
    Aggregates gene expression data for each tile in a WSI slide to create a pseudo-bulk profile.

    Parameters:
    - WSI_root (str): Root folder of the original Xenium WSI dataset.
    - slide_folder (str): Absolute path of the tiled slides folder.
    - suffix (str): File extension of tile images (default is '.jpeg').

    Returns:
    - None. Saves a CSV file with pseudo-bulk gene expression data for each tile.
    """
    # Extract slide name and initialize parameters (could be read dynamically if needed)
    slide_name = os.path.split(slide_folder)[1]
    tile_size = 224  # TODO: Load this from metadata if available in slide_folder
    mpp_info = 0.5  # TODO: Load this from metadata if available in slide_folder

    # Load gene description and tile locations for the slide
    gene_description_csv = load_gene_description_csv(WSI_root, slide_name)
    tile_locations = read_locations_from_tiles_for_one_slide(slide_folder, suffix)

    # Initialize a list to store results for all tiles
    aggregated_data = []

    for tile_location in tile_locations:
        # Aggregate gene data for the tile
        pseudo_bulk_for_one_tile = aggregate_pseudo_bulk_for_one_tile(
            gene_description_csv, tile_location, tile_size, mpp_info)

        # Construct a unique tile name
        tile_name = f"{tile_location[0]:05d}y_{tile_location[1]:05d}x" + suffix

        # Append tile data to aggregated data list
        tile_data = {'slide_name':slide_name, 'tile_name': tile_name}
        tile_data.update(pseudo_bulk_for_one_tile)
        aggregated_data.append(tile_data)

    # Convert aggregated data to a DataFrame and save as CSV
    aggregated_df = pd.DataFrame(aggregated_data)
    output_csv_path = os.path.join(slide_folder, "pseudo_bulk_gene_expression.csv")
    aggregated_df.to_csv(output_csv_path, index=False)
    print(f"Aggregated pseudo-bulk gene expression data saved to {output_csv_path}")


def aggregate_pseudo_bulk_for_all_slides(WSI_root, Tiled_WSI_root, suffix='.jpeg'):
    """
    Aggregates gene expression data for all slides in a given dataset.

    Parameters:
    - WSI_root (str): Root folder of the original Xenium WSI dataset.
    - Tiled_WSI_root (str): Root folder containing the tiled slide folders for each WSI.
    - suffix (str): File extension of tile images (default is '.jpeg').

    Returns:
    - None. Saves a CSV file with pseudo-bulk gene expression data for each slide.
    """
    # Iterate over each slide folder in the tiled dataset root
    for slide_folder_name in tqdm(os.listdir(Tiled_WSI_root)):
        slide_folder_path = os.path.join(Tiled_WSI_root, slide_folder_name)

        if os.path.isdir(slide_folder_path):  # Ensure it's a folder
            aggregate_pseudo_bulk_for_one_slide(WSI_root, slide_folder_path, suffix)


class Bulk_ROI_Dataset(Dataset):
    def __init__(self, root_path: str, tile_suffix: str = '.jpeg', edge_size=224, transform=None,
                 stopping_folder_name_list: list = ['thumbnails']):
        """
        Custom Dataset to load pseudo-bulk image and their gene label data for all slides.

        Parameters:
        - root_path (str): Path to the folder containing tiles and bulk label for each slide.
        - tile_suffix (str): File extension of tile images (default is '.jpeg').
        - edge_size (int): Target size for resizing images (default is 224).
        - transform (callable, optional): Optional transforms to apply to each sample.
        """
        self.root_path = root_path
        self.tile_suffix = tile_suffix

        # Find slide paths and IDs
        self.slide_paths = self.find_slide_paths_and_ids(stopping_folder_name_list=stopping_folder_name_list)
        self.slide_ids = list(self.slide_paths.keys())

        # Aggregate gene expression data across all slides into a single DataFrame
        self.tile_info_for_all_slide = pd.concat(
            [pd.read_csv(os.path.join(self.slide_paths[slide_id],
                                      "pseudo_bulk_gene_expression.csv")).assign(slide_name=slide_id)
             for slide_id in self.slide_ids], ignore_index=True)

        # Default transform (resize, to tensor, normalize)
        default_transform = transforms.Compose([
            transforms.Resize((edge_size, edge_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        ])
        self.transform = transform or default_transform

    def find_slide_paths_and_ids(self, stopping_folder_name_list=['thumbnails']):
        """
        Finds slide folder paths, stopping search in specified folders.

        Parameters:
        - stopping_folder_name_list (list): List of folder names to ignore in search.
        """
        slide_paths = {}
        for dirpath, dirnames, _ in os.walk(self.root_path):
            dirnames[:] = [d for d in dirnames if d not in stopping_folder_name_list]
            for dirname in dirnames:
                slide_folder_path = os.path.join(dirpath, dirname)
                if any(fname.endswith(self.tile_suffix) for fname in os.listdir(slide_folder_path)):
                    slide_paths[dirname] = Path(slide_folder_path)
                    break
        return slide_paths

    def __len__(self):
        return len(self.tile_info_for_all_slide)

    def __getitem__(self, idx):
        # Extract information for the current tile
        row = self.tile_info_for_all_slide.iloc[idx]
        slide_name = row['slide_name']
        tile_name = row['tile_name']
        gene_expression = row.drop(['slide_name', 'tile_name']).values.astype(float)

        # Construct image path and coordinates
        img_path = os.path.join(self.slide_paths[slide_name], tile_name)
        y, x = map(int, [tile_name.split('y')[0], tile_name.split('x')[1].split('.')[0]])
        patch_coord_yx_tensor = torch.tensor([y, x], dtype=torch.int32)

        # Load and transform the image
        with open(img_path, "rb") as f:
            patch_image = Image.open(f).convert("RGB")
            patch_image_tensor = self.transform(patch_image)

        # Prepare sample with image, gene expression data, and coordinates
        sample = {'patch_image_tensor': patch_image_tensor, 'gene_expression': gene_expression,
                  'patch_coord_yx_tensor': patch_coord_yx_tensor}
        return sample


if __name__ == '__main__':
    # build aggregate_pseudo_bulk_for_one_tile
    # then run aggregate_pseudo_bulk_for_all_slides
    # lastly test loading the bulk with pytorch dataset Bulk_ROI_Dataset
    pass