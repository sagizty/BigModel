import os
import shutil
from pathlib import Path
from multiprocessing import Pool, cpu_count
from tqdm import tqdm


def copy_slide(slide_info):
    """ Function to copy a single slide to a new directory """
    slide_image_path, new_slide_root = slide_info
    wsi_id = os.path.split(slide_image_path)[-1].split('.')[0]

    # Create a new directory in the new_slide_root with the wsi_id as the folder name
    new_wsi_dir = os.path.join(new_slide_root, wsi_id)
    os.makedirs(new_wsi_dir, exist_ok=True)  # Ensure the directory is created

    # Copy the slide file to the newly created directory
    shutil.copy(slide_image_path, new_wsi_dir)


def copy_TCGA_raw_to_new_path(original_slide_root, new_slide_root, slide_suffixes=['.svs']):
    """
    Traverse the original_slide_root directory to find all WSI files with the given suffixes
    copy the WSIs to the new_slide_root each in a folder of its id (TCGA)
    Use multiprocessing to speed up the copy process.
    Show progress with tqdm.
    """
    slide_files = []

    # Traverse the slide_root directory to find all files with the given suffixes
    for root, _, files in os.walk(original_slide_root):
        for file in files:
            if any(file.endswith(suffix) for suffix in slide_suffixes):
                slide_image_path = Path(root) / file
                slide_files.append((slide_image_path, new_slide_root))  # Store the file path and destination

    # Set up tqdm progress bar
    with tqdm(total=len(slide_files), desc="Copying slides", unit="file") as pbar:
        # Use multiprocessing to copy the files
        num_workers = min(cpu_count(),
                          len(slide_files))  # Use available CPU cores, or number of files, whichever is smaller
        with Pool(num_workers) as pool:
            for _ in pool.imap_unordered(copy_slide, slide_files):
                pbar.update(1)  # Update the progress bar after each file is copied


'''
# Example usage:
copy_TCGA_raw_to_new_path('/data/hdd_1/ITH/WSI-raw/TCGA_LUAD_WSIs',
                          '/data/hdd_1/BigModel/TCGA-LUAD-LUSC/TCGA-LUAD-raw')
'''


def count_tiles_for_all_slides(root_path='/data/hdd_1/BigModel/TCGA-LUAD-LUSC/tiles_datasets'):
    import os
    import pandas as pd
    import matplotlib.pyplot as plt
    # Dictionary to store the number of tiles for each WSI folder
    tile_counts = {}

    # Loop through all folders in the root path
    for folder in os.listdir(root_path):
        folder_path = os.path.join(root_path, folder)

        # Check if it's a directory and contains a dataset.csv file
        if os.path.isdir(folder_path):
            dataset_file = os.path.join(folder_path, 'dataset.csv')
            if os.path.exists(dataset_file):
                # Load the dataset.csv file
                df = pd.read_csv(dataset_file)

                # Count the number of tiles (rows in the CSV) and store it
                tile_counts[folder] = len(df)

    # Convert the dictionary to a pandas DataFrame for plotting
    tile_counts_df = pd.DataFrame(list(tile_counts.items()), columns=['WSI_Folder', 'Tile_Count'])

    # Sort the dataframe by tile count for better visualization
    tile_counts_df = tile_counts_df.sort_values(by='Tile_Count', ascending=False)

    # Plot a bar chart to show the distribution of tile counts across WSI folders
    plt.figure(figsize=(12, 8))
    plt.bar(tile_counts_df['WSI_Folder'], tile_counts_df['Tile_Count'])
    plt.xlabel('WSI Folder')
    plt.ylabel('Number of Tiles')
    plt.title('Tile Count Distribution across WSI Folders')
    plt.xticks(rotation=90, ha='right')
    plt.tight_layout()

    # Show the plot
    plt.show()


def check_folder_name(directory_path='/data/hdd_1/BigModel/TCGA-LUAD-LUSC/tiles_datasets',
                      csv_file_path='/data/hdd_1/BigModel/TCGA-LUAD-LUSC/tcga_folders.csv',
                      folder_startswith='TCGA'):
    """
    :param directory_path: Define the directory path
    :param csv_file_path:Path to save the CSV file
    :param folder_startswith:
    """
    import os
    import csv
    # Get all folder names that start with 'TCGA'
    tcga_folders = [folder for folder in os.listdir(directory_path) if folder.startswith(folder_startswith)]

    # Save the folder names to a CSV file
    with open(csv_file_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Folder Name'])  # Write the header
        for folder in tcga_folders:
            writer.writerow([folder])

    print(f'Successfully saved {len(tcga_folders)} folder names to {csv_file_path}')


