'''
MTL dataset framework       Script  ver： Oct 24th 16:00
'''

import os
import sys
from pathlib import Path

# For convenience
this_file_dir = Path(__file__).resolve().parent
sys.path.append(str(this_file_dir.parent.parent))  # Go up 2 levels
try:
    from slide_dataset_tools import *
except:
    from .slide_dataset_tools import *

import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import pandas as pd

import h5py


# pseudo Bulk dataset builder
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


# WSI MTL dataset builder
class SlideDataset(Dataset):
    def __init__(self, root_path: str, task_description_csv: str = None,
                 task_setting_folder_name: str = 'task_settings',
                 split_name: str = 'train',
                 slide_id_key='slide_id', split_target_key='split',
                 possible_suffixes=('.h5', '.pt', '.jpeg', '.jpg'),
                 stopping_folder_name_list=['thumbnails', ],
                 task_type='MTL',
                 data_name_mode=None,
                 max_tiles=None,
                 **kwargs):
        """
        Slide dataset class for retrieving slide_feature samples for different tasks.

        Each WSI is a folder (slide_folder, named by slide_id), and all cropped tiles are embedded as one .h5 file:
            h5file['features'] is a list of numpy features, each feature (can be of multiple dims: dim1, dim2, ...)
                            for transformer embedding, the feature dim is [768]
            h5file['coords_yx'] is a list of coordinates, each item is a [Y, X], Y, X is slide_feature index in WSI

        Arguments:
        ----------
        root_path : str
            The root path of the tile embeddings
        task_description_csv : label csv
        task_setting_folder_name:
        split_name : str
            The key word of patient_ids/labeled_slide_names as the split lists to build dataset
        slide_id_key : str
            The key that contains the slide_feature id
        split_target_key : str
            The key that specifies the column name for taking the split_name,
            'split_nfold-k', n is the total fold number and k is the fold index

        possible_suffixes: supported suffix for taking the samples
        stopping_folder_name_list:

        task_type: 'MTL' for building slide_feature level mtl dataset, 'embedding' for doing slide level embedding

        data_name_mode: slide name rule, default is None, 'TCGA' for TCGA names,
                        by default will try to load the config file

        max_tiles: tile taking maximum number for each slide, default is None,
                        by default will try to load the config file


        everytime it get a sample WSI:
        ----------
        sample = {'image_features': image features [N, D] tensor,
              'image_features_lens': data_dict['image_features_lens'],
              'pad_mask': data_dict['pad_mask'],
              'coords_yx': [N, 2] tensor,
              'slide_id': slide_id,
              'task_name_list': task_name_list,
              'task_description_list': task_description_list}
        """
        # add default stop list:
        stopping_folder_name_list.extend(['task-settings', 'task-settings-5folds'])

        super(SlideDataset, self).__init__(**kwargs)
        self.task_type = task_type

        self.root_path = root_path
        self.possible_suffixes = possible_suffixes

        if self.task_type == 'embedding':
            # here the slide dataset is called without requirement of task labels,
            # in this case we dont need config and csv label file
            split_name = 'all_embedding'
            self.task_name_list = 'slide_embedding'
            try:
                self.task_cfg = load_yaml_config(os.path.join(root_path, task_setting_folder_name, 'task_configs.yaml'))
                self.data_name_mode = data_name_mode or self.task_cfg.get('mode')
            except:
                self.task_cfg = None
                self.data_name_mode = data_name_mode

            # Find valid slide_feature paths that have tile encodings
            self.slide_ids, valid_sample_ids = self.get_valid_slides(None, stopping_folder_name_list, self.data_name_mode)
            self.setup_task_data(None, None, task_type=self.task_type)

        else:
            # the slide dataset need task labels
            # load task config
            self.task_cfg = load_yaml_config(os.path.join(root_path, task_setting_folder_name, 'task_configs.yaml'))
            self.data_name_mode = data_name_mode or self.task_cfg.get('mode')
            self.split_target_key = split_target_key  # the key to record the fold infor
            self.slide_id_key = slide_id_key

            # load label csv
            task_description_csv = task_description_csv or \
                                   os.path.join(root_path, task_setting_folder_name, 'task_description.csv')
            task_description_data_df = pd.read_csv(task_description_csv)
            # Get the label from CSV file with WSIs assigned with the target split (such as 'train').
            task_description_data_df = task_description_data_df[
                task_description_data_df[self.split_target_key] == split_name]
            # Set up the task
            self.task_name_list = self.task_cfg.get('tasks_to_run')
            assert self.task_name_list is not None
            # Find valid slide_feature paths that have tile encodings
            self.slide_ids, valid_sample_ids = self.get_valid_slides(task_description_data_df[self.slide_id_key].values,
                                                                     stopping_folder_name_list, self.data_name_mode)

            if self.data_name_mode == 'TCGA':
                # in tcga csv, we only have slide_id_key indicating the label to the patient,
                # therefore we need to map the slide_feature id into the csv by duplicating the labels.
                # certain patients can have multiple slides
                print('In TCGA mode, we take the patient record as label'
                      ' for the slides corresponding to the same patient')
                task_description_data_df = \
                    task_description_data_df[task_description_data_df[self.slide_id_key].isin(valid_sample_ids)]
            else:
                print('In dataset framework, we take the slide_id_key to pair slide_feature id and label csv')
                task_description_data_df = \
                    task_description_data_df[task_description_data_df[self.slide_id_key].isin(self.slide_ids)]

            self.setup_task_data(task_description_data_df, self.task_name_list, task_type=self.task_type)

        # Load from settings or set default value
        self.max_tiles = max_tiles or self.task_cfg.get('max_tiles', 10000) if self.task_cfg is not None else 10000
        self.shuffle_tiles = self.task_cfg.get('shuffle_tiles', False) if self.task_cfg is not None else False
        print('Dataset has been initialized with ' + str(len(self.slide_ids)) +
              ' slides for split:', str(split_name))

        '''
        # fixme notice this is very slow when the hard disk is in use
        # check tile distribution 
        self.check_tile_num_distribution(draw_path=os.path.join(root_path, task_setting_folder_name,
                                                                str(split_name) + '.jpeg'))
        '''

    def find_slide_paths_and_ids(self, stopping_folder_name_list):
        """
        Find slide_feature paths and their corresponding IDs.

        This operation can be slow as there are many '.jpg' files in the slide_folder.
        Therefore, when it detects one slide_folder, all files inside should not be tested again.


        stopping_folder_name_list: a list of the not-taking slide names, default is none
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

    def get_valid_slides(self, labeled_slide_names, stopping_folder_name_list, data_name_mode='TCGA'):
        """Get the slides that have tile encodings stored in the tile directory.

        labeled_slide_names: a list of the slide names, default is none for all slides
        stopping_folder_name_list: a list of the not-taking slide names, default is none

        """
        slide_paths = self.find_slide_paths_and_ids(stopping_folder_name_list=stopping_folder_name_list)

        self.slide_paths = {}

        valid_slide_ids = []
        valid_sample_ids = []
        for slide_id in slide_paths:
            slide_id_name = slide_id[0:12] if data_name_mode == 'TCGA' else slide_id
            slide_path = slide_paths[slide_id]
            if labeled_slide_names is not None:
                if slide_id_name not in labeled_slide_names:
                    # when this sample is not required in this current split
                    continue
                else:
                    if 'pt_files' in self.root_path.split('/')[-1]:
                        embedded_slide_file = slide_id.replace(".svs", "") + '.pt'
                    else:
                        embedded_slide_file = slide_id.replace(".svs", "") + '.h5'
                    embedded_slide_path = os.path.join(slide_path, embedded_slide_file)
                    if not os.path.exists(embedded_slide_path):
                        print('Data Missing for: ', slide_id)
                    else:
                        # Add to valid list
                        valid_slide_ids.append(slide_id)
                        valid_sample_ids.append(slide_id_name)
                        self.slide_paths[slide_id] = embedded_slide_path
            else:
                if 'pt_files' in self.root_path.split('/')[-1]:
                    embedded_slide_file = slide_id.replace(".svs", "") + '.pt'
                else:
                    embedded_slide_file = slide_id.replace(".svs", "") + '.h5'
                embedded_slide_path = os.path.join(slide_path, embedded_slide_file)
                if not os.path.exists(embedded_slide_path):
                    print('Data Missing for: ', slide_id)
                else:
                    # Add to valid list
                    valid_slide_ids.append(slide_id)
                    valid_sample_ids.append(slide_id_name)
                    self.slide_paths[slide_id] = embedded_slide_path

        return valid_slide_ids, valid_sample_ids

    def check_tile_num_distribution(self, draw_path):
        # fixme notice this is very slow when the hard disk is in use
        import matplotlib.pyplot as plt
        tile_num_list = []
        for slide_id in self.slide_ids:
            # Get the slide_feature path
            embedded_slide_path = self.slide_paths[slide_id]

            # Read assets from the slide_feature
            assets, _ = self.read_assets_from_h5(embedded_slide_path)
            tile_num = len(assets['coords_yx'])
            tile_num_list.append((slide_id, tile_num))

        # Sort the list based on tile numbers
        tile_num_list.sort(key=lambda x: x[1])

        # Extract slide_feature IDs and corresponding tile counts
        slide_ids_sorted, tile_counts_sorted = zip(*tile_num_list)

        # Plotting the distribution of tile numbers
        plt.figure(figsize=(12, 6))
        plt.bar(range(len(slide_ids_sorted)), tile_counts_sorted)
        plt.xticks(range(len(slide_ids_sorted)), [''] * len(slide_ids_sorted))  # Disable printing the slide_feature IDs
        plt.xlabel('Slide ID')
        plt.ylabel('Number of Tiles')
        plt.title('Distribution of Tile Numbers Across Slides')

        # Adding a horizontal orange line indicating `self.max_tiles`
        plt.axhline(y=self.max_tiles, color='orange', linestyle='--', linewidth=2,
                    label=f'Taking Max Tiles ({self.max_tiles})')

        # Add the value of `self.max_tiles` to the y-tick labels
        y_ticks = plt.yticks()[0]  # Get the current y-tick values
        plt.yticks(list(y_ticks) + [self.max_tiles])  # Add `self.max_tiles` to the y-ticks

        # Add a legend to the plot
        plt.legend()
        # Save the plot to the specified path
        plt.tight_layout()
        plt.savefig(draw_path)
        # Display the plot
        plt.show()

    '''
    fixme demo code from previous
    def prepare_single_task_data_list(self, df, split_name_list, task_name_list=['label']):
        """
        Prepare the sample for single task.
        # fixme demo code from previous
        """
        task_name = task_name_list[0]

        # Set up the label_dict
        label_dict_settings = self.task_cfg.get('label_dict', {})  # One-hot encoding dict
        assert label_dict_settings, 'No label_dict found in the task configuration'
        label_dict = label_dict_settings[task_name]

        # Set up the mappings
        assert task_name in df.columns, 'No label column found in the dataframe'
        df[task_name] = df[task_name].map(label_dict)
        n_classes = len(label_dict)  # If 1, it's a regression task

        # Get the corresponding splits
        # here split_target_key is the patient id, indication to take name form it, but its not good for multiple fold
        assert self.split_target_key in df.columns, 'No {} column found in the dataframe'.format(self.split_target_key)
        df = df[df[self.split_target_key].isin(split_name_list)]
        slide_ids = df[self.slide_id_key].tolist()
        if n_classes == 1:
            slide_labels = df[[task_name]].to_numpy().astype(int)  # Manual long-int encoding in df[['label']]
        else:
            slide_labels = df[[task_name]].to_numpy()
        return df, slide_ids, slide_labels, n_classes

    '''

    def prepare_MTL_data_list(self, task_description_csv, task_name_list):
        """Prepare the sample for multi-label task.

        return
        task_dict,
        one_hot_table,
        labels: a dict recording the labels for each slide_feature by
                loading the corresponding self.slide_id_key in csv
        """
        task_dict = self.task_cfg.get('all_task_dict')
        one_hot_table = self.task_cfg.get('one_hot_table')

        WSI_names = task_description_csv[self.slide_id_key]

        labels = {}
        for WSI_name in WSI_names:
            task_description_list = []
            # we use split_target_key to indicate fold
            loaded_task_description = task_description_csv[task_description_csv[self.slide_id_key]
                                                           == WSI_name].to_dict('records')[0]
            for task in task_name_list:
                data_type = task_dict[task]
                if not pd.isna(loaded_task_description[task]):  # In case of available label
                    if data_type == 'float':  # Regression task
                        # Convert the string to a float if it's not already a float
                        value = float(loaded_task_description[task]) if \
                            isinstance(loaded_task_description[task], str) \
                            else loaded_task_description[task]
                        task_description_list.append(torch.tensor(value))
                        # e.g., torch.tensor(0.69)
                    else:  # Classification task
                        label = loaded_task_description[task]  # e.g., label = 'lusc'
                        one_hot_label = torch.tensor(one_hot_table[task][label])
                        long_label = one_hot_label.argmax()
                        # e.g., the index of the one-hot label, e.g., torch.LongTensor(1)
                        task_description_list.append(long_label)  # e.g., torch.tensor(1)
                else:  # In case of missing label
                    if data_type == 'float':  # Regression task
                        task_description_list.append(torch.tensor(99999999.99))  # Missing label
                    else:  # Classification task
                        task_description_list.append(torch.tensor(99999999))  # Missing label

            labels[WSI_name] = task_description_list

        return task_dict, one_hot_table, labels

    def setup_task_data(self, task_description_csv=None, task_name_list=None, task_type='MTL'):
        """Prepare the sample for single task tasks_to_run or multi-task tasks_to_run.

        old demo self.prepare_single_task_data_list, the split is a list of wsi name
        """

        # todo multiple modality func
        if task_type == 'MTL':
            assert task_description_csv is not None and task_name_list is not None
            self.task_dict, self.one_hot_table, self.labels = \
                self.prepare_MTL_data_list(task_description_csv, task_name_list)
        elif task_type == 'embedding':
            self.task_dict, self.one_hot_table, self.labels = None, None, None
        else:
            raise NotImplementedError  # currently we only have task_type == 'MTL'

    def shuffle_data_pairs(self, images: torch.Tensor, coords: torch.Tensor) -> tuple:
        """Shuffle the serialized images and coordinates"""
        indices = torch.randperm(len(images))
        images_ = images[indices]
        coords_ = coords[indices]
        return images_, coords_

    def read_assets_from_h5(self, h5_path: str) -> tuple:
        """Read the assets from the h5 file"""
        assets = {}
        attrs = {}
        with h5py.File(h5_path, 'r') as f:
            for key in f.keys():
                assets[key] = f[key][:]
                if f[key].attrs is not None:
                    attrs[key] = dict(f[key].attrs)
        return assets, attrs

    def get_embedded_data_dict(self, embedding_file_path: str) -> dict:
        """Get the image_features from the path"""
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
        """Get one sample from the dataset"""
        # get the slide_feature id
        slide_id = self.slide_ids[idx]
        # get the slide_feature path
        embedded_slide_path = self.slide_paths[slide_id]

        # get the slide_feature tile embeddings
        data_dict = self.get_embedded_data_dict(embedded_slide_path)
        # get the slide_feature label
        slide_id_name = slide_id[0:12] if self.data_name_mode == 'TCGA' else slide_id
        task_description_list = self.labels[slide_id_name] if self.task_type != 'embedding' else 'None'

        # set the sample dict
        sample = {'image_features': data_dict['image_features'],
                  'image_features_lens': data_dict['image_features_lens'],
                  'pad_mask': data_dict['pad_mask'],
                  'coords_yx': data_dict['coords_yx'],
                  'slide_id': slide_id,
                  'task_name_list': self.task_name_list,
                  'task_description_list': task_description_list}

        return sample

    def get_embedded_sample_with_try(self, idx, n_try=3):
        """Get the sample with n_try, handles missing/failed sample, but not nicely"""

        for _ in range(n_try):
            try:
                one_embedded_sample = self.get_one_embedded_sample(idx)
                return one_embedded_sample
            except:
                print('Error in getting the sample, try another index')
                idx = np.random.randint(0, len(self.slide_ids))
        print('Error in getting one sample with n_try: ', n_try)
        # raise RuntimeError('Failed to get a valid sample after {} tries'.format(n_try))
        return -1

    def __len__(self):
        return len(self.slide_ids)

    def __getitem__(self, idx):
        """
        everytime it get a sample WSI:
        ----------
        sample = {'image_features': image features [N, D] tensor,
              'image_features_lens': data_dict['image_features_lens'],
              'pad_mask': data_dict['pad_mask'],
              'coords_yx': [N, 2] tensor,
              'slide_id': slide_id,
              'task_name_list': task_name_list,
              'task_description_list': task_description_list}
        """
        # fixme in this current framework the model always trained with wsi batch size of 1
        slide_level_sample = self.get_embedded_sample_with_try(idx)
        return slide_level_sample


def MTL_WSI_collate_fn(batch):  # todo havent designed for dictionary version
    # Filter out bad data (data loader return -1)

    cleaned_batch = [(data['image_features'], data['coords_yx'], data['task_description_list'], data['slide_id'])
                     for data in batch if data != -1]  # -1 for not valid return from dataset
    # If after filtering, the batch is empty, return None
    if len(cleaned_batch) == 0:
        return None
    else:
        # Use the default collate function to stack remaining data samples into batches
        return torch.utils.data.dataloader.default_collate(cleaned_batch)


if __name__ == '__main__':
    root_path = '/data/hdd_1/BigModel/TCGA-COAD-READ/Tile_embeddings/gigapath'
    task_description_csv = \
        '/data/hdd_1/BigModel/TCGA-COAD-READ/Tile_embeddings/gigapath/task-settings-5folds/20240827_TCGA_log_marker10.csv'
    slide_id_key = 'patient_id'
    split_target_key = 'fold_information_5fold-5'
    task_setting_folder_name = 'task-settings-5folds'
    data_name_mode = 'TCGA'
    max_tiles = 10000

    dataset_name = 'TCGA-COAD-READ',
    tasks_to_run = 'iCMS%CMS%MSI.status%EPCAM%COL3A1%CD3E%PLVAP%C1QA%IL1B%MS4A1%CD79A'.split('%')
    task_type = 'MTL'  # MTL or embedding, if its embedding, the task config is not composable

    '''
    build_split_and_task_configs(root_path, task_description_csv, dataset_name, tasks_to_run,
                                 slide_id_key, split_target_key, task_setting_folder_name, mode)
    '''

    # check the dataset
    Train_dataset = SlideDataset(root_path, task_description_csv,
                                 task_setting_folder_name=task_setting_folder_name,
                                 split_name='Train', slide_id_key=slide_id_key,
                                 split_target_key=split_target_key,
                                 data_name_mode=data_name_mode, max_tiles=max_tiles,
                                 task_type=task_type)
    Val_dataset = SlideDataset(root_path, task_description_csv,
                               task_setting_folder_name=task_setting_folder_name,
                               split_name='Val', slide_id_key=slide_id_key,
                               split_target_key=split_target_key,
                               data_name_mode=data_name_mode, max_tiles=max_tiles,
                               task_type=task_type)

    print(Train_dataset.get_embedded_sample_with_try(20))
    dataloaders = {
        'Train': torch.utils.data.DataLoader(Train_dataset, batch_size=1,
                                             collate_fn=MTL_WSI_collate_fn,
                                             shuffle=True, num_workers=2, drop_last=True),
        'Val': torch.utils.data.DataLoader(Val_dataset, batch_size=1,
                                           collate_fn=MTL_WSI_collate_fn,
                                           shuffle=False, num_workers=2, drop_last=True)}
