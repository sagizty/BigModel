'''
MTL dataset framework       Script  ver： Aug 22nd 19:30
'''

import os
import sys
from pathlib import Path

# For convinience
this_file_dir = Path(__file__).resolve().parent
sys.path.append(str(this_file_dir.parent.parent))  # Go up 2 levels
try:
    from slide_dataset_tools import *
except:
    from .slide_dataset_tools import *

import torch
from torch.utils.data import Dataset
import pandas as pd


# MTL dataset builder
class SlideDataset(Dataset):
    def __init__(self, root_path: str, task_description_csv: str = None,
                 task_setting_folder_name: str = 'task_settings',
                 split_name: str = 'train',
                 slide_id_key='slide_id', split_target_key='split',
                 possible_suffixes=('.h5', '.pt', '.jpeg', '.jpg'),
                 stopping_folder_name_list=['thumbnails', ],
                 dataset_type='MTL',
                 max_tiles=10000,
                 **kwargs):
        """
        Slide dataset class for retrieving slide samples for different tasks.

        Each WSI is a folder (slide_folder, named by slide_id), and all cropped tiles are embedded as one .h5 file:
            h5file['features'] is a list of numpy features, each feature (can be of multiple dims: dim1, dim2, ...)
                            for transformer embedding, the feature dim is [768]
            h5file['coords_yx'] is a list of coordinates, each item is a [Y, X], Y, X is patch index in WSI

        Arguments:
        ----------
        root_path : str
            The root path of the tile embeddings
        task_description_csv : label csv
        task_setting_folder_name:
        split_name : str
            The key word of patient_ids/labeled_slide_names as the split lists to build dataset
        slide_id_key : str
            The key that contains the slide id
        split_target_key : str
            The key that specifies the column name for taking the split_name,
            'split_nfold-k', n is the total fold number and k is the fold index

        possible_suffixes: supported suffix for taking the samples
        dataset_type: 'MTL' for building slide level mtl dataset

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
        super(SlideDataset, self).__init__(**kwargs)
        self.root_path = root_path
        self.possible_suffixes = possible_suffixes
        self.task_cfg = load_yaml_config(os.path.join(root_path, task_setting_folder_name, 'task_configs.yaml'))
        self.split_target_key = split_target_key  # the key to record the fold infor
        self.slide_id_key = slide_id_key
        self.mode = self.task_cfg.get('mode')

        task_description_csv = task_description_csv or \
                               os.path.join(root_path, task_setting_folder_name, 'task_description.csv')
        task_description_data_df = pd.read_csv(task_description_csv)
        # Get the label from CSV file with WSIs assigned with the target split (such as 'train').
        task_description_data_df = task_description_data_df[
            task_description_data_df[self.split_target_key] == split_name]

        # Find valid slide paths that have tile encodings
        self.slide_ids, valid_sample_ids = self.get_valid_slides(task_description_data_df[slide_id_key].values,
                                                                 stopping_folder_name_list, mode)

        if self.mode == 'TCGA':
            # in tcga csv, we only have slide_id_key indicating the label to the patient,
            # therefore we need to map the slide id into the csv by duplicating the labels.
            # certain patients can have multiple slides
            print('In TCGA mode, we take the patient record as label'
                  ' for the slides corresponding to the same patient')
            task_description_data_df = \
                task_description_data_df[task_description_data_df[self.slide_id_key].isin(valid_sample_ids)]
        else:
            print('In dataset framework, we take the slide_id_key to pair slide id and label csv')
            task_description_data_df = \
                task_description_data_df[task_description_data_df[self.slide_id_key].isin(self.slide_ids)]

        # Set up the task
        self.task_name_list = self.task_cfg.get('tasks_to_run')
        assert self.task_name_list is not None

        self.setup_task_data(task_description_data_df, self.task_name_list, dataset_type=dataset_type)

        # Load from settings or set default value
        self.max_tiles = max_tiles or self.task_cfg.get('max_tiles', 1000)
        self.shuffle_tiles = self.task_cfg.get('shuffle_tiles', False)
        print('Dataset has been initialized with ' + str(len(self.slide_ids)) +
              ' slides for split:', split_name)
        # check tile distribution
        self.check_tile_num_distribution(draw_path=os.path.join(root_path, task_setting_folder_name,
                                                                split_name+'.jpeg'))

    def find_slide_paths_and_ids(self, stopping_folder_name_list):
        """
        Find slide paths and their corresponding IDs.

        This operation can be slow as there are many '.jpg' files in the slide_folder.
        Therefore, when it detects one slide_folder, all files inside should not be tested again.
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

    def get_valid_slides(self, labeled_slide_names, stopping_folder_name_list, mode='TCGA'):
        """Get the slides that have tile encodings stored in the tile directory."""
        slide_paths = self.find_slide_paths_and_ids(stopping_folder_name_list)

        self.slide_paths = {}

        valid_slide_ids = []
        valid_sample_ids = []
        for slide_id in slide_paths:
            slide_id_name = slide_id[0:12] if mode == 'TCGA' else slide_id
            slide_path = slide_paths[slide_id]

            if slide_id_name not in labeled_slide_names:
                # this sample is not required in this current split
                pass
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
        import matplotlib.pyplot as plt

        tile_num_list = []
        for slide_id in self.slide_ids:
            # Get the slide path
            embedded_slide_path = self.slide_paths[slide_id]

            # Read assets from the slide
            assets, _ = self.read_assets_from_h5(embedded_slide_path)
            tile_num = len(assets['coords_yx'])
            tile_num_list.append((slide_id, tile_num))

        # Sort the list based on tile numbers
        tile_num_list.sort(key=lambda x: x[1])

        # Extract slide IDs and corresponding tile counts
        slide_ids_sorted, tile_counts_sorted = zip(*tile_num_list)

        # Plotting the distribution of tile numbers
        plt.figure(figsize=(12, 6))
        plt.bar(range(len(slide_ids_sorted)), tile_counts_sorted)
        plt.xticks(range(len(slide_ids_sorted)), [''] * len(slide_ids_sorted))  # Disable printing the slide IDs
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
        labels: a dict recording the labels for each slide by
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

    def setup_task_data(self, task_description_csv, task_name_list, dataset_type='MTL'):
        """Prepare the sample for single task tasks_to_run or multi-task tasks_to_run.

        old demo self.prepare_single_task_data_list, the split is a list of wsi name
        """

        # todo multiple modality func
        if dataset_type == 'MTL':
            self.task_dict, self.one_hot_table, self.labels = \
                self.prepare_MTL_data_list(task_description_csv, task_name_list)
        else:
            raise NotImplementedError  # currently we only have dataset_type == 'MTL'

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
        # get the slide id
        slide_id = self.slide_ids[idx]
        # get the slide path
        embedded_slide_path = self.slide_paths[slide_id]

        # get the slide tile embeddings
        data_dict = self.get_embedded_data_dict(embedded_slide_path)

        # get the slide label
        slide_id_name = slide_id[0:12] if self.mode == 'TCGA' else slide_id
        task_description_list = self.labels[slide_id_name]

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
    root_path = '/data/BigModel/embedded_datasets/'
    task_description_csv = \
        '/home/zhangty/Desktop/BigModel/prov-gigapath/PuzzleAI/Archive/dataset_csv/TCGA_Log_Transcriptome_Final.csv'
    slide_id_key = 'patient_id'
    split_target_key = 'fold_information'
    task_setting_folder_name = 'task-settings'
    mode = 'TCGA'

    dataset_name = 'TCGA-COAD-READ',
    tasks_to_run = 'iCMS%CMS%MSI.status%EPCAM%COL3A1%CD3E%PLVAP%C1QA%IL1B%MS4A1%CD79A'.split('%')

    '''
    build_split_and_task_configs(root_path, task_description_csv, dataset_name, tasks_to_run,
                                 slide_id_key, split_target_key, task_setting_folder_name, mode)
    '''

    # check the dataset
    Train_dataset = SlideDataset(root_path, task_description_csv,
                                 task_setting_folder_name=task_setting_folder_name,
                                 split_name='train', slide_id_key=slide_id_key,
                                 split_target_key=split_target_key)
    Val_dataset = SlideDataset(root_path, task_description_csv,
                               task_setting_folder_name=task_setting_folder_name,
                               split_name='val', slide_id_key=slide_id_key,
                               split_target_key=split_target_key)

    print(Train_dataset.get_embedded_sample_with_try(20))
    dataloaders = {
        'Train': torch.utils.data.DataLoader(Train_dataset, batch_size=1,
                                             collate_fn=MTL_WSI_collate_fn,
                                             shuffle=True, num_workers=2, drop_last=True),
        'Val': torch.utils.data.DataLoader(Val_dataset, batch_size=1,
                                           collate_fn=MTL_WSI_collate_fn,
                                           shuffle=False, num_workers=2, drop_last=True)}
