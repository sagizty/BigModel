"""
tools for slide level dataset      Script  ver： Aug 4th 10:00

build and load task config
"""
import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
import h5py
import yaml  # Ensure pyyaml is installed: pip install pyyaml


def build_yaml_config_from_csv(task_description_path, output_dir, dataset_name='lung-mix',
                               setting='MTL', max_tiles=1000000, shuffle_tiles=True,
                               excluding_list=('WSI_name', 'split',)):
    """
    Build a YAML configuration file from a CSV file containing task descriptions.

    Parameters:
    task_description_path (str): Path to the task_description .csv file.
    output_dir (str): Output directory for the YAML file.
    dataset_name (str): Name of the dataset. Default is 'lung-mix'.
    setting (str): Setting type (e.g., 'MTL'). Default is 'MTL'.
    max_tiles (int): Maximum number of tiles. Default is 1000000.
    shuffle_tiles (bool): Whether to shuffle tiles or not. Default is True.
    excluding_list (tuple): List of columns to exclude. Default is ('WSI_name', 'split').
    """

    task_description = pd.read_csv(task_description_path)
    one_hot_dict, all_task_dict = {}, {}

    tasks = task_description.columns.tolist()
    for task in tasks:
        if task in excluding_list:
            continue

        content_list = task_description[task].unique()
        content_list = [x for x in content_list if not pd.isna(x)]
        content_list = sorted(content_list)

        try:
            for content in content_list:
                float(content)
            all_task_dict[task] = 'float'
        except ValueError:
            if len(content_list) > 1:
                all_task_dict[task] = 'list'
                value_list = np.eye(len(content_list), dtype=int).tolist()

                one_hot_dict[task] = {}
                for idx, content in enumerate(content_list):
                    one_hot_dict[task][content] = value_list[idx]

    config = {
        'name': dataset_name,
        'setting': setting,
        'all_task_dict': all_task_dict,
        'one_hot_dict': one_hot_dict,
        'max_tiles': max_tiles,
        'shuffle_tiles': shuffle_tiles
    }

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    yaml_output_path = os.path.join(output_dir, 'task_configs.yaml')
    if os.path.exists(yaml_output_path):
        os.remove(yaml_output_path)

    with open(yaml_output_path, 'w') as file:
        yaml.dump(config, file, default_flow_style=False)

    return all_task_dict, one_hot_dict

def load_yaml_config(yaml_path):
    """Load the YAML configuration file."""
    with open(yaml_path, 'r') as file:
        config = yaml.safe_load(file)
    return config


def load_task_settings(root_path, task_setting_folder_name='task_settings', splits='train'):
    """Load task settings including task description CSV and YAML configuration."""
    task_description_path = os.path.join(root_path, task_setting_folder_name, 'task_description.csv')
    data_df = pd.read_csv(task_description_path)
    task_config = load_yaml_config(os.path.join(root_path, task_setting_folder_name, 'task_configs.yaml'))
    slide_id_key = 'slide_id'
    split_target_key = 'pat_id'
    return data_df, root_path, splits, task_config, slide_id_key, split_target_key

class SlideDataset(Dataset):
    def __init__(self, data_df: pd.DataFrame, root_path: str, splits: list, task_config: dict,
                 slide_id_key='slide_id', split_target_key='pat_id', possible_suffixes=('.h5', '.pt', '.jpeg', '.jpg'),
                 stopping_folder_name_list=['thumbnails'], **kwargs):
        """
        Slide dataset class for retrieving slide samples for different tasks.

        Each WSI is a folder (slide_folder, named by slide_id), and all cropped tiles are embedded as one .h5 file:
            h5file['features'] is a list of numpy features, each feature (can be of multiple dims: dim1, dim2, ...)
                            for transformer embedding, the feature dim is [768]
            h5file['coords_yx'] is a list of coordinates, each item is a [Y, X], Y, X is patch index in WSI

        Arguments:
        ----------
        data_df : pd.DataFrame
            The dataframe that contains the slide sample, from a csv file of slide and labels
        root_path : str
            The root path of the tile embeddings
        task_config : dict
            The task configuration dictionary
        splits : list
            The list of patient_ids/slide_ids as the split lists to build dataset
        slide_id_key : str
            The key that contains the slide id
        split_target_key : str
            The key that specifies the column name for taking the splits
        """
        super(SlideDataset, self).__init__(**kwargs)

        self.root_path = root_path
        self.possible_suffixes = possible_suffixes
        self.task_cfg = task_config
        self.split_target_key = split_target_key
        self.slide_id_key = slide_id_key

        # Find valid slide paths
        self.slide_paths = self.find_slide_paths_and_ids(stopping_folder_name_list)
        self.slide_ids = list(self.slide_paths.keys())

        # Get slides that have tile encodings
        valid_slide_ids = self.get_valid_slides(data_df[self.slide_id_key].values)
        data_df = data_df[data_df[self.slide_id_key].isin(valid_slide_ids)]

        # Set up the task
        task_name_list = task_config.get('setting', ['label'])
        self.setup_task_data(data_df, splits, task_name_list)

        # Load from settings or set default value
        self.max_tiles = task_config.get('max_tiles', 1000)
        self.shuffle_tiles = task_config.get('shuffle_tiles', False)
        print('Dataset has been initialized!')

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

    def get_valid_slides(self, slide_ids):
        """Get the slides that have tile encodings stored in the tile directory."""
        valid_slide_ids = []
        for slide_id in slide_ids:
            if 'pt_files' in self.root_path.split('/')[-1]:
                embedded_slide_file = slide_id.replace(".svs", "") + '.pt'
            else:
                embedded_slide_file = slide_id.replace(".svs", "") + '.h5'

            embedded_slide_path = os.path.join(self.slide_paths[slide_id], embedded_slide_file)
            if not os.path.exists(embedded_slide_path):
                print('Missing: ', embedded_slide_path)
            else:
                # Add to valid list
                valid_slide_ids.append(slide_id)
                self.embedded_slide_paths[slide_id] = embedded_slide_path

        return valid_slide_ids

    def prepare_single_task_data_list(self, df, splits, task_name_list=['label']):
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
        assert self.split_target_key in df.columns, 'No {} column found in the dataframe'.format(self.split_target_key)
        df = df[df[self.split_target_key].isin(splits)]
        slide_ids = df[self.slide_id_key].tolist()
        if n_classes == 1:
            slide_labels = df[[task_name]].to_numpy().astype(int)  # Manual long-int encoding in df[['label']]
        else:
            slide_labels = df[[task_name]].to_numpy()
        return df, slide_ids, slide_labels, n_classes

    def prepare_MTL_data_list(self, task_description_csv, splits, task_name_list):
        """Prepare the sample for multi-label task."""
        task_dict = self.task_cfg.get('all_task_dict')
        one_hot_dict = self.task_cfg.get('one_hot_dict')

        # Get the label from CSV file with WSIs assigned with the target split (such as train).
        task_description_csv = task_description_csv[task_description_csv[self.split_target_key] == splits]
        WSI_names = task_description_csv[self.slide_id_key]

        labels = []
        for WSI_name in WSI_names:
            task_description_list = []

            loaded_task_description = task_description_csv[task_description_csv[self.slide_id_key]
                                                           == WSI_name].to_dict('records')[0]

            for task in task_name_list:
                data_type = task_dict[task]
                if not pd.isna(loaded_task_description[task]):  # In case of available label
                    if data_type == 'float':  # Regression task
                        task_description_list.append(torch.tensor(loaded_task_description[task]))
                        # e.g., torch.tensor(0.69)
                    else:  # Classification task
                        label = loaded_task_description[task]  # e.g., label = 'lusc'
                        one_hot_label = torch.tensor(one_hot_dict[task][label])
                        long_label = one_hot_label.argmax()
                        # e.g., the index of the one-hot label, e.g., torch.LongTensor(1)
                        task_description_list.append(long_label)  # e.g., torch.tensor(1)
                else:  # In case of missing label
                    if data_type == 'float':  # Regression task
                        task_description_list.append(torch.tensor(99999999.99))  # Missing label
                    else:  # Classification task
                        task_description_list.append(torch.tensor(99999999))  # Missing label

            labels.append(task_description_list)

        return task_description_csv, WSI_names, labels, None

    def setup_task_data(self, df, splits, task_name_list):
        """Prepare the sample for single task setting or multi-task setting."""
        # todo multiple modality func
        if len(task_name_list) == 1:
            prepare_data_func = self.prepare_single_task_data_list
        elif len(task_name_list) > 1:
            prepare_data_func = self.prepare_MTL_data_list
        else:
            raise ValueError('Invalid task: {}'.format(task_name_list))

        self.slide_data, self.slide_ids, self.labels, self.n_classes = prepare_data_func(df, splits, task_name_list)
        # reset the embedded_slide_paths dict
        self.embedded_slide_paths = {slide_id: self.embedded_slide_paths[slide_id] for slide_id in self.slide_ids}

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

    def get_slide_name_from_path(self, sld: str) -> str:
        """Get the slide name from the slide path"""
        slide_name = os.path.basename(sld).split('.h5')[0]
        return slide_name

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
        embedded_slide_path = self.embedded_slide_paths[slide_id]

        # get the slide tile embeddings
        data_dict = self.get_embedded_data_dict(embedded_slide_path)
        # get the slide label
        label = torch.from_numpy(self.labels[idx])
        # set the sample dict
        sample = {'image_features': data_dict['image_features'],
                  'image_features_lens': data_dict['image_features_lens'],
                  'pad_mask': data_dict['pad_mask'],
                  'coords_yx': data_dict['coords_yx'],
                  'slide_id': slide_id,
                  'labels': label}
        return sample

    def get_embedded_sample_with_try(self, idx, n_try=3):
        """Get the sample with n_try, handles missing/failed sample, but not nicely"""
        for _ in range(n_try):
            try:
                one_embedded_sample = self.get_one_embedded_sample(idx)
                return one_embedded_sample
            except:
                print('Error in getting the sample, try another index')
                idx = np.random.randint(0, len(self.slide_data))
        print('Error in getting one sample with n_try: ', n_try)
        raise RuntimeError('Failed to get a valid sample after {} tries'.format(n_try))

    def __len__(self):
        return len(self.slide_data)

    def __getitem__(self, idx):
        slide_level_sample = self.get_embedded_sample_with_try(idx)
        return slide_level_sample

