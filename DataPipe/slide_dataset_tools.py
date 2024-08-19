"""
tools for slide level dataset      Script  ver： Aug 19th 20:00

build and load task config
"""
import os
import random
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
import h5py
import yaml  # Ensure pyyaml is installed: pip install pyyaml
from sklearn.model_selection import GroupKFold


def read_df_from_file(file_path: str):
    """Read file into a dataframe

    Args:
        file_path (str): Read file path.

    Returns:
        df: dataframe object
    """
    file_type = file_path.split('.')[-1]

    if file_type == 'tsv':
        df = pd.read_csv(file_path, sep='\t')
    elif file_type == 'csv':
        df = pd.read_csv(file_path)
    elif file_type == 'txt':
        df = pd.read_csv(file_path, sep='\t')
    else:
        raise ValueError(f'{file_type}: File type not supported.')

    # Convert to numeric if possible
    df.replace(to_replace="'--", value=None, inplace=True)
    df = df.apply(pd.to_numeric, errors='ignore')

    return df


# csv data split tools:
def write_csv_data(task_description_csv, id_key, id_data, key='split', val='train'):
    """
    Edit the CSV file by adding (if not there, otherwise edit) a column name of key (such as 'split')

    Parameters:
    - task_description_csv: Path to the CSV file.
    - id_key: The name of the column that contains the IDs to match with id_data.
    - id_data: A list of values corresponding to the id_key column, for which the key column should be updated.
    - key: The name of the column to add or update. Defaults to 'split'.
    - val: The value to set in the key column for the matching rows. Defaults to 'train'.
    """

    # Load the CSV into a DataFrame
    df = pd.read_csv(task_description_csv)

    # If the key column does not exist, create it and fill with empty strings, else will rewrite
    if key not in df.columns:
        df[key] = ""

    # Update the rows where the id_key matches any of the values in id_data
    df.loc[df[id_key].isin(id_data), key] = val

    # Write the updated DataFrame back to the CSV
    df.to_csv(task_description_csv, index=False)


def build_data_split_for_csv(task_description_csv, slide_id_key='slide_id', test_ratio=0.2, k=1, mode='TCGA',
                             key='split'):
    """
        Edit the csv file by adding n new columns to indicate the split information for k-fold

        # name of the n new columns: 'split_nfold-k'
                                    n is the total fold number and k is the fold index
        if n == 1, only have singel fold data, we add a column of 'split'

        the value is 'train' or 'val' or 'test'
    """
    if k > 1:
        n_splits = k
    else:
        # fixme when k=1, we use single fold of 5-fold as for train val test evaluation
        n_splits = 5

    # Get a list of all WSI samples
    all_wsi_folders = list(pd.read_csv(task_description_csv)[slide_id_key])

    # Index the WSIs in a dict by the patient names
    patient_wsis = {}
    # all_wsi_patient_names = []
    for sample in all_wsi_folders:
        patient_name = sample[:12] if mode == 'TCGA' else sample[:]
        # Assuming the first 12 characters are the patient name in TCGA
        # all_wsi_patient_names.append(patient_name)
        if patient_name not in patient_wsis:
            patient_wsis[patient_name] = []
        patient_wsis[patient_name].append(sample)

    # Shuffle the patient names
    shuffled_patients = list(patient_wsis.keys())
    random.shuffle(shuffled_patients)

    # duplicate the patient with multiple samples to creat a list of samples' patient names
    name_of_the_samples_shuffled_by_patients = []
    list_of_the_samples_shuffled_by_patients = []
    for patient_name in shuffled_patients:
        for sample in patient_wsis[patient_name]:
            name_of_the_samples_shuffled_by_patients.append(patient_name)
            list_of_the_samples_shuffled_by_patients.append(sample)
    assert len(name_of_the_samples_shuffled_by_patients) == len(all_wsi_folders)

    # Calculate number of samples at test set
    total_samples_num = len(all_wsi_folders)
    test_samples_num = int(total_samples_num * test_ratio)
    # get the patients who should be in the test set
    last_name = name_of_the_samples_shuffled_by_patients[test_samples_num - 1]
    # get the first name's index after testing set
    trigger = False
    for index in range(total_samples_num):
        patient_name = name_of_the_samples_shuffled_by_patients[index]
        if patient_name == last_name:
            trigger = True
        if trigger and patient_name != last_name:
            break

    test_data = list_of_the_samples_shuffled_by_patients[0:index]
    test_names = name_of_the_samples_shuffled_by_patients[0:index]
    test_patient_names_noduplicate = list(dict.fromkeys(test_names))

    trainval_samples = np.array(list_of_the_samples_shuffled_by_patients[index:])
    trainval_patients = np.array(name_of_the_samples_shuffled_by_patients[index:])

    # Initialize GroupKFold
    group_kfold = GroupKFold(n_splits=n_splits)

    # Split the data
    print('Total samples num:', len(all_wsi_folders), 'Total patients num:', len(shuffled_patients))
    for fold, (train_idx, val_idx) in enumerate(group_kfold.split(trainval_samples, groups=trainval_patients)):
        print(f"Fold {fold + 1}")
        # print(f"TRAIN: {train_idx}, VALIDATION: {val_idx}")
        train_data = list(trainval_samples[train_idx])
        train_patient_names = list(trainval_patients[train_idx])
        train_patient_names_noduplicate = list(dict.fromkeys(train_patient_names))

        val_data = list(trainval_samples[val_idx])
        val_patient_names = list(trainval_patients[val_idx])
        val_patient_names_noduplicate = list(dict.fromkeys(val_patient_names))
        print(f"TRAIN samples num: {len(train_data)}, "
              f"TRAIN patients num: {len(train_patient_names_noduplicate)}")
        print(f"VALIDATION samples num: {len(val_data)}, "
              f"VALIDATION patients num: {len(val_patient_names_noduplicate)}")

        if k == 1:
            write_csv_data(task_description_csv, id_key=slide_id_key, id_data=train_data, key=key, val='train')
            write_csv_data(task_description_csv, id_key=slide_id_key, id_data=val_data, key=key, val='val')
            write_csv_data(task_description_csv, id_key=slide_id_key, id_data=test_data, key=key, val='test')
            break
        else:
            write_csv_data(task_description_csv, id_key=slide_id_key, id_data=train_data,
                           key=key + '_{}fold-{}'.format(k, fold + 1), val='train')
            write_csv_data(task_description_csv, id_key=slide_id_key, id_data=val_data,
                           key=key + '_{}fold-{}'.format(k, fold + 1), val='val')
            write_csv_data(task_description_csv, id_key=slide_id_key, id_data=test_data,
                           key=key + '_{}fold-{}'.format(k, fold + 1), val='test')

    print('\nTEST samples num:', len(test_data), 'TEST patients:', len(test_patient_names_noduplicate), )


# task config tools:
def build_task_config_settings(df, new_labels, one_hot_table={}, all_task_dict={}, max_possible_values=100):
    assert all(label in df.columns for label in new_labels)

    selected_new_labels = []

    for label in new_labels:
        # new label should not be in existing config
        if label in one_hot_table or label in all_task_dict:
            raise ValueError(f'Duplicate label: {label}')

        # get the list of all possible values under the current column
        content_list = list(df[label].value_counts().keys())  # this also removes the duplicates
        # change all value type to string
        valid_content_list = [str(i) for i in content_list if i != 'missing in csv']
        # fixme this is to handel bug outside

        try:
            # ensure all can be converted to float
            for content in valid_content_list:
                tmp = float(content)
        except:
            # consider as classification task if any data cannot be transformed into float.
            str_flag = True
        else:
            str_flag = False

        if not str_flag:
            all_task_dict[label] = 'float'
            print(f'Regression task added to task settings: {label}')
        else:  # maybe it's a cls task
            # skip if too many possible values
            if len(valid_content_list) > max_possible_values:
                continue  # jump this label
            # skip if the value is constant
            elif len(valid_content_list) == 1:
                continue  # jump this label
            # confirm its a valid cls task
            all_task_dict[label] = 'list'
            # generate task settings
            value_list = np.eye(len(valid_content_list), dtype=int)
            value_list = value_list.tolist()
            idx = 0
            one_hot_table[label] = {}
            for content in valid_content_list:
                one_hot_table[label][content] = value_list[idx]
                idx += 1
            print(f'Classification task added to task settings: {label}')

        selected_new_labels.append(label)

    return one_hot_table, all_task_dict, selected_new_labels


def build_yaml_config_from_csv(task_description_csv, output_dir, dataset_name='lung-mix',
                               tasks_to_run=None, max_tiles=1000000, shuffle_tiles=True,
                               excluding_list=('WSI_name', 'split',)):
    """
    Build a YAML configuration file from a CSV file containing task descriptions.

    Parameters:
    task_description_csv (str): Path to the task_description .csv file.
    output_dir (str): Output directory for the YAML file. (task-settings path)

    dataset_name (str): Name of the dataset. Default is 'lung-mix'.
    tasks_to_run (str): Setting type (e.g., 'MTL'). Default is 'MTL'.
    max_tiles (int): Maximum number of tiles. Default is 1000000.
    shuffle_tiles (bool): Whether to shuffle tiles or not. Default is True.
    excluding_list (tuple): List of columns to exclude. Default is ('WSI_name', ...).
                            the attribute starts with 'split' will be ignored as they are designed for control split
                            EG: 'split_nfold-k', n is the total fold number and k is the fold index
    """

    try:
        task_description = read_df_from_file(task_description_csv)
    except:  # no valid label selected
        raise ValueError('Invalid input!', task_description_csv)

    one_hot_table, all_task_dict = {}, {}
    excluding_list = list(excluding_list)
    excluding_list.append(slide_id_key)

    # select columns in csv to be used as the labels.
    # By default, all columns except slide_id_key will be used as label.
    tentative_task_labels = [col for col in task_description.columns if col not in excluding_list]

    one_hot_table, all_task_dict, selected_new_labels = \
        build_task_config_settings(task_description, tentative_task_labels, one_hot_table, all_task_dict)

    print(f'#' * 30)
    print(f'Add labels to config: {selected_new_labels}')
    print(f'#' * 30)

    config = {
        'name': dataset_name,
        'tasks_to_run': tasks_to_run,
        'all_task_dict': all_task_dict,
        'one_hot_table': one_hot_table,
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

    return all_task_dict, one_hot_table


def load_yaml_config(yaml_path):
    """Load the YAML configuration file."""
    with open(yaml_path, 'r') as file:
        config = yaml.safe_load(file)
    return config


# MTL dataset builder
class SlideDataset(Dataset):
    def __init__(self, root_path: str, task_description_csv: str = None,
                 task_setting_folder_name: str = 'task_settings',
                 split_name: str = 'train',
                 slide_id_key='slide_id', split_target_key='split',
                 possible_suffixes=('.h5', '.pt', '.jpeg', '.jpg'), mode: str = 'TCGA',
                 stopping_folder_name_list=['thumbnails', ], **kwargs):
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
        split_name : str
            The key word of patient_ids/labeled_slide_names as the split lists to build dataset
        slide_id_key : str
            The key that contains the slide id
        split_target_key : str
            The key that specifies the column name for taking the split_name,
            'split_nfold-k', n is the total fold number and k is the fold index


        everytime it get a sample WSI:
        ----------
        sample = {'image_features': data_dict['image_features'],
                  'image_features_lens': data_dict['image_features_lens'],
                  'pad_mask': data_dict['pad_mask'],
                  'coords_yx': data_dict['coords_yx'],
                  'slide_id': slide_id,
                  'task_name_list':self.task_name_list,
                  'task_description_list': task_description_list}
        """
        super(SlideDataset, self).__init__(**kwargs)

        task_description_csv = task_description_csv or \
                               os.path.join(root_path, task_setting_folder_name, 'task_description.csv')
        data_df = pd.read_csv(task_description_csv)

        self.root_path = root_path
        self.possible_suffixes = possible_suffixes
        self.task_cfg = load_yaml_config(os.path.join(root_path, task_setting_folder_name, 'task_configs.yaml'))
        self.split_target_key = split_target_key  # the key to record the fold infor
        self.slide_id_key = slide_id_key

        # Find valid slide paths that have tile encodings
        valid_slide_ids, valid_sample_ids = self.get_valid_slides(data_df[slide_id_key].values,
                                                                  stopping_folder_name_list,
                                                                  mode)
        if mode == 'TCGA':
            # rename the csv index by assiging the val in slide_id_key to slide_id, previous it val is patient_id
            slide_name_to_id_dic = {}
            for index, sample_ids in enumerate(valid_sample_ids):
                slide_name_to_id_dic[sample_ids] = valid_slide_ids[index]
            # Remove rows in data_df where the sample ID (data_df[slide_id_key]) is not in slide_name_to_id_dic
            data_df = data_df[data_df[self.slide_id_key].isin(slide_name_to_id_dic.keys())]

            # Replace the values in data_df[slide_id_key] with the corresponding slide IDs from slide_name_to_id_dic
            data_df[self.slide_id_key] = data_df[self.slide_id_key].map(slide_name_to_id_dic)
        else:
            raise NotImplementedError

        # Set up the task
        self.task_name_list = self.task_cfg.get('tasks_to_run')
        assert self.task_name_list is not None

        self.setup_task_data(data_df, split_name, self.task_name_list)

        # Load from settings or set default value
        self.max_tiles = self.task_cfg.get('max_tiles', 1000)
        self.shuffle_tiles = self.task_cfg.get('shuffle_tiles', False)
        print('Dataset has been initialized with ' + str(len(self.slide_ids)) +
              ' slides for split:', split_name)

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
                # no label
                print('Label Missing for : ', slide_id)
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

        self.slide_ids = list(self.slide_paths.keys())

        return valid_slide_ids, valid_sample_ids

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

    def prepare_MTL_data_list(self, task_description_csv, split_name, task_name_list):
        """Prepare the sample for multi-label task."""
        task_dict = self.task_cfg.get('all_task_dict')
        one_hot_table = self.task_cfg.get('one_hot_table')

        # Get the label from CSV file with WSIs assigned with the target split (such as 'train').
        task_description_csv = task_description_csv[task_description_csv[self.split_target_key] == split_name]

        WSI_names = task_description_csv[self.slide_id_key]

        labels = []
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

            labels.append(task_description_list)

        return task_dict, one_hot_table, labels

    def setup_task_data(self, task_description_csv, split_name, task_name_list):
        """Prepare the sample for single task tasks_to_run or multi-task tasks_to_run.

        old demo self.prepare_single_task_data_list, the split is a list of wsi name
        """
        # todo multiple modality func
        self.task_dict, self.one_hot_table, self.labels = \
            self.prepare_MTL_data_list(task_description_csv, split_name, task_name_list)

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
        task_description_list = self.labels[idx]
        # set the sample dict
        sample = {'image_features': data_dict['image_features'],
                  'image_features_lens': data_dict['image_features_lens'],
                  'pad_mask': data_dict['pad_mask'],
                  'coords_yx': data_dict['coords_yx'],
                  'slide_id': slide_id,
                  'task_name_list':self.task_name_list,
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
        raise RuntimeError('Failed to get a valid sample after {} tries'.format(n_try))

    def __len__(self):
        return len(self.slide_ids)

    def __getitem__(self, idx):
        # fixme in this current framework the model always trained with wsi batch size of 1
        slide_level_sample = self.get_embedded_sample_with_try(idx)
        return slide_level_sample


def WSI_collate_fn(batch):  # todo havent designed for dic version
    # Filter out bad data (data loader return -1)
    cleaned_batch = [data for data in batch if data != -1]  # -1 for not valid return from dataset
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

    build_data_split_for_csv(task_description_csv, slide_id_key=slide_id_key, test_ratio=0.2, k=1,
                             mode='TCGA', key=split_target_key)
    output_dir = os.path.join(root_path, task_setting_folder_name)
    build_yaml_config_from_csv(task_description_csv, output_dir, dataset_name='lung-mix',
                               tasks_to_run=['CMS', 'COL3A1'],
                               max_tiles=1000000, shuffle_tiles=True,
                               excluding_list=(slide_id_key, split_target_key))
    # instantiate the dataset
    DatasetClass_train = SlideDataset(root_path, task_description_csv,
                                      task_setting_folder_name=task_setting_folder_name,
                                      split_name='train', slide_id_key=slide_id_key, split_target_key=split_target_key)

    print(DatasetClass_train.get_embedded_sample_with_try(20))
