"""
tools for slide level dataset      Script  ver： Aug 26th 15:30

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
    for f, (train_idx, val_idx) in enumerate(group_kfold.split(trainval_samples, groups=trainval_patients)):
        print(f"Fold {f + 1}")
        fold = f + 1
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
                           key=key + '_{}fold-{}'.format(k, fold), val='train')
            write_csv_data(task_description_csv, id_key=slide_id_key, id_data=val_data,
                           key=key + '_{}fold-{}'.format(k, fold), val='val')
            write_csv_data(task_description_csv, id_key=slide_id_key, id_data=test_data,
                           key=key + '_{}fold-{}'.format(k, fold), val='test')

    print('\nTEST samples num:', len(test_data), 'TEST patients:', len(test_patient_names_noduplicate), )


def load_pickle_data_split_for_csv(task_description_csv, slide_id_key='slide_id', key='split', input_pkl_rootpath=None,
                                   mode='TCGA', k=1):
    """
    This write previous pkl split into csv
    Args:
        task_description_csv:
        slide_id_key:
        key:
        input_pkl_rootpath:
        mode:
        k:

    Returns:

    Demo
    # load 5 fold pkl to csv
    load_pickle_data_split_for_csv(task_description_csv='/data/BigModel/embedded_datasets/task-settings-5folds/TCGA_Log_Transcriptome_Final.csv',
                                   slide_id_key='patient_id', key='fold_information', input_pkl_rootpath='/data/ai4dd/TCGA_5-folds',
                                   mode='TCGA', k=5)
    """
    import pickle

    def load_data(file_path):
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        return data

    if k == 1:
        Train_list = load_data(os.path.join(input_pkl_rootpath, 'Train.pkl'))
        Val_list = load_data(os.path.join(input_pkl_rootpath, 'Val.pkl'))
        Test_list = load_data(os.path.join(input_pkl_rootpath, 'Test.pkl'))

        # in TCGA the csv slide_id_key is patient name so we need to write the patient split into it
        if mode == 'TCGA':
            Train_list = [sample[:12] for sample in Train_list]
            Val_list = [sample[:12] for sample in Val_list]
            Test_list = [sample[:12] for sample in Test_list]

        write_csv_data(task_description_csv, id_key=slide_id_key, id_data=Train_list, key=key, val='train')
        write_csv_data(task_description_csv, id_key=slide_id_key, id_data=Val_list, key=key, val='val')
        write_csv_data(task_description_csv, id_key=slide_id_key, id_data=Test_list, key=key, val='test')
    else:
        for fold in range(1,k+1):
            fold_pkl_rootpath = os.path.join(input_pkl_rootpath, 'task-settings-' + str(k) + 'folds_fold-' + str(fold))

            Train_list = load_data(os.path.join(fold_pkl_rootpath, 'Train.pkl'))
            Val_list = load_data(os.path.join(fold_pkl_rootpath, 'Val.pkl'))
            Test_list = load_data(os.path.join(fold_pkl_rootpath, 'Test.pkl'))

            # in TCGA the csv slide_id_key is patient name so we need to write the patient split into it
            if mode == 'TCGA':
                Train_list = [sample[:12] for sample in Train_list]
                Val_list = [sample[:12] for sample in Val_list]
                Test_list = [sample[:12] for sample in Test_list]

            write_csv_data(task_description_csv, id_key=slide_id_key, id_data=Train_list,
                           key=key + '_{}fold-{}'.format(k, fold), val='train')
            write_csv_data(task_description_csv, id_key=slide_id_key, id_data=Val_list,
                           key=key + '_{}fold-{}'.format(k, fold), val='val')
            write_csv_data(task_description_csv, id_key=slide_id_key, id_data=Test_list,
                           key=key + '_{}fold-{}'.format(k, fold), val='test')

    print('done')


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


def build_yaml_config_from_csv(task_description_csv, task_settings_path, dataset_name='lung-mix',
                               tasks_to_run=None, max_tiles=1000000, mode='TCGA', shuffle_tiles=True,
                               excluding_list=('WSI_name', 'split',), yaml_config_name='task_configs.yaml'):
    """
    Build a YAML configuration file from a CSV file containing task descriptions.

    Parameters:
    task_description_csv (str): Path to the task_description .csv file.
    task_settings_path (str): Output directory for the YAML file. (task-settings path)

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
        'shuffle_tiles': shuffle_tiles,
        'mode': mode
    }

    if not os.path.exists(task_settings_path):
        os.makedirs(task_settings_path)

    yaml_output_path = os.path.join(task_settings_path, yaml_config_name)
    if os.path.exists(yaml_output_path):
        os.remove(yaml_output_path)

    with open(yaml_output_path, 'w') as file:
        yaml.dump(config, file, default_flow_style=False)

    return all_task_dict, one_hot_table


'''
def load_yaml_config(yaml_path):
    """Load the YAML configuration file."""
    with open(yaml_path, 'r') as file:
        config = yaml.safe_load(file)
    return config
    The error you're encountering suggests that the YAML file contains a Python tuple, which yaml.safe_load 
    doesn't know how to handle by default. By default, yaml.safe_load does not allow the loading of arbitrary 
    Python objects, including tuples, for security reasons.

To solve this issue, you can use yaml.Loader instead of SafeLoader, which supports loading Python objects like tuples. 
'''

def load_yaml_config(file_path):
    with open(file_path, 'r') as file:
        config = yaml.load(file, Loader=yaml.Loader)
    return config


def build_split_and_task_configs(root_path, task_description_csv, dataset_name,
                                 tasks_to_run, slide_id_key, split_target_key='fold_information',
                                 task_setting_folder_name='task-settings',
                                 mode='TCGA', test_ratio=0.2, k=1, yaml_config_name='task_configs.yaml'):
    build_data_split_for_csv(task_description_csv, slide_id_key=slide_id_key, test_ratio=test_ratio, k=k,
                             mode=mode, key=split_target_key)
    output_dir = os.path.join(root_path, task_setting_folder_name)
    build_yaml_config_from_csv(task_description_csv, output_dir, dataset_name=dataset_name,
                               tasks_to_run=tasks_to_run,
                               max_tiles=1000000, shuffle_tiles=True,mode=mode,
                               excluding_list=(slide_id_key, split_target_key),
                               yaml_config_name=yaml_config_name)
    # check
    load_yaml_config(os.path.join(root_path, task_setting_folder_name, yaml_config_name))


if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser(description='Build split and task configs.')

    parser.add_argument('--root_path', type=str, default='/data/BigModel/embedded_datasets/',
                        help='Root path for the datasets')
    parser.add_argument('--task_description_csv', type=str,
                        default='/home/zhangty/Desktop/BigModel/prov-gigapath/PuzzleAI/Archive/dataset_csv/TCGA_Log_Transcriptome_Final.csv',
                        help='Path to the task description CSV')
    parser.add_argument('--slide_id_key', type=str, default='patient_id',
                        help='Slide ID key in the dataset')
    parser.add_argument('--split_target_key', type=str, default='fold_information',
                        help='Key to split the dataset')
    parser.add_argument('--task_setting_folder_name', type=str, default='task-settings',
                        help='Folder name for task settings')
    parser.add_argument('--mode', type=str, default='TCGA',
                        help='Mode (e.g., TCGA)')
    parser.add_argument('--dataset_name', type=str, default='lung-mix',
                        help='Name of the dataset')
    parser.add_argument('--tasks_to_run', type=str,
                        default='iCMS%CMS%MSI.status%EPCAM%COL3A1%CD3E%PLVAP%C1QA%IL1B%MS4A1%CD79A',
                        help='Tasks to run, separated by "%"')

    args = parser.parse_args()

    # Convert tasks_to_run to a list
    tasks_to_run = args.tasks_to_run.split('%')

    build_split_and_task_configs(
        root_path=args.root_path,
        task_description_csv=args.task_description_csv,
        dataset_name=args.dataset_name,
        tasks_to_run=tasks_to_run,
        slide_id_key=args.slide_id_key,
        split_target_key=args.split_target_key,
        task_setting_folder_name=args.task_setting_folder_name,
        mode=args.mode
    )

    '''
    python slide_dataset_tools.py --root_path /data/BigModel/embedded_datasets/ \
    --task_description_csv /home/zhangty/Desktop/BigModel/prov-gigapath/PuzzleAI/Archive/dataset_csv/TCGA_Log_Transcriptome_Final.csv \
    --slide_id_key patient_id \
    --split_target_key fold_information \
    --task_setting_folder_name task-settings \
    --mode TCGA \
    --dataset_name lung-mix \
    --tasks_to_run iCMS%CMS%MSI.status%EPCAM%COL3A1%CD3E%PLVAP%C1QA%IL1B%MS4A1%CD79A
    '''
