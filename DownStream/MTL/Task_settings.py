"""
MTL Task settings   Script  ver： Oct 16th 13:30

flexible to multiple-tasks and missing labels
"""

import os
import json
import copy
import numpy as np
import torch.nn as nn
import yaml  # Ensure pyyaml is installed: pip install pyyaml


def build_all_tasks(task_config_path=None, latent_feature_dim=128):
    try:
        with open(task_config_path, 'r') as file:
            config = yaml.load(file, Loader=yaml.Loader)
            all_task_name_list = config.get('tasks_to_run')
            all_task_dict = config.get('all_task_dict')
            all_task_one_hot_describe = config.get('one_hot_table')

    except:
        raise  # no task-settings folder for the dataset
    else:
        # Generate task_idx_to_name, task_name_to_idx, all_class_num, all_loss_weight, all_criterions
        idx = 0
        task_name_to_idx = {}  # task name to idx list
        task_idx_to_name = []  # task idx to name list

        all_class_num = []  # class number list
        all_loss_weight = []  # todo: need to allow manually config in the future, maybe config in yaml file?
        all_criterions = []  # task loss
        all_MTL_heads = []  # MTL heads list in string, initialize the layers later when they are actually needed.

        for task in all_task_dict:
            if all_task_dict[task] == 'float':
                all_task_dict[task] = float
                all_class_num.append(0)
                all_criterions.append(nn.L1Loss(size_average=None, reduce=None))
                all_MTL_heads.append(f"nn.Linear({latent_feature_dim}, 1)")
            elif all_task_dict[task] == 'list':
                all_task_dict[task] = list
                all_class_num.append(len(all_task_one_hot_describe[task]))
                all_criterions.append(nn.CrossEntropyLoss())
                # pred (type: float): [Batch, cls], GT (type: long int): [Batch]
                # (content of GT should stack together, which means their format should be the same)
                all_MTL_heads.append(f"nn.Linear({latent_feature_dim}, {all_class_num[idx]})")
            else:
                raise ValueError('Not valid data type!')

            task_name_to_idx[task] = idx
            task_idx_to_name.append(task)
            all_loss_weight.append(1.0)
            idx += 1

    return (all_task_name_list, task_name_to_idx, task_idx_to_name, all_task_dict, all_MTL_heads, all_criterions,
            all_loss_weight, all_class_num, all_task_one_hot_describe)


def task_filter_auto(WSI_task_idx_or_name_list=None, task_config_path=None, latent_feature_dim=128):
    """Auto task filter defined by json files, currently used by Shangqing's TCGA dataset

    Args:
        task_config_path (str): task_settings_path/task_configs.yaml
        latent_feature_dim (int, optional): _description_. Defaults to 768.

    Raises:
        ValueError: _description_

    Returns:
        _type_: _description_
    """
    assert task_config_path is not None  # the task_config should not be none

    (all_task_name_list, task_name_to_idx, task_idx_to_name, all_task_dict, all_MTL_heads, all_criterions,
     all_loss_weight, all_class_num, all_task_one_hot_describe) = (
        build_all_tasks(task_config_path, latent_feature_dim))

    # building running config according to WSI_task_idx_or_name_list(support both index and text)
    if WSI_task_idx_or_name_list is None:
        WSI_task_idx_or_name_list = all_task_name_list

    if type(WSI_task_idx_or_name_list[0]) == int:
        task_idx_list = WSI_task_idx_or_name_list
    else:
        for task in WSI_task_idx_or_name_list:
            assert task in all_task_name_list, f"task {task} not in yaml config"
        task_idx_list = [task_name_to_idx[task] for task in WSI_task_idx_or_name_list]

    # build tasks
    task_dict = {}
    MTL_heads = []
    criterions = []
    loss_weight = []
    class_num = []
    task_describe = {}

    for idx in task_idx_list:
        task_dict[task_idx_to_name[idx]] = all_task_dict[task_idx_to_name[idx]]
        MTL_heads.append(eval(all_MTL_heads[idx]))  # initialize the layers when they are actually needed.
        criterions.append(all_criterions[idx])
        loss_weight.append(all_loss_weight[idx])
        class_num.append([all_class_num[idx]])
        if task_idx_to_name[idx] in all_task_one_hot_describe:  # if its cls task
            task_describe[task_idx_to_name[idx]] = all_task_one_hot_describe[task_idx_to_name[idx]]

    return task_dict, MTL_heads, criterions, loss_weight, class_num, task_describe


# base on the task dict to convert the task idx of model output
class task_idx_converter:
    def __init__(self, old_task_dict, new_task_dict):
        idx_dict = {}
        idx_project_dict = {}
        for old_idx, key in enumerate(old_task_dict):
            idx_dict[key] = old_idx
        for new_idx, key in enumerate(new_task_dict):
            idx_project_dict[new_idx] = idx_dict[key]

        self.idx_project_dict = idx_project_dict

    def __call__(self, new_idx):
        back_to_old_idx = self.idx_project_dict[new_idx]
        return back_to_old_idx


def listed_onehot_dic_to_longint_name_dic(name_onehot_list_dic):
    """
    converting name_onehot_list_dic to longint_name_dic
    Example
    name_onehot_list_dic = {'3RD': [0, 0, 0, 0, 1],
                            '4TH': [0, 0, 0, 1, 0],
                            '5TH': [0, 0, 1, 0, 0],
                            '6TH': [0, 1, 0, 0, 0],
                            '7TH': [1, 0, 0, 0, 0]}

    out
    longint_name_dic = {4: '3RD', 3: '4TH', 2: '5TH', 1: '6TH', 0: '7TH'}
    """

    longint_name_dic = {}

    for cls_name in name_onehot_list_dic:
        listed_onehot = name_onehot_list_dic[cls_name]
        long_int = np.array(listed_onehot).argmax()
        longint_name_dic[long_int] = cls_name

    return longint_name_dic


class result_recorder:
    def __init__(self, task_dict, task_describe, batch_size=1, total_size=10, runs_path=None):
        assert runs_path is not None
        self.runs_path = runs_path

        self.task_dict = task_dict
        self.batch_size = batch_size
        self.total_size = total_size

        # set up the indicators
        self.longint_to_name_dic_for_all_task = {}
        self.task_template = {}
        self.task_idx_to_key = {}
        task_idx = 0

        for key in task_dict:
            self.task_template[key] = {"pred": None, "label": None}

            self.task_idx_to_key[task_idx] = key
            task_idx += 1

            if task_dict[key] == list:
                self.longint_to_name_dic_for_all_task[key] = listed_onehot_dic_to_longint_name_dic(task_describe[key])

        # set up the sample list
        self.record_samples = {}
        self.data_iter_step = 0

    def add_step(self, data_iter_step):
        self.data_iter_step = data_iter_step
        for i in range(self.batch_size):
            record_idx = data_iter_step * self.batch_size + i
            if record_idx == self.total_size:
                break  # now at the next of last sample
            else:
                self.record_samples[record_idx] = copy.deepcopy(self.task_template)

    def add_data(self, batch_idx, task_idx, pred, label=None):
        record_idx = self.data_iter_step * self.batch_size + batch_idx

        key = self.task_idx_to_key[task_idx]
        # rewrite template
        if self.task_dict[key] == list:
            self.record_samples[record_idx][key]["pred"] = self.longint_to_name_dic_for_all_task[key][pred]
            self.record_samples[record_idx][key]["label"] = \
                self.longint_to_name_dic_for_all_task[key][label] if label is not None else None
        else:  # reg
            self.record_samples[record_idx][key]["pred"] = pred
            self.record_samples[record_idx][key]["label"] = label if label is not None else None

    def finish_and_dump(self, tag='test_'):
        # save json_log  indent=2 for better view
        log_path = os.path.join(self.runs_path, str(tag) + 'predication.json')
        json.dump(self.record_samples, open(log_path, 'w'), ensure_ascii=False, indent=2)

