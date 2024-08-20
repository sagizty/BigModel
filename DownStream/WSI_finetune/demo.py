'''
MTL training panel       Script  ver： Aug 21th 01:00
'''
import os
import sys
from pathlib import Path

# For convinience
this_file_dir = Path(__file__).resolve().parent
sys.path.append(str(this_file_dir.parent.parent.parent))  # Go up 3 levels

import torch
import pandas as pd
import numpy as np

from training_tools import train
from params import get_finetune_params
from task_configs.utils import load_task_config
from utils import seed_torch, get_exp_code, get_splits, get_loader

try:
    from DataPipe.slide_dataset_tools import *
    from ModelBase.Get_WSI_model import build_WSI_task_model
    from ModelBase.Task_settings import *
except:
    from PuzzleAI.DataPipe.slide_dataset_tools import *
    from PuzzleAI.ModelBase.Get_WSI_model import build_WSI_task_model
    from PuzzleAI.ModelBase.Task_settings import *

if __name__ == '__main__':
    root_path = '/data/BigModel/embedded_datasets/'
    task_description_csv = \
        '/home/zhangty/Desktop/BigModel/prov-gigapath/PuzzleAI/Archive/dataset_csv/TCGA_Log_Transcriptome_Final.csv'
    slide_id_key = 'patient_id'
    split_target_key = 'fold_information'
    task_setting_folder_name = 'task-settings'
    latent_feature_dim = 128

    build_data_split_for_csv(task_description_csv, slide_id_key=slide_id_key, test_ratio=0.2, k=1,
                             mode='TCGA', key=split_target_key)
    task_settings_path = os.path.join(root_path, task_setting_folder_name)
    build_yaml_config_from_csv(task_description_csv, task_settings_path, dataset_name='lung-mix',
                               tasks_to_run=['CMS', 'COL3A1'],
                               max_tiles=1000000, shuffle_tiles=True,
                               excluding_list=(slide_id_key, split_target_key))
    # instantiate the dataset
    DatasetClass_train = SlideDataset(root_path, task_description_csv,
                                      task_setting_folder_name=task_setting_folder_name,
                                      split_name='train', slide_id_key=slide_id_key, split_target_key=split_target_key)

    print('get sample')
    sample = DatasetClass_train.get_embedded_sample_with_try(20)
    # print(sample)
    '''
    sample = {'image_features': image features [N, D] tensor,
              'image_features_lens': data_dict['image_features_lens'],
              'pad_mask': data_dict['pad_mask'],
              'coords_yx': [N, 2] tensor,
              'slide_id': slide_id,
              'task_name_list': self.task_name_list,
              'task_description_list': task_description_list}
    '''

    task_config_path = os.path.join(task_settings_path, 'task_configs.yaml')
    WSI_task_dict, MTL_heads, WSI_criterions, loss_weight, class_num, WSI_task_describe = \
        task_filter_auto(task_config_path, latent_feature_dim=latent_feature_dim)
    print('WSI_task_dict', WSI_task_dict)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = build_WSI_task_model(model_name='gigapath', local_weight_path=None, ROI_feature_dim=1536,
                                 MTL_heads=MTL_heads,latent_feature_dim=latent_feature_dim)
    model = model.to(device)

    # example with a sample
    # load the batch and transform this batch
    images, img_coords, task_description_list = \
        sample['image_features'], sample['coords_yx'], sample['task_description_list']
    # stack to a pseudo batch
    images = images.unsqueeze(0)
    img_coords = img_coords.unsqueeze(0)

    images = images.to(device, non_blocking=True)
    img_coords = img_coords.to(device, non_blocking=True)
    # label = label.to(device, non_blocking=True).long()

    with torch.cuda.amp.autocast(dtype=torch.float16):
        slide_embeds = model(images, img_coords)
    # layer_outputs = {"layer_{}_embed".format(i): slide_embeds[i].cpu() for i in range(len(slide_embeds))}
    # layer_outputs["last_layer_embed"] = slide_embeds[-1].cpu()
    print(slide_embeds)
