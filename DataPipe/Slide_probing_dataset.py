"""
Embedding slide_feature dataset     Script  ver： Sep 3rd 21:30

flexible to multiple-tasks and missing labels

we have enable multiple samples training by controlling the gradient in different task labels
we break the process of controlling when calculating the gradient, and
we use loss-aggregate technique to combine each sample for back-propagation
"""
import os
import sys
from pathlib import Path

# For convinience
this_file_dir = Path(__file__).resolve().parent
sys.path.append(str(this_file_dir.parent.parent))  # Go up 3 levels

import argparse
import time
import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
from tqdm import tqdm

try:
    from DataPipe.h5tools import hdf5_save_a_slide_embedding_dataset
    from DownStream.MTL.Dataset_Framework import *
    from ModelBase.Get_WSI_model import build_WSI_prob_embedding_model
    from Utils.tools import setup_seed
except:
    from PuzzleAI.DataPipe.h5tools import hdf5_save_a_slide_embedding_dataset
    from PuzzleAI.DownStream.MTL.Dataset_Framework import *
    from PuzzleAI.ModelBase.Get_WSI_model import build_WSI_prob_embedding_model
    from PuzzleAI.Utils.tools import setup_seed


def building_prob_dataset(model, dataloader, dataset_size, h5_file_path, device=torch.device("cpu")):
    model.eval()  # Set model to evaluate mode
    epoch_time = time.time()
    failed_sample_count = 0
    # data loop
    for data_iter_step, sample in tqdm(enumerate(dataloader), desc="Embedding Slide level features for",
                                       unit="batch (of {} slides)".format(dataloader.batch_size),
                                       total=dataset_size // dataloader.batch_size):
        '''
        sample = {'image_features': data_dict['image_features'],
                  'image_features_lens': data_dict['image_features_lens'],
                  'pad_mask': data_dict['pad_mask'],
                  'coords_yx': data_dict['coords_yx'],
                  'slide_id': slide_id,
                  'task_name_list': self.task_name_list,
                  'task_description_list': task_description_list}
        '''
        # jump the batch if it cannot correct by WSI_collate_fn in dataloader
        if sample is None:
            failed_sample_count += dataloader.batch_size
            continue
        else:
            # take data and task_description_list from sample
            image_features, coords_yx, task_description_list, slide_ids = sample
            # image_features is a tensor of [B,N,D],  coords_yx is tensor of [B,N,2]
            image_features = image_features.to(device)
            coords_yx = coords_yx.to(device)
            # task_description_list [task, batch_size] batch-stacked tensors, element of long-int or float

        # count failed samples in dataloader (should be 0, normally)
        # default B - B = 0, we dont have the last batch issue 'drop last batch in training code'
        failed_sample_count += dataloader.batch_size - len(slide_ids)

        with torch.cuda.amp.autocast():
            with torch.no_grad():
                slide_features = model(image_features, coords_yx)
                for slide_idx in range(len(slide_ids)):
                    slide_feature = slide_features[slide_idx].cpu().numpy()
                    slide_id = slide_ids[slide_idx]
                    # fixme save the output features in what format ?
                    print('slide_id', slide_id)
                    print('slide_feature', slide_feature)
                    hdf5_save_a_slide_embedding_dataset(h5_file_path, slide_id, slide_feature,
                                                        section_type='slide_features')

    # total samples (remove dataloader-failed samples)
    valid_samples = dataset_size - failed_sample_count

    time_elapsed = time.time() - epoch_time
    print('Embedding complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))


def main(args):
    h5_file_path = os.path.join(args.root_path, args.task_setting_folder_name, 'slide_embedding.h5')
    # instantiate the dataset
    dataset = SlideDataset(args.root_path, args.task_description_csv,
                           task_setting_folder_name=args.task_setting_folder_name,
                           split_name=None, slide_id_key=args.slide_id_key, split_target_key=None,
                           dataset_type='embedding', max_tiles=args.max_tiles)

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size,
                                             collate_fn=MTL_WSI_collate_fn,
                                             shuffle=False, num_workers=args.num_workers)
    # info
    dataset_size = len(dataset)

    # GPU idx start with0, -1 to use multipel GPU
    if args.gpu_idx == -1:  # use all cards
        if torch.cuda.device_count() > 1:
            print("Use", torch.cuda.device_count(), "GPUs!")
            # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
            gpu_use = args.gpu_idx
        else:
            print('we dont have more GPU idx here, try to use gpu_idx=0')
            try:
                os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # setting k for: only card idx k is sighted for this code
                gpu_use = 0
            except:
                print("GPU distributing ERRO occur use CPU instead")
                gpu_use = 'cpu'

    else:
        # Decide which device we want to run on
        try:
            # setting k for: only card idx k is sighted for this code
            os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_idx)
            gpu_use = args.gpu_idx
        except:
            print('we dont have that GPU idx here, try to use gpu_idx=0')
            try:
                # setting 0 for: only card idx 0 is sighted for this code
                os.environ['CUDA_VISIBLE_DEVICES'] = '0'
                gpu_use = 0
            except:
                print("GPU distributing ERRO occur use CPU instead")
                gpu_use = 'cpu'
    # device enviorment
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # build model
    model = build_WSI_prob_embedding_model(model_name=args.model_name,
                                           local_weight_path=args.local_weight_path,
                                           ROI_feature_dim=args.ROI_feature_dim)

    model = torch.compile(model)
    print('GPU:', gpu_use)
    if gpu_use == -1:
        model = nn.DataParallel(model)
    model.to(device)

    building_prob_dataset(model, dataloader, dataset_size, h5_file_path, device=device)


def get_args_parser():
    parser = argparse.ArgumentParser(description='building slide_level prob_dataset')

    # Environment parameters
    parser.add_argument('--gpu_idx', default=-1, type=int,
                        help='use a single GPU with its index, -1 to use multiple GPU')

    # PATH
    parser.add_argument('--root_path', default='/data/BigModel/embedded_datasets/', type=str,
                        help='MTL dataset root')
    parser.add_argument('--local_weight_path',
                        default=None, type=str,
                        help='local weight path')
    # labels
    parser.add_argument('--task_description_csv',
                        default='/home/zhangty/Desktop/BigModel/prov-gigapath/PuzzleAI/Archive/dataset_csv/TCGA_Log_Transcriptome_Final.csv',
                        type=str, help='label csv file path')

    # Task settings and configurations for dataloaders
    parser.add_argument('--task_setting_folder_name', default='task-settings-5folds', type=str,
                        help='task-settings folder name')
    parser.add_argument('--slide_id_key', default='patient_id', type=str,
                        help='key for mapping the label')
    parser.add_argument('--num_workers', default=2, type=int, help='dataloader num_workers')
    parser.add_argument('--max_tiles', default=10000, type=int, help='max tile for loading')

    # module settings
    parser.add_argument('--ROI_feature_dim', default=1536, type=int,
                        help='feature embed_dim , default 768')

    # Model settings
    parser.add_argument('--model_name', default='gigapath', type=str, help='slide_feature level model name')

    # training settings
    parser.add_argument('--batch_size', default=1, type=int,
                        help='batch_size , default 1')

    # helper
    parser.add_argument('--check_minibatch', default=25, type=int,
                        help='check batch_size')

    return parser


if __name__ == '__main__':
    # setting up the random seed
    setup_seed(42)

    parser = get_args_parser()
    args = parser.parse_args()
    main(args)
