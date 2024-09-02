"""
MTL Train     Script  ver： Sep 2nd 19:30

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
sys.path.append(str(this_file_dir.parent.parent.parent))  # Go up 3 levels

import argparse
import json
import copy
import time
import math
import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
from tensorboardX import SummaryWriter


try:
    from DownStream.MTL.Dataset_Framework import *
    from DownStream.MTL.Task_settings import task_filter_auto
    from ModelBase.Get_WSI_model import build_WSI_task_model
    from Utils.MTL_plot_json import check_json_with_plot
    from Utils.tools import setup_seed
except:
    from PuzzleAI.DownStream.MTL.Dataset_Framework import *
    from PuzzleAI.DownStream.MTL.Task_settings import task_filter_auto
    from PuzzleAI.ModelBase.Get_WSI_model import build_WSI_task_model
    from PuzzleAI.Utils.MTL_plot_json import check_json_with_plot
    from PuzzleAI.Utils.tools import setup_seed


def train(model, dataloaders, dataset_sizes, criterions, optimizer, LR_scheduler, loss_weight, task_dict, task_describe,
          num_epochs=25, accum_iter_train=1, check_minibatch=5, intake_epochs=1,
          runs_path='./', writer=None, device=torch.device("cpu")):
    since = time.time()

    task_num = len(task_dict)
    running_key_CLS = []
    running_key_REG = []
    running_task_name_dict = {}
    for key in task_dict:
        if task_dict[key] == list:  # CLS sign
            running_task_name_dict[key] = task_describe[key]
            running_key_CLS.append(key)
            running_key_REG.append('not applicable')
        else:  # REG sigh
            running_key_CLS.append('not applicable')
            running_key_REG.append(key)

    # log dict
    log_dict = {}

    # loss scaler
    Scaler = torch.cuda.amp.GradScaler()
    # for recording and epoch selection
    temp_best_epoch_loss = 100000  # this one should be very big
    best_epoch_idx = 1

    epoch_average_sample_loss = 0.0  # for bag analysis

    for epoch in range(num_epochs):

        for phase in ['Train', 'Val']:

            if phase == 'Train':
                accum_iter = accum_iter_train
                model.train()  # Set model to training mode
            else:
                accum_iter = 1
                model.eval()  # Set model to evaluate mode

            # Track information
            epoch_time = time.time()
            model_time = time.time()
            index = 0  # check minibatch index

            loss = 0.0  # for back propagation, assign as float, if called it will be tensor (Train)
            accum_average_sample_loss = 0.0  # for bag analysis

            failed_sample_count = 0  # default: the whole batch is available by dataloader

            phase_running_loss = [0.0 for _ in range(task_num)]  # phase-Track on loss, separate by tasks, not weighted
            temp_running_loss = [0.0 for _ in range(task_num)]  # temp record on loss, for iteration updating
            accum_running_loss = [0.0 for _ in range(task_num)]  # accumulated record on loss, for iteration updating

            running_measurement = [0.0 for _ in range(task_num)]  # track on measurement(ACC/L1 loss etc)
            temp_running_measurement = [0.0 for _ in range(task_num)]  # temp record for iteration updating
            accum_running_measurement = [0.0 for _ in range(task_num)]  # accumulated record for iteration updating

            # missing count (track and record for validation check)
            minibatch_missing_task_sample_count = [0 for _ in range(task_num)]  # for accum iter and minibatch
            total_missing_task_sample_count = [0 for _ in range(task_num)]  # for phase

            # all matrix dict (initialize them with 0 values for each task)
            epoch_cls_task_matrix_dict = {}
            for cls_task_name in running_task_name_dict:
                class_names = [key for key in running_task_name_dict[cls_task_name]]
                # initiate the empty matrix_log_dict
                matrix_log_dict = {}
                for cls_idx in range(len(class_names)):
                    # only float type is allowed in json, set to float inside
                    matrix_log_dict[class_names[cls_idx]] = {'tp': 0.0, 'tn': 0.0, 'fp': 0.0, 'fn': 0.0}
                epoch_cls_task_matrix_dict[cls_task_name] = matrix_log_dict

            # data loop for a epoch
            for data_iter_step, sample in enumerate(dataloaders[phase]):

                # jump the batch if it cannot correct by WSI_collate_fn in dataloader
                if sample is None:
                    failed_sample_count += dataloaders[phase].batch_size
                    continue
                else:
                    # take data and task_description_list from sample
                    image_features, coords_yx, task_description_list, slide_id = sample
                    # image_features is a tensor of [B,N,D],  coords_yx is tensor of [B,N,2]
                    image_features = image_features.to(device)
                    coords_yx = coords_yx.to(device)
                    # task_description_list [task, batch_size] batch-stacked tensors, element of long-int or float

                # count failed samples in dataloader (should be 0, normally)
                # default B - B = 0, we don't have the last batch issue 'drop last batch in training code'
                failed_sample_count += dataloaders[phase].batch_size - len(task_description_list[0]) # batch size

                # Tracking the loss for loss-drive bag settings update and also recordings, initialize with 0.0
                running_loss = 0.0  # all tasks loss over a batch

                # zero the parameter gradients if its accumulated enough wrt. accum_iter
                if data_iter_step % accum_iter == 0:
                    optimizer.zero_grad()

                # fixme should have phase == 'Train', but val is too big for gpu
                with torch.cuda.amp.autocast():
                    y = model(image_features, coords_yx)

                    # we calculate the all-batch loss for each task, and do backward
                    for task_idx in range(task_num):
                        head_loss = 0.0  # head_loss for a task_idx with all samples in a same batch
                        # for back propagation, assign as float, if called it will be tensor (Train)

                        # task_description_list[task_idx].shape = [B] as long-int (CLS) or float tensor (REG)
                        # for each sample in batch
                        for batch_idx, label_value in enumerate(task_description_list[task_idx]):
                            if label_value >= 99999999:  # stop sign fixme: maybe it's a temporary stop sign
                                # Task track on missing or not
                                total_missing_task_sample_count[task_idx] += 1
                                minibatch_missing_task_sample_count[task_idx] += 1
                                continue  # record jump to skip this missing task

                            # take corresponding task criterion for task_description_list and predict output
                            # y[task_idx][batch_idx] we have [bag_size]:batch inside model pred (y)
                            output = y[task_idx][batch_idx].unsqueeze(0)  # [1,bag_size] conf. or [1] reg.
                            label = label_value.to(device).unsqueeze(0)  # [1] long-int value out of k or [1] reg.
                            head_loss += criterions[task_idx](output, label)  # calculate B[1] loss and aggregate

                            # calculate and note down the measurement
                            if running_key_CLS[task_idx] != 'not applicable':
                                # for CLS task, record the measurement in ACC
                                task_name = running_key_CLS[task_idx]
                                class_names = [key for key in running_task_name_dict[task_name]]

                                _, preds = torch.max(output.cpu().data, 1)
                                long_labels = label.cpu().data
                                # check the tp for running_measurement
                                running_measurement[task_idx] += torch.sum(preds == long_labels)

                                # Compute tp tn fp fn for each class.
                                for cls_idx in range(len(epoch_cls_task_matrix_dict[task_name])):
                                    tp = np.dot((long_labels == cls_idx).numpy().astype(int),
                                                (preds == cls_idx).cpu().numpy().astype(int))
                                    tn = np.dot((long_labels != cls_idx).numpy().astype(int),
                                                (preds != cls_idx).cpu().numpy().astype(int))
                                    fp = np.sum((preds == cls_idx).cpu().numpy()) - tp
                                    fn = np.sum((long_labels == cls_idx).numpy()) - tp
                                    # epoch_cls_task_matrix_dict[task_name][cls_idx] = {'tp': 0.0, 'tn': 0.0,...}
                                    epoch_cls_task_matrix_dict[task_name][class_names[cls_idx]]['tp'] += tp
                                    epoch_cls_task_matrix_dict[task_name][class_names[cls_idx]]['tn'] += tn
                                    epoch_cls_task_matrix_dict[task_name][class_names[cls_idx]]['fp'] += fp
                                    epoch_cls_task_matrix_dict[task_name][class_names[cls_idx]]['fn'] += fn
                            else:  # for REG tasks
                                running_measurement[task_idx] += head_loss.item()

                        # build up loss for bp
                        # phase_running_loss will always record the loss on each sample
                        if type(head_loss) == float:  # the loss is not generated: maintains as float
                            pass  # they are all missing task_description_list for this task over the whole batch
                        else:
                            # Track the task wrt. the phase
                            phase_running_loss[task_idx] += head_loss.item()
                            # Track a task's loss (over a batch) into running_loss (all tasks loss / a batch)
                            if phase == 'Train':
                                # todo in the future make a scheduler for loss_weight
                                running_loss += head_loss * loss_weight[task_idx]
                                # accum the loss for bag analysis
                                accum_average_sample_loss += running_loss.item()
                            else:  # val
                                running_loss += head_loss.item() * loss_weight[task_idx]
                                # accum the loss for bag analysis, no need gradient if not at training
                                accum_average_sample_loss += running_loss

                    # accum the bp loss from all tasks (over accum_iter of batches), if called it will be tensor (Train)
                    loss += running_loss / accum_iter  # loss for minibatch, remove the influence by loss-accumulate

                    if data_iter_step % accum_iter == accum_iter - 1:
                        index += 1  # index start with 1

                        # backward + optimize only if in training phase
                        if phase == 'Train':
                            # in all-tasks in the running of batch * accum_iter,
                            if type(loss) == float:  # the loss is not generated: maintains as float
                                pass  # minor issue, just pass (only very few chance)
                            else:
                                Scaler.scale(loss).backward()
                                Scaler.step(optimizer)
                                Scaler.update()
                            # flush loss (accum_iter) only for train, but for val it will be compared later
                            loss = 0.0  # for back propagation, re-assign as float, if called it will be tensor (Train)
                        else:
                            # for val we keep the loss to find the best epochs
                            pass

                        # triggering minibatch (over accum_iter * batch) check (updating the recordings)
                        if index % check_minibatch == 0:

                            check_time = time.time() - model_time
                            model_time = time.time()
                            check_index = index // check_minibatch

                            print('In epoch:', epoch + 1, ' ', phase, '   index of ' + str(accum_iter) + ' * '
                                  + str(check_minibatch) + ' minibatch:', check_index, '     time used:', check_time)

                            check_minibatch_results = []
                            for task_idx in range(task_num):
                                # temp loss sum values for check, accum loss is the accum loss until the previous time
                                temp_running_loss[task_idx] = phase_running_loss[task_idx] \
                                                              - accum_running_loss[task_idx]
                                # update accum
                                accum_running_loss[task_idx] += temp_running_loss[task_idx]
                                # update average running
                                valid_num = accum_iter * check_minibatch * len(task_description_list[0]) - \
                                            minibatch_missing_task_sample_count[task_idx]
                                assert valid_num != 0  # whole check runs should have at least 1 sample with 1 label
                                temp_running_loss[task_idx] /= valid_num

                                # create value
                                temp_running_measurement[task_idx] = running_measurement[task_idx] - \
                                                                     accum_running_measurement[task_idx]
                                # update accum
                                accum_running_measurement[task_idx] += temp_running_measurement[task_idx]

                                # CLS
                                if running_key_CLS[task_idx] != 'not applicable':
                                    check_minibatch_acc = temp_running_measurement[task_idx] / valid_num * 100

                                    # TP int(temp_running_measurement[task_idx])
                                    temp_running_results = (running_key_CLS[task_idx], float(check_minibatch_acc))
                                    check_minibatch_results.append(temp_running_results)
                                # REG
                                elif running_key_REG[task_idx] != 'not applicable':
                                    temp_running_results = (running_key_REG[task_idx],
                                                            float(temp_running_measurement[task_idx]) / valid_num)
                                    check_minibatch_results.append(temp_running_results)
                                else:
                                    print('record error in task_idx', task_idx)
                            print('Average_sample_loss:', temp_running_loss, '\n', check_minibatch_results, '\n')
                            # clean the missing count
                            minibatch_missing_task_sample_count = [0 for _ in range(task_num)]

            # total samples (remove dataloader-failed samples)
            total_samples = dataset_sizes[phase] - failed_sample_count
            # after an epoch, in train report loss for bag analysis
            if phase == 'Train':
                epoch_average_sample_loss = accum_average_sample_loss / total_samples

                # todo use epoch_average_sample_loss to decide bag number? currently its wasted
                if LR_scheduler is not None:  # lr scheduler: update
                    LR_scheduler.step()
            # In val, update best-epoch model index
            else:
                if type(loss) == float and loss == 0.0:
                    # in all running of val, the loss is not generated
                    print('in all running of val, the loss is not generated')
                    raise
                # compare the validation loss
                if loss <= temp_best_epoch_loss and epoch + 1 >= intake_epochs:
                    best_epoch_idx = epoch + 1
                    temp_best_epoch_loss = loss
                    best_model_wts = copy.deepcopy(model.state_dict())

            epoch_results = []
            for task_idx in range(task_num):
                epoch_valid_num = total_samples - total_missing_task_sample_count[task_idx]
                assert epoch_valid_num != 0  # whole epoch should have at least 1 sample with 1 label
                # CLS
                if running_key_CLS[task_idx] != 'not applicable':
                    epoch_acc = running_measurement[task_idx] / epoch_valid_num * 100
                    results = (running_key_CLS[task_idx],
                               epoch_cls_task_matrix_dict[running_key_CLS[task_idx]],
                               float(epoch_acc))
                    epoch_results.append(results)
                # REG
                elif running_key_REG[task_idx] != 'not applicable':
                    results = (running_key_REG[task_idx], float(running_measurement[task_idx]) / epoch_valid_num)
                    epoch_results.append(results)
                else:
                    print('record error in task_idx', task_idx)
                # loss
                phase_running_loss[task_idx] /= epoch_valid_num
            print('\nEpoch:', epoch + 1, ' ', phase, '     time used:', time.time() - epoch_time,
                  'Average_sample_loss:', phase_running_loss, '\n', epoch_results, '\n\n')

            # attach the records to the tensorboard backend
            if writer is not None:
                # ...log the running loss
                for task_idx, task_name in enumerate(task_dict):
                    writer.add_scalar(phase + '_' + task_name + ' loss',
                                      float(phase_running_loss[task_idx]),
                                      epoch + 1)
                    # use the last indicator as the measure indicator
                    writer.add_scalar(phase + '_' + task_name + ' measure',
                                      float(epoch_results[task_idx][-1]),
                                      epoch + 1)

            if phase == 'Train':
                # create the dict
                log_dict[epoch + 1] = {
                    phase: {'Average_sample_loss': phase_running_loss, 'epoch_results': epoch_results}}
            else:
                log_dict[epoch + 1][phase] = {'Average_sample_loss': phase_running_loss, 'epoch_results': epoch_results}

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('best_epoch_idx:', best_epoch_idx)
    for phase in log_dict[best_epoch_idx]:
        best_epoch_running_loss = log_dict[best_epoch_idx][phase]['Average_sample_loss']
        best_epoch_results = log_dict[best_epoch_idx][phase]['epoch_results']
        print('In:', phase, '    Average_sample_loss:', best_epoch_running_loss, '\nepoch_results:', best_epoch_results)
        # load best model weights as final model training result

    # attach the records to the tensorboard backend
    if writer is not None:
        writer.close()

    # save json_log  indent=2 for better view
    log_path = os.path.join(runs_path, time.strftime('%Y_%m_%d') + '_log.json')
    json.dump(log_dict, open(log_path, 'w'), ensure_ascii=False, indent=2)

    model.load_state_dict(best_model_wts)
    return model, log_path


def main(args):
    
    if not os.path.exists(args.save_model_path):          
        os.mkdir(args.save_model_path)
    if not os.path.exists(args.runs_path):
        os.mkdir(args.runs_path)

    run_name = 'MTL_' + args.model_name
    run_name = run_name + '_' + str(args.tag) if args.tag is not None else run_name

    save_model_path = os.path.join(args.save_model_path, run_name + '.pth')
    draw_path = os.path.join(args.runs_path, run_name)
    if not os.path.exists(draw_path):
        os.mkdir(draw_path)

    if args.enable_tensorboard:
        writer = SummaryWriter(draw_path)
        # if u run locally
        # nohup tensorboard --logdir=/4tbB/WSIT/runs --host=0.0.0.0 --port=7777 &
        # tensorboard --logdir=/4tbB/WSIT/runs --host=0.0.0.0 --port=7777
        # python3 -m tensorboard.main --logdir=/Users/zhangtianyi/Desktop/ITH/results --host=172.31.209.166 --port=7777
    else:
        writer = None

    # build task settings
    task_config_path = os.path.join(args.root_path, args.task_setting_folder_name, 'task_configs.yaml')
    WSI_task_dict, MTL_heads, WSI_criterions, loss_weight, class_num, WSI_task_describe = \
        task_filter_auto(task_config_path=task_config_path, latent_feature_dim=args.latent_feature_dim)
    print('WSI_task_dict', WSI_task_dict)

    # filtered tasks
    print("*********************************{}*************************************".format('settings'))
    for a in str(args).split(','):
        print(a)
    print("*********************************{}*************************************\n".format('setting'))

    # instantiate the dataset
    Train_dataset = SlideDataset(args.root_path, args.task_description_csv,
                                 task_setting_folder_name=args.task_setting_folder_name,
                                 split_name='train', slide_id_key=args.slide_id_key,
                                 split_target_key=args.split_target_key,
                                 max_tiles=args.max_tiles)
    Val_dataset = SlideDataset(args.root_path, args.task_description_csv,
                               task_setting_folder_name=args.task_setting_folder_name,
                               split_name='val', slide_id_key=args.slide_id_key,
                               split_target_key=args.split_target_key,
                               max_tiles=args.max_tiles)

    # print(Train_dataset.get_embedded_sample_with_try(20))
    dataloaders = {
        'Train': torch.utils.data.DataLoader(Train_dataset, batch_size=args.batch_size,
                                             collate_fn=MTL_WSI_collate_fn,
                                             shuffle=True, num_workers=args.num_workers, drop_last=True),
        'Val': torch.utils.data.DataLoader(Val_dataset, batch_size=args.batch_size,
                                           collate_fn=MTL_WSI_collate_fn,
                                           shuffle=False, num_workers=args.num_workers, drop_last=True)}
    dataset_sizes = {'Train': len(Train_dataset), 'Val': len(Val_dataset)}

    # GPU idx start with 0. -1 to use multiple GPU
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
                print("GPU distributing ERROR occur use CPU instead")
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
                print("GPU distributing ERROR occur use CPU instead")
                gpu_use = 'cpu'

    # device environment
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # build model
    model = build_WSI_task_model(model_name=args.model_name, local_weight_path=args.local_weight_path,
                                 ROI_feature_dim=args.ROI_feature_dim,
                                 MTL_heads=MTL_heads, latent_feature_dim=args.latent_feature_dim)
    model = model.to(device)
    # fixme this have bug for gigapath in train, but ok with val, possible issue with Triton
    # model = torch.compile(model)

    print('GPU:', gpu_use)
    if gpu_use == -1:
        model = nn.DataParallel(model)
    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=0.01)  # fixme play opt

    # cosine Scheduler by https://arxiv.org/pdf/1812.01187.pdf
    lf = lambda x: ((1 + math.cos(x * math.pi / args.num_epochs)) / 2) * (1 - args.lrf) + args.lrf  # cosine
    LR_scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)

    trained_model, log_path = train(model, dataloaders, dataset_sizes, WSI_criterions, optimizer, LR_scheduler,
                                    loss_weight, WSI_task_dict, WSI_task_describe, num_epochs=args.num_epochs,
                                    accum_iter_train=args.accum_iter_train, check_minibatch=args.check_minibatch,
                                    intake_epochs=args.intake_epochs, runs_path=draw_path, writer=writer, device=device)

    torch.save(trained_model.state_dict(), save_model_path)

    # print training summary
    check_json_with_plot(log_path, WSI_task_dict, save_path=draw_path)


def get_args_parser():
    parser = argparse.ArgumentParser(description='MTL Training')

    # Environment parameters
    parser.add_argument('--gpu_idx', default=-1, type=int,
                        help='use a single GPU with its index, -1 to use multiple GPU')

    # Model tag (for example k-fold)
    parser.add_argument('--tag', default=None, type=str, help='Model tag (for example 5-fold)')

    # PATH
    parser.add_argument('--root_path', default='/data/BigModel/embedded_datasets/', type=str,
                        help='MTL dataset root')
    parser.add_argument('--local_weight_path', default='/home/workenv/PuzzleAI/ModelWeight/prov-gigapath/slide_encoder.pth', type=str,
                        help='local weight path')
    parser.add_argument('--save_model_path', default='../saved_models', type=str,
                        help='save model root')
    parser.add_argument('--runs_path', default='../runs', type=str, help='save runing results path')
    # labels
    parser.add_argument('--task_description_csv',
                        default='/home/zhangty/Desktop/BigModel/prov-gigapath/PuzzleAI/Archive/dataset_csv/TCGA_Log_Transcriptome_Final.csv',
                        type=str, help='label csv file path')

    # Task settings and configurations for dataloaders
    parser.add_argument('--task_setting_folder_name', default='task-settings', type=str,
                        help='task-settings folder name')
    parser.add_argument('--slide_id_key', default='patient_id', type=str,
                        help='key for mapping the label')
    parser.add_argument('--split_target_key', default='fold_information', type=str,
                        help='key identifying the split information')
    parser.add_argument('--num_workers', default=2, type=int, help='dataloader num_workers')
    parser.add_argument('--max_tiles', default=10000, type=int, help='max tile for loading')

    # module settings
    parser.add_argument('--latent_feature_dim', default=128, type=int, help='MTL module dim')
    parser.add_argument('--embed_dim', default=768, type=int, help='feature embed_dim , default 768')
    parser.add_argument('--ROI_feature_dim', default=1536, type=int,
                        help='feature embed_dim , default 768')

    # Model settings
    parser.add_argument('--model_name', default='gigapath', type=str, help='slide level model name')

    # training settings
    parser.add_argument('--batch_size', default=1, type=int,
                        help='batch_size , default 1')
    parser.add_argument('--num_epochs', default=100, type=int,
                        help='total training epochs, default 200')
    parser.add_argument('--warmup_epochs', default=10, type=int,
                        help='warmup_epochs training epochs, default 50')
    parser.add_argument('--intake_epochs', default=50, type=int,
                        help='only save model at epochs after intake_epochs')
    parser.add_argument('--accum_iter_train', default=2, type=int,
                        help='training accum_iter for loss accuming, default 2')
    parser.add_argument('--lr', default=0.000001, type=float,
                        help='training learning rate, default 0.00001')
    parser.add_argument('--lrf', default=0.1, type=float,
                        help='Cosine learning rate decay times, default 0.1')

    # helper
    parser.add_argument('--check_minibatch', default=25, type=int,
                        help='check batch_size')
    parser.add_argument('--enable_notify', action='store_true',
                        help='enable notify to send email')
    parser.add_argument('--enable_tensorboard', action='store_true',
                        help='enable tensorboard to save status')

    return parser


if __name__ == '__main__':
    # setting up the random seed
    setup_seed(42)

    parser = get_args_parser()
    args = parser.parse_args()
    main(args)
