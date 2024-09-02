#!/bin/bash
# Script  ver： Aug 22nd 22:00

# To run this script:
# nohup bash run_MTL.sh > ./logs/run_MTL.log 2>&1 &

# To kill the related processes:
# ps -ef | grep run_MTL.sh | awk '{print $2}' |xargs kill
# ps -ef | grep Tiles_dataset.py | awk '{print $2}' |xargs kill
# ps -ef | grep Embedded_dataset.py | awk '{print $2}' |xargs kill
# ps -ef | grep gigapath | awk '{print $2}' |xargs kill

set -e

export CUDA_VISIBLE_DEVICES=1,2

source /root/miniforge3/bin/activate
conda activate gigapath
cd /home/workenv/PuzzleAI

# # Tile the dataseta
# python DataPipe/Tiles_dataset.py \
#     --WSI_dataset_path /data/hdd_1/CPIA/TCGA-READ \
#     --tiled_WSI_dataset_path /data/ssd_1/CPIA_processed/tiles_datasets/TCGA-READ \
#     --edge_size 224 \
#     --target_mpp 0.5

# # Embed the dataset
# python DataPipe/Embedded_dataset.py \
#     --WSI_dataset_path /data/ssd_1/CPIA_processed/tiles_datasets/TCGA-READ \
#     --embedded_WSI_dataset_path /data/ssd_1/CPIA_processed/embedded_datasets/TCGA-READ \
#     --model_name gigapath \
#     --edge_size 224 \
#     --batch_size 2048

# # Make the dataset for MTL
# python DownStream/MTL/slide_dataset_tools.py \
#     --root_path /data/ssd_1/CPIA_processed/embedded_datasets/TCGA-READ \
#     --task_description_csv /home/workenv/PuzzleAI/Archive/dataset_csv/TCGA_Log_Transcriptome_Final.csv \
#     --slide_id_key patient_id \
#     --split_target_key fold_information \
#     --task_setting_folder_name task-settings \
#     --mode TCGA \
#     --dataset_name lung-mix \
#     --tasks_to_run iCMS%CMS%MSI.status%EPCAM%COL3A1%CD3E%PLVAP%C1QA%IL1B%MS4A1%CD79A

# Train
python DownStream/WSI_finetune/MTL_Train.py \
    --model_name gigapath \
    --root_path /data/ssd_1/CPIA_processed/embedded_datasets/TCGA-READ \
    --local_weight_path /home/workenv/PuzzleAI/ModelWeight/prov-gigapath/slide_encoder.pth \
    --save_model_path /data/private/BigModel/saved_models \
    --runs_path /data/private/BigModel/runs \
    --task_description_csv /home/workenv/PuzzleAI/Archive/dataset_csv/TCGA_Log_Transcriptome_Final.csv \
    --task_setting_folder_name task-settings \
    --slide_id_key patient_id \
    --split_target_key fold_information \
    --num_epochs 100 \
    --warmup_epochs 10 \
    --intake_epochs 50

# Test
python DownStream/WSI_finetune/MTL_Test.py \
    --model_name gigapath \
    --root_path /data/ssd_1/CPIA_processed/embedded_datasets/TCGA-READ \
    --save_model_path /data/private/BigModel/saved_models \
    --runs_path /data/private/BigModel/runs \
    --task_description_csv /home/workenv/PuzzleAI/Archive/dataset_csv/TCGA_Log_Transcriptome_Final.csv \
    --task_setting_folder_name task-settings \
    --slide_id_key patient_id \
    --split_target_key fold_information

# Decode the test results to csv
python Utils/Decode_correlation.py \
    --model_name gigapath \
    --root_path /data/ssd_1/CPIA_processed/embedded_datasets/TCGA-READ \
    --runs_path /data/private/BigModel/runs \
    --WSI_tasks True \
    --task_setting_folder_name task-settings

set +e
