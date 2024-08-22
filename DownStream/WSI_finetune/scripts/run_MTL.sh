#!/bin/bash
# Script  ver： Aug 22nd 21:00
set -e

conda activate gigapath
cd /home/zhangty/Desktop/BigModel/prov-gigapath/PuzzleAI

# Tile the dataset
python DataPipe/Tiles_dataset.py \
    --WSI_dataset_path /data/hdd_1/ai4dd/metadata/TCGA-READ/raw_data_sample \
    --tiled_WSI_dataset_path /data/hdd_1/BigModel/sampled_tiles_datasets \
    --edge_size 224 \
    --target_mpp 0.5

# Embed the dataset
python DataPipe/Tiles_dataset.py \
    --WSI_dataset_path /data/hdd_1/BigModel/sampled_tiles_datasets \
    --embedded_WSI_dataset_path /data/hdd_1/BigModel/sampled_embedded_datasets \
    --model_name gigapath \
    --edge_size 224 \
    --batch_size 256 \
    --target_mpp 0.5

# Make the dataset for MTL
python DownStream/MTL/slide_dataset_tools.py \
    --root_path /data/BigModel/embedded_datasets/ \
    --task_description_csv /home/zhangty/Desktop/BigModel/prov-gigapath/PuzzleAI/Archive/dataset_csv/TCGA_Log_Transcriptome_Final.csv \
    --slide_id_key patient_id \
    --split_target_key fold_information \
    --task_setting_folder_name task-settings \
    --mode TCGA \
    --dataset_name lung-mix \
    --tasks_to_run iCMS%CMS%MSI.status%EPCAM%COL3A1%CD3E%PLVAP%C1QA%IL1B%MS4A1%CD79A

# Train
python DownStream/WSI_finetune/MTL_Train.py \
    --model_name gigapath \
    --root_path /data/BigModel/embedded_datasets/ \
    --save_model_path /data/BigModel/saved_models/ \
    --runs_path /data/BigModel/runs/ \
    --task_description_csv /home/zhangty/Desktop/BigModel/prov-gigapath/PuzzleAI/Archive/dataset_csv/TCGA_Log_Transcriptome_Final.csv \
    --task_setting_folder_name task-settings \
    --slide_id_key patient_id \
    --split_target_key fold_information \
    --num_epochs 100 \
    --warmup_epochs 10 \
    --intake_epochs 50

# Test
python DownStream/WSI_finetune/MTL_Test.py \
    --model_name gigapath \
    --root_path /data/BigModel/embedded_datasets/ \
    --save_model_path /data/BigModel/saved_models/ \
    --runs_path /data/BigModel/runs/ \
    --task_description_csv /home/zhangty/Desktop/BigModel/prov-gigapath/PuzzleAI/Archive/dataset_csv/TCGA_Log_Transcriptome_Final.csv \
    --task_setting_folder_name task-settings \
    --slide_id_key patient_id \
    --split_target_key fold_information

# Decode the test results to csv
python Utils/Decode_correlation.py \
    --model_name gigapath \
    --root_path /data/BigModel/embedded_datasets/ \
    --runs_path /data/BigModel/runs/ \
    --WSI_tasks True \
    --task_setting_folder_name task-settings

set +e
