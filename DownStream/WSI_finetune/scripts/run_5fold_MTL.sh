#!/bin/bash
# Script  ver： Aug 28th 20:00
set -e

# conda activate gigapath
cd /home/zhangty/Desktop/BigModel/prov-gigapath/PuzzleAI

# Train
python DownStream/WSI_finetune/MTL_Train.py \
    --model_name gigapath \
    --root_path /data/BigModel/embedded_datasets/ \
    --save_model_path /data/BigModel/saved_models/ \
    --runs_path /data/BigModel/runs/ \
    --task_description_csv /data/BigModel/embedded_datasets/task-settings-5folds/20240827_TCGA_log_marker10.csv \
    --task_setting_folder_name task-settings-5folds \
    --slide_id_key patient_id \
    --split_target_key fold_information_5fold-1 \
    --num_epochs 200 \
    --warmup_epochs 10 \
    --intake_epochs 50 \
    --tag fold-1

# Test
python DownStream/WSI_finetune/MTL_Test.py \
    --model_name gigapath \
    --root_path /data/BigModel/embedded_datasets/ \
    --save_model_path /data/BigModel/saved_models/ \
    --runs_path /data/BigModel/runs/ \
    --task_description_csv /data/BigModel/embedded_datasets/task-settings-5folds/20240827_TCGA_log_marker10.csv \
    --task_setting_folder_name task-settings-5folds \
    --slide_id_key patient_id \
    --split_target_key fold_information_5fold-1 \
    --tag fold-1

# Decode the test results to csv
python Utils/Decode_correlation.py \
    --model_name gigapath \
    --root_path /data/BigModel/embedded_datasets/ \
    --runs_path /data/BigModel/runs/ \
    --WSI_tasks True \
    --task_setting_folder_name task-settings-5folds \
    --tag fold-1

# Train
python DownStream/WSI_finetune/MTL_Train.py \
    --model_name gigapath \
    --root_path /data/BigModel/embedded_datasets/ \
    --save_model_path /data/BigModel/saved_models/ \
    --runs_path /data/BigModel/runs/ \
    --task_description_csv /data/BigModel/embedded_datasets/task-settings-5folds/20240827_TCGA_log_marker10.csv \
    --task_setting_folder_name task-settings-5folds \
    --slide_id_key patient_id \
    --split_target_key fold_information_5fold-2 \
    --num_epochs 200 \
    --warmup_epochs 10 \
    --intake_epochs 50 \
    --tag fold-2

# Test
python DownStream/WSI_finetune/MTL_Test.py \
    --model_name gigapath \
    --root_path /data/BigModel/embedded_datasets/ \
    --save_model_path /data/BigModel/saved_models/ \
    --runs_path /data/BigModel/runs/ \
    --task_description_csv /data/BigModel/embedded_datasets/task-settings-5folds/20240827_TCGA_log_marker10.csv \
    --task_setting_folder_name task-settings-5folds \
    --slide_id_key patient_id \
    --split_target_key fold_information_5fold-2 \
    --tag fold-2

# Decode the test results to csv
python Utils/Decode_correlation.py \
    --model_name gigapath \
    --root_path /data/BigModel/embedded_datasets/ \
    --runs_path /data/BigModel/runs/ \
    --WSI_tasks True \
    --task_setting_folder_name task-settings-5folds \
    --tag fold-2

# Train
python DownStream/WSI_finetune/MTL_Train.py \
    --model_name gigapath \
    --root_path /data/BigModel/embedded_datasets/ \
    --save_model_path /data/BigModel/saved_models/ \
    --runs_path /data/BigModel/runs/ \
    --task_description_csv /data/BigModel/embedded_datasets/task-settings-5folds/20240827_TCGA_log_marker10.csv \
    --task_setting_folder_name task-settings-5folds \
    --slide_id_key patient_id \
    --split_target_key fold_information_5fold-3 \
    --num_epochs 200 \
    --warmup_epochs 10 \
    --intake_epochs 50 \
    --tag fold-3

# Test
python DownStream/WSI_finetune/MTL_Test.py \
    --model_name gigapath \
    --root_path /data/BigModel/embedded_datasets/ \
    --save_model_path /data/BigModel/saved_models/ \
    --runs_path /data/BigModel/runs/ \
    --task_description_csv /data/BigModel/embedded_datasets/task-settings-5folds/20240827_TCGA_log_marker10.csv \
    --task_setting_folder_name task-settings-5folds \
    --slide_id_key patient_id \
    --split_target_key fold_information_5fold-3 \
    --tag fold-3

# Decode the test results to csv
python Utils/Decode_correlation.py \
    --model_name gigapath \
    --root_path /data/BigModel/embedded_datasets/ \
    --runs_path /data/BigModel/runs/ \
    --WSI_tasks True \
    --task_setting_folder_name task-settings-5folds \
    --tag fold-3

# Train
python DownStream/WSI_finetune/MTL_Train.py \
    --model_name gigapath \
    --root_path /data/BigModel/embedded_datasets/ \
    --save_model_path /data/BigModel/saved_models/ \
    --runs_path /data/BigModel/runs/ \
    --task_description_csv /data/BigModel/embedded_datasets/task-settings-5folds/20240827_TCGA_log_marker10.csv \
    --task_setting_folder_name task-settings-5folds \
    --slide_id_key patient_id \
    --split_target_key fold_information_5fold-4 \
    --num_epochs 200 \
    --warmup_epochs 10 \
    --intake_epochs 50 \
    --tag fold-4

# Test
python DownStream/WSI_finetune/MTL_Test.py \
    --model_name gigapath \
    --root_path /data/BigModel/embedded_datasets/ \
    --save_model_path /data/BigModel/saved_models/ \
    --runs_path /data/BigModel/runs/ \
    --task_description_csv /data/BigModel/embedded_datasets/task-settings-5folds/20240827_TCGA_log_marker10.csv \
    --task_setting_folder_name task-settings-5folds \
    --slide_id_key patient_id \
    --split_target_key fold_information_5fold-4 \
    --tag fold-4

# Decode the test results to csv
python Utils/Decode_correlation.py \
    --model_name gigapath \
    --root_path /data/BigModel/embedded_datasets/ \
    --runs_path /data/BigModel/runs/ \
    --WSI_tasks True \
    --task_setting_folder_name task-settings-5folds \
    --tag fold-4

# Train
python DownStream/WSI_finetune/MTL_Train.py \
    --model_name gigapath \
    --root_path /data/BigModel/embedded_datasets/ \
    --save_model_path /data/BigModel/saved_models/ \
    --runs_path /data/BigModel/runs/ \
    --task_description_csv /data/BigModel/embedded_datasets/task-settings-5folds/20240827_TCGA_log_marker10.csv \
    --task_setting_folder_name task-settings-5folds \
    --slide_id_key patient_id \
    --split_target_key fold_information_5fold-5 \
    --num_epochs 200 \
    --warmup_epochs 10 \
    --intake_epochs 50 \
    --tag fold-5

# Test
python DownStream/WSI_finetune/MTL_Test.py \
    --model_name gigapath \
    --root_path /data/BigModel/embedded_datasets/ \
    --save_model_path /data/BigModel/saved_models/ \
    --runs_path /data/BigModel/runs/ \
    --task_description_csv /data/BigModel/embedded_datasets/task-settings-5folds/20240827_TCGA_log_marker10.csv \
    --task_setting_folder_name task-settings-5folds \
    --slide_id_key patient_id \
    --split_target_key fold_information_5fold-5 \
    --tag fold-5

# Decode the test results to csv
python Utils/Decode_correlation.py \
    --model_name gigapath \
    --root_path /data/BigModel/embedded_datasets/ \
    --runs_path /data/BigModel/runs/ \
    --WSI_tasks True \
    --task_setting_folder_name task-settings-5folds \
    --tag fold-5

set +e
