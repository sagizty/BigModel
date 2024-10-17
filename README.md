# Fundational model pipeline for WSI + ROI
<img width="555" alt="Screenshot 2024-08-17 at 12 08 03 PM" src="https://github.com/user-attachments/assets/0114b72e-3fb8-470d-9648-43e09260ff97">

This is an opensource learning pipeline containing the multiple fractions for WSI and ROI foundational models.

The licenses for the improted code follows their original code.


## Install

On an NVIDIA A100 Tensor Core GPU machine, with CUDA toolkit enabled.

1. Download our repository and open the path
```
git clone https://github.com/sagizty/BigModel.git
cd BigModel
```

2. Install dependencies

```Shell
conda env create -f environment.yaml
conda activate BigModel
pip install -e .
```

3. Tile Cropping
```Shell
python Tiles_dataset.py \
    --WSI_dataset_path /data/hdd_1/BigModel/TCGA-LUAD-LUSC/TCGA-LUAD-raw \
    --tiled_WSI_dataset_path /data/hdd_1/BigModel/TCGA-LUAD-LUSC/tiles_datasets \
    --edge_size 224 \
    --target_mpp 0.5
```

4. Tile Embedding
```Shell
python Embedded_dataset.py \
    --WSI_dataset_path /data/hdd_1/BigModel/TCGA-LUAD-LUSC/tiles_datasets \
    --embedded_WSI_dataset_path /data/hdd_1/BigModel/TCGA-LUAD-LUSC/slide_embeddings/gigapath \
    --model_name gigapath \
    --edge_size 224 \
    --PrefetchDataLoader_num_workers 10 \
    --batch_size 256
```

5. Build MTL dataset for WSI
```Shell
python DownStream/MTL/slide_dataset_tools.py \
    --root_path /data/hdd_1/BigModel/embedded_datasets/TCGA-LUAD-LUSC-gigapath \
    --task_description_csv /home/workenv/PuzzleAI/Archive/dataset_csv/TCGA_Log_Transcriptome_Final.csv \
    --slide_id_key patient_id \
    --split_target_key fold_information \
    --task_setting_folder_name task-settings \
    --mode TCGA \
    --dataset_name luad-lusc
```

6. Run MTL task with WSI MTL framwork

```Shell
# Train
python DownStream/WSI_finetune/MTL_Train.py \
    --model_name gigapath \
    --root_path /data/ssd_1/CPIA_processed/embedded_datasets/TCGA-COAD \
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
    --root_path /data/ssd_1/CPIA_processed/embedded_datasets/TCGA-COAD \
    --save_model_path /data/private/BigModel/saved_models \
    --runs_path /data/private/BigModel/runs \
    --task_description_csv /home/workenv/PuzzleAI/Archive/dataset_csv/TCGA_Log_Transcriptome_Final.csv \
    --task_setting_folder_name task-settings \
    --slide_id_key patient_id \
    --split_target_key fold_information

# Decode the test results to csv
python Utils/Decode_correlation.py \
    --model_name gigapath \
    --root_path /data/ssd_1/CPIA_processed/embedded_datasets/TCGA-COAD \
    --runs_path /data/private/BigModel/runs \
    --WSI_tasks True \
    --task_setting_folder_name task-settings

```

7. Run ROI level tasks

```Shell
# todo need demo here
```

8. Run ROI level SSL pretraining

```Shell
# todo
```

9. Run WSI level SSL pretraining

```Shell
# todo
```

10. Run WSI level VQA-tuning after pretraining

```Shell
# todo
```

11. Run WSI level VQA application

```Shell
# todo
```

12. Run ROI level VQA application

```Shell
# todo
```

