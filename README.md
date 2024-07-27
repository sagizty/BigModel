# Fundational model pipeline via learning Prov-GigaPath

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

3. Cropping
```Shell
cd BigModel/DataPipe
# set path inside and run
nohup python tiles_dataset.py &
```

4. Embedding the ROIs
```Shell
cd BigModel/DataPipe
# set path inside and run
nohup python embedded_dataset.py &
```

5. Running the models
   
todo

## A whole-slide foundation model for digital pathology from real-world data

[[`Model`](https://huggingface.co/prov-gigapath/prov-gigapath)] [[`Paper`](https://aka.ms/gigapath)] [[`BibTeX`](#Citation)]

Hanwen Xu*, Naoto Usuyama*, Jaspreet Bagga, Sheng Zhang, Rajesh Rao, Tristan Naumann, Cliff Wong, Zelalem Gero, Javier González, Yu Gu, Yanbo Xu, Mu Wei, Wenhui Wang, Shuming Ma, Furu Wei, Jianwei Yang, Chunyuan Li, Jianfeng Gao, Jaylen Rosemon, Tucker Bower, Soohee Lee, Roshanthi Weerasinghe, Bill J. Wright, Ari Robicsek, Brian Piening, Carlo Bifulco, Sheng Wang, Hoifung Poon (*Equal Contribution)

[![License](https://img.shields.io/badge/Code%20License-Prov%20GigaPath-red)]()


## Model Download

The Prov-GigaPath models can be accessed from [HuggingFace Hub](https://huggingface.co/prov-gigapath/prov-gigapath).

You need to agree to the terms to access the models. Once you have the necessary access, set your HuggingFace read-only token as an environment variable:
```
export HF_TOKEN=<huggingface read-only token>
```

If you don’t set the token, you might encounter the following error:
```
ValueError: We have no connection or you passed local_files_only, so force_download is not an accepted option.
```

## Inference

The Prov-GigaPath model consists of a tile encoder, that extracts local patterns at patch level, and a slide encoder, that outputs representations at slide level. This model can be used in both tile-level and slide-level tasks. When doing inference at the slide level, we recommend following this pipeline: (1) Tile the whole slide into N image tiles, with the coordinates of each tile. (2) Get the embeddings for each tile using our tile encoder. (3) Pass the N image tile embeddings and their coordinates into the slide encoder, to get slide level representations.

### Inference with the tile encoder

First, load GigaPath tile encoder:

```Python
import timm
from PIL import Image
from torchvision import transforms
import torch

# Older versions of timm have compatibility issues. Please ensure that you use a newer version by running the following command: pip install timm>=1.0.3.
tile_encoder = timm.create_model("hf_hub:prov-gigapath/prov-gigapath", pretrained=True)

transform = transforms.Compose(
    [
        transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ]
)
```

Running inference to extract tile level features:

```Python
img_path = "PuzzleAI/Archive/images/prov_normal_000_1.png"
sample_input = transform(Image.open(img_path).convert("RGB")).unsqueeze(0)

tile_encoder.eval()
with torch.no_grad():
    output = tile_encoder(sample_input).squeeze()
```

**

### Inference with the slide encoder

To inference with our slide encoder, we need both the tile embeddings and their coordinates as input. First, let's load the GigaPath slide encoder:

```Python
import gigapath

slide_encoder = gigapath.slide_encoder.create_model("hf_hub:prov-gigapath/prov-gigapath", "gigapath_slide_enc12l768d", 1536)
```

Run the inference to get the slide level embeddings:

```Python
slide_encoder.eval()
with torch.no_grad():
    output = slide_encoder(tile_embed, coordinates).squeeze()
```


**Note** Older versions of timm have compatibility issues. Please ensure that you use a newer version by running the following command: pip install timm>=1.0.3.


## Fine-tuning

### Tile-Level Linear Probing Example Using PCam Dataset

For your convenience, we provide the pre-extracted embeddings for the PCam dataset. You can download them from the link below. Note that the file size is 2GB.
```sh
wget -nc https://hanoverprod.z21.web.core.windows.net/gigapath/GigaPath_PCam_embeddings.zip -P data/
```

There is no need to unzip this file.

To run the fine-tuning experiment, execute the following script:
```sh
bash scripts/run_pcam.sh data/GigaPath_PCam_embeddings.zip
```

### Slide-Level Fine-Tuning Example Using PANDA Dataset

For your convenience, we provide the pre-extracted embeddings for the PANDA dataset. You can download them from the link below. Note that the file size is 32GB. Please unzip this file.
```sh
wget -nc https://hanoverprod.z21.web.core.windows.net/gigapath/GigaPath_PANDA_embeddings.zip -P data/
unzip -n data/GigaPath_PANDA_embeddings.zip -d data/
```

To run the fine-tuning experiment, execute the following script:
```sh
bash scripts/run_panda.sh data/GigaPath_PANDA_embeddings/h5_files
```

## Sample Data Download

A sample de-identified subset of the Prov-Path data can be accessed from these links [[1](https://zenodo.org/records/10909616), [2](https://zenodo.org/records/10909922)].

