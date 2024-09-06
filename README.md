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

3. Application and framework
