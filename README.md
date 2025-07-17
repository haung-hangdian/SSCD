# Beyond Entropy Filtering: Spatial-Semantic Curriculum Decoupling for Semi-Supervised Medical Image Segmentation 🧠💻

[![GitHub Stars](https://img.shields.io/github/stars/haung-hangdian/SSCD?style=social)](https://github.com/haung-hangdian/SSCD)
[![Paper PDF](https://img.shields.io/badge/Paper-PDF-orange)](https://arxiv.org/abs/XXXX.XXXXX)
[![Conference](https://img.shields.io/badge/Conference-AAAI_2026-blue)]([https://conferences.miccai.org/2024/en/](https://aaai.org/conference/aaai/aaai-26/))
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/Python-3.8%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Pytorch](https://img.shields.io/badge/PyTorch-2.0.0-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)

You are my ![Visitor Count](https://visitor-badge.laobi.icu/badge?page_id=hauang-hangdian), Thank You! &#x1F618;&#x1F618;

> **Abstract**: The Spatial-Semantic Curriculum Decoupling (SSCD) framework addresses the critical challenge of foreground-background class imbalance in semi-supervised medical image segmentation through task-driven curriculum learning. By adaptively decoupling spatial and semantic complexity during training, our method achieves state-of-the-art performance on multiple benchmarks.

## 🔬 Framework Overview

<p align="center">
  <img src="https://github.com/haung-hangdian/SSCD/blob/main/figure/Overview.png" width="90%">
  <p align="center">Fig 1. Spatial-Semantic Curriculum Decoupling (SSCD) Framework</p>
</p>

## 🛠️ Installation
```path
https://github.com/haung-hangdian/SSCD.git
```

This repository is based on PyTorch 2.0.0, CUDA 11.8. All experiments in our paper were conducted on NVIDIA GeForce RTX 4070S GPU with an identical experimental setting.

## 📁 Dataset 
```path
📁 data/
├── 📁 ACDC/
│   └── 📁 [data]/
│       ├── 📄 test.list
│       ├── 📄 train_slices.list
│       ├── 📄 train.list
│       └── 📄 val.list
└── 📁 promise12/
    ├── 📄 CaseXX_segmentation.mhd
    ├── 📄 CaseXX_segmentation.raw
    ├── 📄 CaseXX.mhd
    ├── 📄 CaseXX.raw
    ├── 📄 test.list
    └── 📄 val.list
```
### ACDC
Data could be got at [ACDC](https://github.com/HiLab-git/SSL4MIS/tree/master/data/ACDC)

### PROMISE12

Data could be got at [PROMISE12](https://promise12.grand-challenge.org/Download/) 
## 🚀 Usage
### 🏋️‍♂️ Training
```path
python ./code/train_ACDC_SSCD.py  # for ACDC training 
```
### 📊 Evaluation
```path
python ./code/test_ACDC.py  # for ACDC testing 
```
## 📈 Results

### Quantitative Comparison
<p align="center">
  <img src="https://github.com/haung-hangdian/SSCD/blob/main/figure/results.png" width="600" height="400">
</p>

<p align="center">Fig 2. Comparisons with other methods on the PROMISE12 test set.</p>

### Visualization
<p align="center">
  <img src="https://github.com/haung-hangdian/SSCD/blob/main/figure/view.png" width="300" height="500">
</p>
<p align="center">Fig 3. Our method to visualize the segmentation of ACDC dataset.</p>

## 📚 Citation

## 🤝 Contributing
We thank the MICCAI ACDC and MICCAI PROMISE12 challenges for providing the benchmark dataset.
## 📬 Contact
If you have any questions, welcome contact me at 'hengyi.huang@hdu.edu.cn'.
