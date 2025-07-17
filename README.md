# Beyond Entropy Filtering: Spatial-Semantic Curriculum decouple for Semi-Supervised Medical Image Segmentation

This repo implements the segmentation method in the paper Beyond Entropy Filtering: Spatial-Semantic Curriculum decouple for Semi-Supervised Medical Image Segmentation.

You are my ![](https://visitor-badge.laobi.icu/badge?page_id=hauang-hangdian), Thank You! &#x1F618;&#x1F618;

![](https://github.com/haung-hangdian/SSCD/blob/main/figure/Overview.png)
<p align="center">Fig 1. Detailed framework structure of the SSCD.</p>

The Spatial-Semantic Curriculum Decoupling (SSCD) framework to address foreground-background class imbalance through task-driven curriculum learning.
## Installation
```path
https://github.com/haung-hangdian/SSCD.git
```

This repository is based on PyTorch 2.0.0, CUDA 11.8. All experiments in our paper were conducted on NVIDIA GeForce RTX 4070S GPU with an identical experimental setting.

## Dataset
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
## Usage
To train a model,
```path
python ./code/train_ACDC_SSCD.py  # for ACDC training 
```
To test a model,
```path
python ./code/test_ACDC.py  # for ACDC testing 
```
## Results
<p align="center">
  <img src="https://github.com/haung-hangdian/SSCD/blob/main/figure/results.png" width="600" height="400">
</p>

<p align="center">Fig 2. Comparisons with other methods on the PROMISE12 test set.</p>

<p align="center">
  <img src="https://github.com/haung-hangdian/SSCD/blob/main/figure/view.png" width="300" height="500">
</p>
<p align="center">Fig 3. Our method to visualize the segmentation of ACDC dataset.</p>

## Quastions
If you have any questions, welcome contact me at 'hengyi.huang@hdu.edu.cn'.
