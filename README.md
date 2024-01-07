# Enhanced Anatomy-guided Multimodal Registration by Learning Segmentation without Ground Truth: Application to Intraprocedural CBCT/MR Liver Segmentation and Registration

rusthigh

Medical Image Analysis (MedIA), 2021

[[Paper](https://www.sciencedirect.com/science/article/pii/S1361841521000876)]

This repository contains the PyTorch implementation of the enhanced APA2Seg-Net.

### Citation
If you use this code for your research or project, please cite:

    @article{zhou2021anatomy,
      title={Anatomy-guided Multimodal Registration by Learning Segmentation without Ground Truth: Application to Intraprocedural CBCT/MR Liver Segmentation and Registration},
      author={Zhou, Bo and Augenfeld, Zachary and Chapiro, Julius and Zhou, S Kevin and Liu, Chi and Duncan, James S},
      journal={Medical Image Analysis},
      pages={102041},
      year={2021},
      publisher={Elsevier}
    }


### Environment and Dependencies
Requirements:
* Python 3.7
* Pytorch 1.4.0
* scipy
* scikit-image
* pillow
* itertools

Our code has been tested with Python 3.7, Pytorch 1.4.0, CUDA 10.0 on Ubuntu 18.04.


### Dataset Setup
 ....... 

### To Run Our Code
- Train the model
```bash
python main.py \
--name experiment_apada2seg \
--raw_A_dir ./preprocess/MRI_SEG/PROC/DCT/ \
--raw_A_seg_dir ./preproc