# ESFPNet
Official Implementation of "ESFPNet: efficient deep learning architecture for real-time lesion segmentation in autofluorescence bronchoscopic video"

**:fire: NEWS :fire:**
**The full paper is available:** [The complete paper of ESFPNet](https://arxiv.org/pdf/2207.07759v3.pdf)

**The polyp datasets' results is available:** [Polyp Datasets' ESFPNet Models and Image Results](https://drive.google.com/drive/folders/1I4vsts-dfyUgrnbKi-Z8XQYVhVICYpOs?usp=share_link).

**:fire: CHEERS! :fire:** 
**This paper is selected as a [finalist of the Robert F. Wagner All-Conference Best Student Paper Award at SPIE Medical Imaging 2023](https://drive.google.com/file/d/1974ALKd6X0EUzm4nuR3kBWr2Gj0ctpdz/view?usp=share_link)**


## Global Rank

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/esfpnet-efficient-deep-learning-architecture/medical-image-segmentation-on-cvc-clinicdb)](https://paperswithcode.com/sota/medical-image-segmentation-on-cvc-clinicdb?p=esfpnet-efficient-deep-learning-architecture)

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/esfpnet-efficient-deep-learning-architecture/medical-image-segmentation-on-etis)](https://paperswithcode.com/sota/medical-image-segmentation-on-etis?p=esfpnet-efficient-deep-learning-architecture)

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/esfpnet-efficient-deep-learning-architecture/medical-image-segmentation-on-kvasir-seg)](https://paperswithcode.com/sota/medical-image-segmentation-on-kvasir-seg?p=esfpnet-efficient-deep-learning-architecture)

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/esfpnet-efficient-deep-learning-architecture/medical-image-segmentation-on-cvc-colondb)](https://paperswithcode.com/sota/medical-image-segmentation-on-cvc-colondb?p=esfpnet-efficient-deep-learning-architecture)

## Architecture of ESFPNet

<div align=center><img src="https://github.com/dumyCq/ESFPNet/blob/main/Figures/Network.jpg" width="1000" height="500" alt="Result"/></div>

## Installation & Usage
### Enviroment (Python 3.8)
- Install Pytorch (version 1.11.0, torchvision == 0.12.0):
```
conda install pytorch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 cudatoolkit=11.3 -c pytorch
```
- Install image reading and writting library (version 2.21.2):
```
conda install -c conda-forge imageio
```
- Install image processing library:
```
pip install scikit-image
```
- Install unofficial pytorch image model:
```
pip install timm
```
- Install OpenMMLab computer vision foundation:
```
pip install mmcv
```
- Install library for parsing and emitting YAML:
```
pip install pyyaml
```
- Install other packages:
```
conda install pillow numpy matplotlib
```
- Install Jupyter-Notebook to run .ipynb file
```
conda install -c anaconda jupyter
```
### Dataset

- Extract the folders and copy them under "Endoscope-WL" folder
- The datasets are ordered as follows in "Endoscope-WL" folder:
|-- $DATASET_NAME$_Splited
|   |-- testSplited
|   |   |-- images
|   |   |-- masks
|   |-- trainSplited
|   |   |-- images
|   |   |-- masks
|   |-- validationSplited
|   |   |-- images
|   |   |-- masks
```
- The default dataset paths can be changed in "Configure.yaml"
- To randomly split the CVC-ClincDB or Kvasir dataset, set "if_renew = True" in "ESFPNet_Endoscope_Learning_Ability.ipynb"
- To repeat generate the splitting dataset, previous generated folder shold be detelted first
- To reuse the splitting dataset without generating a new dataset, set "if_renew = False"
### Pretrained Model
- Download the pretrained Mixtransformer from this link: [Pretrained Model](https://drive.google.com/drive/folders/1FLtIfDHDaowqyF_HhmORFMlRzCpB94hV?usp=sharing)
- Put the pretrained models under "Pretrained" folder
## Evaluation

### Citation
If you think this paper helps, please cite:
```
@inproceedings{chang2023esfpnet,
  title={ESFPNet: efficient deep learning architecture for real-time lesion segmentation in autofluorescence bronchoscopic video},
  author={Chang, Qi and Ahmad, Danish and Toth, Jennifer and Bascom, Rebecca and Higgins, William E},
  booktitle={Medical Imaging 2023: Biomedical Applications in Molecular, Structural, and Functional Imaging},
  volume={12468},
  pages={1246803},
  year={2023},
  organization={SPIE}
}
```
Since the training of MixTransformer based network requires a good GPU.
One helpful state-of-the-art work compared in this paper without using MixTransformer backbone is [CARANet](https://github.com/AngeLouCN/CaraNet)
If you also think this work helps, please cite:
```
@inproceedings{lou2021caranet,
author = {Ange Lou and Shuyue Guan and Hanseok Ko and Murray H. Loew},
title = {{CaraNet: context axial reverse attention network for segmentation of small medical objects}},
volume = {12032},
booktitle = {Medical Imaging 2022: Image Processing},
organization = {International Society for Optics and Photonics},
publisher = {SPIE},
pages = {81 -- 92},
year = {2022},
doi = {10.1117/12.2611802}}
```
