# ESFPNet
Official Implementation of "ESFPNet: efficient deep learning architecture for real-time lesion segmentation in autofluorescence bronchoscopic video"



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
```
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
```
- The default dataset paths can be changed in "Configure.yaml"
- To randomly split the CVC-ClincDB or Kvasir dataset, set "if_renew = True" in "ESFPNet_Endoscope_Learning_Ability.ipynb"
- To repeat generate the splitting dataset, previous generated folder shold be detelted first
- To reuse the splitting dataset without generating a new dataset, set "if_renew = False"
```
### Pretrained Model
- Download the pretrained Mixtransformer from this link: [Pretrained Model](https://drive.google.com/drive/folders/1FLtIfDHDaowqyF_HhmORFMlRzCpB94hV?usp=sharing)
- Put the pretrained models under "Pretrained" folder
