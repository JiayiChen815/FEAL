# FEAL
This is the official Pytorch implementation of our CVPR 2024 paper "Think Twice Before Selection: Federated Evidential Active Learning for Medical Image Analysis with Domain Shifts".
![image](https://github.com/JiayiChen815/FEAL/blob/main/framework.png)

## Requirements
Please review the following requirements and install the packages listed in the `requirements.txt`
```bash
$ pip install --upgrade pip
$ pip install -r requirements.txt
```
## Data Preparation
### Datasets
- Classification
  - Fed-ISIC: Download the skin lesion classification dataset ([images](https://isic-challenge-data.s3.amazonaws.com/2019/ISIC_2019_Training_Input.zip) and [labels](https://isic-challenge-data.s3.amazonaws.com/2019/ISIC_2019_Training_GroundTruth.csv)) and [metadata](https://isic-challenge-data.s3.amazonaws.com/2019/ISIC_2019_Training_Metadata.csv) following [Falmby](https://github.com/owkin/FLamby/blob/main/flamby/datasets/fed_isic2019/dataset_creation_scripts/download_isic.py).
  - Fed-Camelyon: Download the histology breast cancer classification [dataset](https://worksheets.codalab.org/rest/bundles/0xe45e15f39fb54e9d9e919556af67aabe/contents/blob/) and [metadata](https://github.com/med-air/HarmoFL/blob/main/data/camelyon17/data.zip) following [HarmoFL](https://github.com/med-air/HarmoFL).
- Segmentation
  - Fed-Polyp: Download the endoscopic polyp segmentation [dataset](https://drive.google.com/file/d/1_sf0W4QmQn-rY7P_-OJMVZn7Hf50jD-w/view?usp=drive_link) following [PraNet](https://github.com/DengPingFan/PraNet).
  - Fed-Prostate: Download the prostate MRI segmentation [dataset](https://liuquande.github.io/SAML/) following [FedDG](https://github.com/liuquande/FedDG-ELCFS).
  - Fed-Fundus: Download the retinal fundus segmentation [dataset](https://drive.google.com/file/d/1p33nsWQaiZMAgsruDoJLyatoq5XAH-TH/view) following [FedDG](https://github.com/liuquande/FedDG-ELCFS).

### Data Preprocessing
After downloading the datasets, please execute `FEAL/data/prepare_dataset.py` for data preprocessing. The folder structure within `Dataset/` should be organized as follows.
```
├── Dataset
  ├── FedISIC_npy
    ├── ISIC_0012653_downsampled.npy, ISIC_0012654_downsampled.npy, ...
  ├── FedCamelyon
    ├── patches
      ├── patient_004_node_4, patient_009_node_1, ...

  ├── FedPolyp_npy
    ├── client1
      ├── sample1.npy, sample2.npy, ...
    ├── client2
    ├── ...
  ├── FedProstate_npy
    ├── client1
      ├── Case00
        ├── slice_012.npy, slice_013.npy, ...
      ├── ...
    ├── client2
    ├── ...
  ├── FedFundus_npy
    ├── client1
      ├── sample1.npy, sample2.npy, ...
    ├── client2
    ├── ...
```

## Usage

## Citation

## Acknowledgment