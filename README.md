# FEAL
This is the official Pytorch implementation of our CVPR 2024 paper "Think Twice Before Selection: Federated Evidential Active Learning for Medical Image Analysis with Domain Shifts".
![image](https://github.com/JiayiChen815/FEAL/blob/master/framework.png)

## Requirements
Please review the following requirements and install the packages listed in the `requirements.txt`
```bash
$ pip install --upgrade pip
$ pip install -r requirements.txt
```
## Data Preparation
### Datasets
- Classification
  - Fed-ISIC: Download the skin lesion classification dataset ([images](https://isic-challenge-data.s3.amazonaws.com/2019/ISIC_2019_Training_Input.zip) and [labels](https://isic-challenge-data.s3.amazonaws.com/2019/ISIC_2019_Training_GroundTruth.csv)) and [metadata](https://isic-challenge-data.s3.amazonaws.com/2019/ISIC_2019_Training_Metadata.csv) following [Flamby](https://github.com/owkin/FLamby/blob/main/flamby/datasets/fed_isic2019/dataset_creation_scripts/download_isic.py).
  - Fed-Camelyon: Download the histology breast cancer classification [dataset](https://worksheets.codalab.org/rest/bundles/0xe45e15f39fb54e9d9e919556af67aabe/contents/blob/) and [metadata](https://github.com/med-air/HarmoFL/blob/main/data/camelyon17/data.zip) following [HarmoFL](https://github.com/med-air/HarmoFL).
- Segmentation
  - Fed-Polyp: Download the endoscopic polyp segmentation [dataset](https://drive.google.com/file/d/1_sf0W4QmQn-rY7P_-OJMVZn7Hf50jD-w/view?usp=drive_link) following [PraNet](https://github.com/DengPingFan/PraNet).
  - Fed-Prostate: Download the prostate MRI segmentation [dataset](https://liuquande.github.io/SAML/) following [FedDG](https://github.com/liuquande/FedDG-ELCFS).
  - Fed-Fundus: Download the retinal fundus segmentation [dataset](https://drive.google.com/file/d/1p33nsWQaiZMAgsruDoJLyatoq5XAH-TH/view) following [FedDG](https://github.com/liuquande/FedDG-ELCFS).

### Data Preprocessing
After downloading the datasets, please naviagte to the `FEAL/data/` directory and execute `prepare_dataset.py` for data preprocessing. The folder structure within `Dataset/` should be organized as follows.
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

### Data Split
The data split of Fed-ISIC and Fed-Camelyon follows [Flamby](https://github.com/owkin/FLamby/blob/main/flamby/datasets/fed_isic2019/dataset_creation_scripts/train_test_split) and [HarmoFL](https://github.com/med-air/HarmoFL/blob/main/data/camelyon17/data.zip), respectively. For Fed-Polyp, Fed-Prostate, and Fed-Fundus, please navigate to the `FEAL/data` directory and execute `train_test_split.py` for the data split process.

## Usage
For skin lesion classification using the Fed-ISIC dataset, the command for execution is as follows:
```
CUDA_VISIBLE_DEVICES=1 python main_cls_al.py --dataset FedISIC --al_method FEAL --query_model both --query_ratio 0 --budget 500 --al_round 5 --max_round 100 --batch_size 32 --base_lr 5e-4 --kl_weight 1e-2 --display_freq 20 
```

## Citation
If you find this work helpful for your research, please consider citing:
```
@inproceedings{chen2024think,
  title={Think Twice Before Selection: Federated Evidential Active Learning for Medical Image Analysis with Domain Shifts},
  author={Chen, Jiayi and Ma, Benteng and Cui, Hengfei and Xia, Yong},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={11439--11449},
  year={2024}
}
```

## Acknowledgment
The codebase is adapted from [FedDG](https://github.com/liuquande/FedDG-ELCFS), [FedLC](https://github.com/jcwang123/FedLC), and [EDL](https://github.com/dougbrion/pytorch-classification-uncertainty). We sincerely appreciate their insightful work and contributions.