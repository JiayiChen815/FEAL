import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms as T
from scipy import ndimage
import albumentations as A
from PIL import Image


import os
from glob import glob
import random
import numpy as np
import json
import pdb
import pandas as pd
import pickle

# classification datasets
class FedISIC(Dataset):
    def __init__(self, fl_method='FedAvg', client_idx=None, mode='train', transform=None):
        assert mode in ['train', 'test']

        self.num_classes = 8
        self.fl_method = fl_method
        self.client_name = ['client1', 'client2', 'client3', 'client4']
        self.client_idx = client_idx    # obtain the dataset of client_name[client_idx]
        self.mode = mode
        self.transform = transform

        df_path = 'data/data_split/FedISIC/train_test_split.csv'
        df = pd.read_csv(df_path,sep='\t')
        self.data_list = df[(df['center']==client_idx) & (df['fold']==mode)]

    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, idx:int):
        img_path = '../Dataset/FedISIC_npy/{}.npy'.format(self.data_list.iloc[idx,0])   # image
        image = np.load(img_path)
        label = self.data_list.iloc[idx,-4] # target

        if self.transform is not None:
            image = self.transform(image=image)['image']
        
        image = image.transpose(2, 0, 1).astype(np.float32)

        return idx, {'image': torch.from_numpy(image.copy()), 'label': torch.tensor(label)}


class Camelyon(Dataset):
    def __init__(self, fl_method='FedAvg', client_idx=None, mode='train', transform=None):
        assert mode in ['train', 'test']

        self.num_classes = 2
        self.fl_method = fl_method
        self.client_name = ['client1', 'client2', 'client3', 'client4', 'client5']
        self.client_idx = client_idx    
        self.mode = mode
        self.transform = transform

        data_path = 'data/data_split/FedCamelyon/{}.pkl'.format(self.client_name[client_idx])

        with open(data_path, 'rb') as f: 
            data = pickle.load(f)        
        
        self.data_list = data[mode]

    def __len__(self):
        return len(self.data_list[1])
    
    def __getitem__(self, idx:int):
        img_path = os.path.join('../Dataset/FedCamelyon', self.data_list[0][idx])
        image = np.asarray(Image.open(img_path).convert('RGB'))
        label = self.data_list[1][idx] # target

        if self.transform is not None:
            image = self.transform(image=image)['image']
        
        image = image.transpose(2, 0, 1).astype(np.float32)

        return idx, {'image': torch.from_numpy(image.copy()), 'label': torch.tensor(label)}


# segmentation datasets
class Fundus(Dataset):
    def __init__(self, fl_method, client_idx=None, mode='train', transform=None):
        assert mode in ['train', 'test']

        self.num_classes = 3
        self.fl_method = fl_method
        self.client_name = ['client1', 'client2', 'client3', 'client4']
        self.client_idx = client_idx    
        self.mode = mode
        self.transform = transform

        self.data_list = []

        with open("data/data_split/FedFundus/{}_{}.txt".format(self.client_name[self.client_idx], mode), "r") as f: 
            self.data_list = json.load(f)

    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, idx: int):
        data_path = self.data_list[idx]
        data = np.load(data_path)

        image = data[..., 0:3]  # (384, 384, 3)
        label = data[..., 3:]   # (384, 384, 1)

        sample = {'image':image, 'label':label}
        if self.transform is not None:
            sample = self.transform(sample) 

        return idx, sample



class Polyp(Dataset):
    def __init__(self, fl_method, client_idx=None, mode='train', transform=None):
        assert mode in ['train', 'test']

        self.num_classes = 2
        self.fl_method = fl_method
        self.client_name = ['client1', 'client2', 'client3', 'client4']
        self.client_idx = client_idx    
        self.mode = mode
        self.transform = transform

        self.data_list = []

        with open("data/data_split/FedPolyp/{}_{}.txt".format(self.client_name[self.client_idx], mode), "r") as f: 
            self.data_list = json.load(f)

    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, idx: int):
        data_path = self.data_list[idx]
        data = np.load(data_path)

        image = data[..., 0:3]  # (384, 384, 3)
        label = data[..., 3:]   # (384, 384, 2)

        sample = {'image':image, 'label':label}
        if self.transform is not None:
            sample = self.transform(sample) 

        return idx, sample


class Prostate(Dataset):
    def __init__(self, fl_method, client_idx=None, mode='train', transform=None):
        assert mode in ['train',  'test']

        self.num_classes = 2
        self.fl_method = fl_method
        self.client_name = ['client1', 'client2', 'client3', 'client4', 'client5', 'client6']
        self.client_idx = client_idx   
        self.mode = mode
        self.transform = transform
        self.data_list = []

        with open("data/data_split/FedProstate/{}_{}.txt".format(self.client_name[self.client_idx], mode), "r") as f: 
            self.data_list = json.load(f)

    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, idx: int):
        data_path = self.data_list[idx]
        data = np.load(data_path)

        image = data[..., 0:1]  # (384, 384, 1)
        label = data[..., 1:]   # (384, 384, 1)

        sample = {'image':image, 'label':label}
        if self.transform is not None:
            sample = self.transform(sample) 

        return idx, sample


class RandomCrop(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image = sample['image']
        label = sample['label']

        GAP_H = image.shape[0] - self.output_size[0]
        GAP_W = image.shape[1] - self.output_size[1]

        H = random.randint(int(0.25*GAP_H), int(0.75*GAP_H))
        W = random.randint(int(0.25*GAP_W), int(0.75*GAP_W))

        image = image[H:H+self.output_size[0], W:W+self.output_size[1], :]
        label = label[H:H+self.output_size[0], W:W+self.output_size[1], :]

        return {'image': image, 'label': label}
        

class CenterCrop(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image = sample['image']
        label = sample['label']

        GAP_H = (image.shape[0] - self.output_size[0])//2
        GAP_W = (image.shape[1] - self.output_size[1])//2

        image = image[GAP_H:GAP_H+self.output_size[0], GAP_W:GAP_W+self.output_size[1], :]
        label = label[GAP_H:GAP_H+self.output_size[0], GAP_W:GAP_W+self.output_size[1], :]

        return {'image': image, 'label': label}
    

class RandomCrop_Polyp(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image = sample['image']
        label = sample['label']

        GAP_H = image.shape[0] - self.output_size[0]
        GAP_W = image.shape[1] - self.output_size[1]

        while(1):
            H = random.randint(0, GAP_H)
            W = random.randint(0, GAP_W)
            if label[H:H+self.output_size[0], W:W+self.output_size[1], :].sum() > 0:
                image = image[H:H+self.output_size[0], W:W+self.output_size[1], :]
                label = label[H:H+self.output_size[0], W:W+self.output_size[1], :]
                break

        return {'image': image, 'label': label}


class ToTensor(object):
    def __call__(self, sample):
        image = sample['image'].transpose(2, 0, 1).astype(np.float32)
        label = sample['label'].transpose(2, 0, 1)

        return {'image': torch.from_numpy(image.copy()/image.max()), 'label': torch.from_numpy(label.copy()).long()}



def generate_dataset(dataset, fl_method, client_idx, args):
    if dataset == 'FedISIC':
        from data.dataset import FedISIC as Med_Dataset
        train_transform = A.Compose([
                                        A.Rotate(10),
                                        A.RandomBrightnessContrast(0.15, 0.1),
                                        A.Flip(p=0.5),
                                        A.CenterCrop(224,224),
                                        A.Normalize(always_apply=True)
                                    ])
        test_transform = A.Compose([
                                        A.CenterCrop(224,224),
                                        A.Normalize(always_apply=True)
                                    ])

    if dataset == 'FedCamelyon':
        from data.dataset import Camelyon as Med_Dataset
        train_transform = A.Compose([A.Normalize(always_apply=True)])
        test_transform = A.Compose([A.Normalize(always_apply=True)])

    elif dataset == 'FedPolyp':
        from data.dataset import Polyp as Med_Dataset
        train_transform = T.Compose([RandomCrop_Polyp((320,320)), ToTensor()])
        test_transform = ToTensor()
        
    elif dataset == 'FedProstate':
        from data.dataset import Prostate as Med_Dataset
        train_transform = T.Compose([RandomCrop((320,320)), ToTensor()])
        test_transform = ToTensor()

    elif dataset == 'FedFundus':
        from data.dataset import Fundus as Med_Dataset
        train_transform = T.Compose([RandomCrop((320,320)), ToTensor()])
        test_transform = ToTensor()


    data_train = Med_Dataset(fl_method=fl_method, 
                                client_idx=client_idx,
                                mode='train',
                                transform=train_transform)
    
    data_unlabeled = Med_Dataset(fl_method=fl_method, 
                                    client_idx=client_idx,
                                    mode='train',
                                    transform=test_transform)  

    data_test = Med_Dataset(fl_method=fl_method,
                                client_idx=client_idx,
                                mode='test',
                                transform=test_transform)
                                
    return data_train, data_unlabeled, data_test