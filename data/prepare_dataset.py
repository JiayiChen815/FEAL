import os
import SimpleITK as sitk
import numpy as np
from glob import glob
import matplotlib.pyplot as plt
from PIL import Image
import pdb
import copy
import pickle


def pkload(fname):
    with open(fname, 'rb') as f:
        return pickle.load(f)

def save_npy(data, p):
    dir = os.path.dirname(p)
    os.makedirs(dir, exist_ok=True)
    np.save(p, data)

def mr_norm(x, r=0.99):
    _x = x.flatten().tolist()
    _x.sort()
    vmax = _x[int(len(_x) * r)]
    vmin = _x[0]
    x = np.clip(x, vmin, vmax)
    x = (x - vmin) / vmax
    return x


def prepare_isic():
    img_path = glob('../../Dataset/FedISIC/ISIC_2019_Training_Input_preprocessed/*')
    for i in range(len(img_path)):
        pdb.set_trace()
        img = Image.open(img_path[i])
        img_np = np.asarray(img)
        np.save('../../Dataset/FedISIC_npy/{}'.format(img_path[i].split('/')[-1].replace('jpg','npy')), img_np)


def prepare_camelyon():
    data_dict = np.load('../../Dataset/Camelyon/data.pkl', allow_pickle=True)

    for i in range(1,6):
        key = 'hospital{}'.format(i)
        new_dict = {}

        new_dict['train']=[]
        new_dict['train'].append(data_dict[key]['train'][0])
        new_dict['train'].append(data_dict[key]['train'][1])

        new_dict['test']=[]
        new_dict['test'].append(data_dict[key]['test'][0])
        new_dict['test'].append(data_dict[key]['test'][1])

        # decompose into 5 files
        if not os.path.exists('data_split/FedCamelyon/'):
            os.makedirs('data_split/FedCamelyon/')
        with open('data_split/FedCamelyon/client{}.pkl'.format(i), 'wb') as f:  
            pickle.dump(new_dict, f)


def prepare_polyp():
    client_name = ['Kvasir', 'ETIS', 'CVC-ColonDB', 'CVC-ClinicDB']  # 1000, 196, 612, 380

    client_data_list = []
    client_mask_list = []
    for i, site_name in enumerate(client_name):
        client_data_list.append(sorted(glob('../../Dataset/FedPolyp/{}/image/*'.format(site_name))))
        client_mask_list.append(sorted(glob('../../Dataset/FedPolyp/{}/mask/*'.format(site_name))))

    for client_idx in range(len(client_name)):
        if not os.path.exists('../../Dataset/FedPolyp_npy/client{}'.format(client_idx+1)):
            os.makedirs('../../Dataset/FedPolyp_npy/client{}'.format(client_idx+1))

        for data_idx in range(len(client_data_list[client_idx])):
            data_path = client_data_list[client_idx][data_idx]
            mask_path = client_mask_list[client_idx][data_idx]

            img = Image.open(data_path)
            # resize
            if img.size[0] >= img.size[1]:  
                W = 384
                H = int((img.size[1]/img.size[0]) * W)
            else:
                H = 384
                W = int((img.size[0]/img.size[1]) * H)

            img = img.resize((W,H), Image.BICUBIC)
            img_np = np.asarray(img)

            # pad to 384x384
            PAD_H1 = (384-H) // 2   
            if (384-H) % 2 == 0:
                PAD_H2 = PAD_H1
            else:
                PAD_H2 = PAD_H1 + 1

            PAD_W1 = (384-W) // 2
            if (384-W) % 2 == 0:
                PAD_W2 = PAD_W1
            else:
                PAD_W2 = PAD_W1 + 1

            img_np = np.pad(img_np, ((PAD_H1, PAD_H2), (PAD_W1,PAD_W2), (0,0)), constant_values=0)

            mask = Image.open(mask_path)
            mask = mask.resize((W,H), Image.NEAREST)
            mask_np = copy.deepcopy(np.asarray(mask))
            
            if len(mask_np.shape)==2:
                mask_np = np.expand_dims(mask_np, axis=2)
            elif len(mask_np.shape)==3:
                mask_np = np.expand_dims(mask_np[...,0], axis=2)
                
            mask_np = np.pad(mask_np, ((PAD_H1, PAD_H2), (PAD_W1,PAD_W2), (0,0)), constant_values=0)

            mask_np[mask_np<128] = 0
            mask_np[mask_np>128] = 1

            sample = np.dstack((img_np, mask_np))
            np.save('../../Dataset/FedPolyp_npy/client{}/sample{}.npy'.format(client_idx+1, data_idx+1), sample)


def prepare_prostate():
    client_name = ['BIDMC', 'BMC', 'HK', 'I2CVB', 'RUNMC', 'UCL']

    for i in range(len(client_name)):
        if not os.path.exists('../../Dataset/FedProstate_npy/{}'.format(client_name[i])):
            os.makedirs('../../Dataset/FedProstate_npy/{}'.format(client_name[i]))

        seg_paths = glob('../../Dataset/FedProstate/{}/*mentation*'.format(client_name[i]))
        seg_paths.sort()
        print('[INFO]', client_name[i], len(seg_paths))

        img_paths = [p[:-20] + '.nii.gz' for p in seg_paths]
        for j in range(len(seg_paths)): # patient
            itk_image = sitk.ReadImage(img_paths[j])
            itk_mask = sitk.ReadImage(seg_paths[j])
            image = sitk.GetArrayFromImage(itk_image)
            mask = sitk.GetArrayFromImage(itk_mask)

            case_name = img_paths[j].split('/')[-1][:6]

            cnt = np.zeros(2, )
            for k in range(image.shape[0]):
                slice_image = mr_norm(image[k])
                slice_mask = (mask[k] > 0).astype(int)

                if slice_mask.max() > 0:    
                    cnt[1] += 1 # positive slice
                else:
                    continue 
                cnt[0] += 1

                sample = np.dstack((np.expand_dims(slice_image,2), np.expand_dims(slice_mask,2)))
                np.save('../../Dataset/FedProstate_npy/client{}/{}/slice{:03d}.npy'.format(client_name[i], case_name, k+1), sample)
            
            print('patient {}, {} positive slices'.format(j, cnt[1]))


def prepare_fundus():
    client_name = ['client1', 'client2', 'client3', 'client4']

    client_data_list = []
    client_mask_list = []
    for client_idx in range(len(client_name)):
        client_data_list.append(sorted(glob('../../Dataset/FedFundus/{}/image/*'.format(client_name[client_idx]))))
        client_mask_list.append(sorted(glob('../../Dataset/FedFundus/{}/mask/*'.format(client_name[client_idx]))))

    for client_idx in range(len(client_name)):
        if not os.path.exists('../../Dataset/FedFundus_npy/{}'.format(client_name[client_idx])):
            os.makedirs('../../Dataset/FedFundus_npy/{}'.format(client_name[client_idx]))

        for data_idx in range(len(client_data_list[client_idx])):
            data_path = client_data_list[client_idx][data_idx]
            mask_path = client_mask_list[client_idx][data_idx]

            img = Image.open(data_path)
            img = img.resize((384,384), Image.BICUBIC)
            img_np = np.asarray(img)

            mask = Image.open(mask_path)
            mask = mask.resize((384,384), Image.NEAREST)
            mask_np = np.asarray(mask)
            mask_np = np.expand_dims(copy.deepcopy(mask_np[...,0]), axis=2)
            mask_np[mask_np==0] = 2     # optic cup
            mask_np[mask_np==128] = 1   # mask_np >= 1 -> optic disc
            mask_np[mask_np==255] = 0   # background

            sample = np.dstack((img_np, mask_np))            
            np.save('../../Dataset/FedFundus_npy/{}/sample{}.npy'.format(client_name[client_idx], data_idx+1), sample)


if __name__ == '__main__':
    prepare_isic()
    prepare_camelyon()

    prepare_polyp()
    prepare_prostate()
    prepare_fundus()