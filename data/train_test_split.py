from glob import glob
import pdb
import json
import random
import os

def split_prostate():
    if not os.path.exists('data_split/FedProstate/'):
        os.makedirs('data_split/FedProstate/')
        
    for i in range(6):
        data_list = glob('/media/userdisk1/jychen/Dataset/FedProstate_npy/client{}/*'.format(i+1))
        random.shuffle(data_list)   

        data_len = len(data_list)   # 
        test_len = int(0.2*data_len)

        test_list = data_list[:test_len]
        train_list = data_list[test_len:]

        train_slice_list = []
        for train_case in train_list:
            train_slice_list.extend(glob('{}/*'.format(train_case)))
        
        with open("data_split/FedProstate/client{}_train.txt".format(i+1), "w") as f: 
            json.dump(train_slice_list, f)

        test_slice_list = []
        for test_case in test_list:
            test_slice_list.extend(glob('{}/*'.format(test_case)))
        with open("data_split/FedProstate/client{}_test.txt".format(i+1), "w") as f: 
            json.dump(test_slice_list, f)


def split_dataset(dataset, client_num):
    if not os.path.exists('data_split/{}/'.format(dataset)):
        os.makedirs('data_split/{}/'.format(dataset))

    for i in range(1, client_num+1):   
        data_list = glob('/media/userdisk1/jychen/Dataset/{}/client{}/*'.format(dataset, i))

        data_len = len(data_list)
        test_len = int(0.2*data_len)

        test_list = data_list[:test_len]
        train_list = data_list[test_len:]

        with open("data_split/{}/client{}_train.txt".format(dataset, i), "w") as f: 
            json.dump(train_list, f)
        with open("data_split/{}/client{}_test.txt".format(dataset, i), "w") as f: 
            json.dump(test_list, f)

if __name__ == '__main__':
    split_dataset('FedPolyp_npy', 4)
    split_prostate()
    split_dataset('FedFundus_npy', 4)