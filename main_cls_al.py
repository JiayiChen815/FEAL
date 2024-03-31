import numpy as np
import argparse
import os
import time
import random
import logging
import sys
import glob
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

from data.dataset import generate_dataset

from utils.fed_merge import FedAvg, FedUpdate

from utils.cls.train_fedavg import train
from utils.cls.test import test

from utils.cls.selection_methods import query_samples
from utils.utils import cnt_sample_num
import pdb

import pickle


parser = argparse.ArgumentParser()
parser.add_argument('--fl_method', type=str,  default='FedAvg', help='federated method')
parser.add_argument('--al_method', type=str,  default='Random', help='sampling method')
parser.add_argument('--dataset', type=str,  default='FedISIC', help='dataset')

parser.add_argument('--max_round', type=int,  default=100, help='maximum round number of FL')
parser.add_argument('--al_round', type=int,  default=5, help='maximum round number of AL')

parser.add_argument('--query_model', type=str,  default='global', help='query model')
parser.add_argument('--query_ratio', type=float,  default=0, help='query ratio')
parser.add_argument('--budget', type=int,  default=500, help='query budget')

parser.add_argument('--batch_size', type=int, default=32, help='batch size')
parser.add_argument('--base_lr', type=float,  default=5e-4, help='learning rate')
parser.add_argument('--deterministic', type=bool,  default=False, help='whether use deterministic training')

parser.add_argument('--seed', type=int,  default=0, help='random seed')
parser.add_argument('--display_freq', type=int, default=25, help='display fequency')

parser.add_argument('--kl_weight', type=float, default=0.01, help='edl kl weight')
parser.add_argument('--annealing_step', type=int, default=10, help='annealing_step')
parser.add_argument('--n_neighbor', type=int, default=5, help='number of neighbors')
parser.add_argument('--cosine', type=float, default=0.85, help='cosine')

args = parser.parse_args()


def worker_init_fn(worker_id):
    random.seed(args.seed+worker_id)


if __name__ == '__main__':
    # log
    localtime = time.localtime(time.time())
    ticks = '{:>02d}{:>02d}{:>02d}{:>02d}{:>02d}'.format(localtime.tm_mon, localtime.tm_mday, localtime.tm_hour, localtime.tm_min,localtime.tm_sec)

    snapshot_path = "logs/{}/{}/{}_{}_{}_{}/".format(args.dataset.lower(), args.query_model, args.dataset, args.fl_method, args.al_method, ticks)
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)
    if not os.path.exists(snapshot_path + '/model'):
        os.makedirs(snapshot_path + '/model')

    logging.basicConfig(filename=snapshot_path+"/log.txt", level=logging.INFO, format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))

    # init
    dataset = args.dataset
    assert dataset in ['FedISIC']
    fl_method = args.fl_method
    assert fl_method in ['FedAvg']

    if dataset == 'FedISIC':
        num_classes = 8
        client_num = 4
        SUBSET = 10000  
        from model.efficientnet import EfficientNetB0 as Model

    train_slice_num = np.zeros(client_num, dtype=int)
    batch_size = args.batch_size
    base_lr = args.base_lr
    max_round = args.max_round
    display_freq = args.display_freq
    
    
    # al
    al_method = args.al_method
    assert al_method in ['Random', 'FEAL']
    al_round = args.al_round
    query_model = args.query_model
    assert query_model in ['global', 'local', 'both']
    query_ratio = args.query_ratio
    query_num = np.zeros(client_num, dtype=int)
    if query_ratio == 0:
        budget = args.budget


    # random seed
    if args.deterministic:
        cudnn.benchmark = False
        cudnn.deterministic = True
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)


    # local dataloader, model, optimizer
    local_models = []

    local_train_data = []
    local_unlabeled_data = []

    local_labeled_sets = []
    local_unlabeled_sets = []

    local_train_loaders = []
    local_test_loaders = []


    for client_idx in range(client_num):
        # data
        data_train, data_unlabeled, data_test = generate_dataset(dataset=dataset, fl_method=fl_method, client_idx=client_idx, args=args)
        
        local_train_data.append(data_train)
        local_unlabeled_data.append(data_unlabeled)

        # init
        train_slice_num[client_idx] = len(data_train)
        if query_ratio == 0:
            if budget <= np.ceil(0.85 * train_slice_num[client_idx]):
                query_num[client_idx] = budget
            else:
                query_num[client_idx] = np.ceil(0.85 * train_slice_num[client_idx])
        else:
            query_num[client_idx] = np.floor(len(data_train) * query_ratio)

        # initial set
        indices = list(range(train_slice_num[client_idx]))
        random.shuffle(indices)
        labeled_set = indices[:query_num[client_idx]]
        unlabeled_set = indices[query_num[client_idx]:]
        local_labeled_sets.append(labeled_set)
        local_unlabeled_sets.append(unlabeled_set)

        # dataloader
        train_loader = DataLoader(dataset=data_train, batch_size=batch_size, sampler = SubsetRandomSampler(labeled_set), num_workers=4, pin_memory=True)
        test_loader = DataLoader(dataset=data_test, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
                
        local_train_loaders.append(train_loader)
        local_test_loaders.append(test_loader)

        # model
        model = Model(num_classes=num_classes).cuda()
        local_models.append(model)


    writer = SummaryWriter(snapshot_path+'/log')


    # active learning
    print('total slice: {}'.format(train_slice_num))
    for al_round_idx in tqdm(range(al_round), ncols=100):
        logging.info('\nAL round {}'.format(al_round_idx+1))

        # global model
        global_model = Model(num_classes=num_classes).cuda()

        local_optimizers = []
        local_schedulers = []
        for client_idx in range(client_num):
            # optimizer
            if dataset == 'FedISIC':
                optimizer = torch.optim.Adam(local_models[client_idx].parameters(), lr=args.base_lr, weight_decay=5e-4)
                scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[50], gamma=0.1)
                
            local_optimizers.append(optimizer)
            local_schedulers.append(scheduler)

        train_num = [len(item) for item in local_labeled_sets]
        with open(os.path.join(snapshot_path, 'global_test_result.txt'), 'a') as f:
            print('train num: {}'.format(train_num), file=f)

        num_per_class = [cnt_sample_num(local_train_loaders[client_idx], num_classes) for client_idx in range(client_num)]
        FedUpdate(global_model, local_models)   # init

        # federated learning in an AL round
        for round_idx in range(max_round):
            for client_idx in range(client_num):
                train(round_idx=round_idx,
                        client_idx=client_idx, 
                        model=local_models[client_idx], 
                        dataloader=local_train_loaders[client_idx], 
                        optimizer=local_optimizers[client_idx], 
                        num_per_class=num_per_class[client_idx],
                        args=args) 
                local_schedulers[client_idx].step()
            

            client_weight = train_num 
            client_weight = client_weight / np.sum(client_weight)
            logging.info(client_weight)
            FedAvg(global_model, local_models, client_weight)   # update the global model


            if (round_idx+1) % 5 == 0:
                with open(os.path.join(snapshot_path, 'global_test_result.txt'), 'a') as f:
                    print('AL round {}, FL round {}'.format(al_round_idx+1, round_idx+1), file=f)

                for client_idx in range(client_num):
                    metric = test(dataset=dataset, model=global_model, dataloader=local_test_loaders[client_idx], client_idx=client_idx)
                    with open(os.path.join(snapshot_path, 'global_test_result.txt'), 'a') as f:
                        if dataset == 'FedISIC':
                            print('client {}. Balanced acc:\t{}'.format(client_idx, metric), file=f)
                print('\n')

            if round_idx == max_round-1 and al_round_idx < al_round-1:    
                # query samples
                for client_idx in range(client_num):
                    # save local models
                    save_model_path = os.path.join(snapshot_path + '/model/AL{}_FL{}_client{}.pth'.format(al_round_idx+1, round_idx+1, client_idx))
                    torch.save(local_models[client_idx].state_dict(), save_model_path)

                    if len(local_labeled_sets[client_idx]) >= np.ceil(0.85 * train_slice_num[client_idx]):
                        continue
                    if len(local_labeled_sets[client_idx]) + query_num[client_idx] > np.ceil(0.85 * train_slice_num[client_idx]):
                        query_num[client_idx] = (np.ceil(0.85 * train_slice_num[client_idx]) - len(local_labeled_sets[client_idx])).astype('int')

                    if len(local_unlabeled_sets[client_idx]) <= query_num[client_idx]:
                        subset = local_unlabeled_sets[client_idx][:SUBSET]
                        rank_arg = list(range(len(local_unlabeled_sets[client_idx])))
                    else:
                        random.shuffle(local_unlabeled_sets[client_idx])
                        subset = local_unlabeled_sets[client_idx][:SUBSET]

                        rank_arg = query_samples(al_method=al_method, 
                                                    global_model=global_model, 
                                                    local_model=local_models[client_idx], 
                                                    tod_model=None, 
                                                    data_unlabeled=local_unlabeled_data[client_idx], 
                                                    unlabeled_set=subset, 
                                                    labeled_set=local_labeled_sets[client_idx], 
                                                    query_num=query_num[client_idx],
                                                    num_per_class=num_per_class[client_idx],
                                                    client_idx=client_idx,
                                                    round_idx=al_round_idx,
                                                    args=args)

                    query_set = list(torch.tensor(subset)[rank_arg][-query_num[client_idx]:].numpy())
                    local_labeled_sets[client_idx] += query_set
                    listd = list(torch.tensor(subset)[rank_arg][:-query_num[client_idx]].numpy())
                    local_unlabeled_sets[client_idx] = listd + local_unlabeled_sets[client_idx][SUBSET:]

                    # update local_train_loaders
                    local_train_loaders[client_idx] = DataLoader(dataset=local_train_data[client_idx],
                                                                    batch_size=batch_size,
                                                                    sampler = SubsetRandomSampler(local_labeled_sets[client_idx]),  
                                                                    num_workers=4,
                                                                    pin_memory=True)

            FedUpdate(global_model, local_models)   # distribute
        
        save_model_path = os.path.join(snapshot_path + '/model/AL{}_FL{}_global.pth'.format(al_round_idx+1, round_idx+1))
        torch.save(global_model.state_dict(), save_model_path)

    writer.close()


    

