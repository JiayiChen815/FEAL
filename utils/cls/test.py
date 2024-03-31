import logging
import torch
import numpy as np
from sklearn.metrics import balanced_accuracy_score, accuracy_score

def test(dataset, model, dataloader, client_idx):
    model.eval()

    pred_list = torch.tensor([]).cuda()
    label_list = torch.tensor([]).cuda()

    with torch.no_grad():
        for _, (_, data) in enumerate(dataloader):
            image, label = data['image'], data['label']

            image = image.cuda()
            label = label.cuda()

            logit = model(image)[0]    

            pred_list = torch.cat((pred_list, torch.argmax(logit, dim=1)))
            label_list = torch.cat((label_list, label))

    if dataset == 'FedISIC':
        return balanced_accuracy_score(label_list.cpu().numpy(), pred_list.cpu().numpy())
