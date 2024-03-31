import torch
from torch.nn.modules.loss import _Loss
import torch.nn.functional as F
import torch.nn as nn
import pdb
import logging

class Dice_Loss(nn.Module):
    def __init__(self):
        super(Dice_Loss, self).__init__()
        self.smooth = 1e-5

    def forward(self, pred, label):
        one_hot_y = torch.zeros(pred.shape).cuda()
        one_hot_y = one_hot_y.scatter_(1, label, 1.0)

        dice_score = 0.0
        
        for class_idx in range(pred.shape[1]):   # 正类  
            inter = (pred[:,class_idx,...] * one_hot_y[:,class_idx,...]).sum()
            union = (pred[:,class_idx,...] ** 2).sum() + (one_hot_y[:,class_idx,...] ** 2).sum()

            dice_score += (2*inter + self.smooth) / (union + self.smooth)

        loss = 1 - dice_score/pred.shape[1]

        return loss


def kl_divergence(alpha):
    shape = list(alpha.shape)
    shape[0] = 1
    ones = torch.ones(tuple(shape)).cuda()

    S = torch.sum(alpha, dim=1, keepdim=True) 
    first_term = (
        torch.lgamma(S)
        - torch.lgamma(alpha).sum(dim=1, keepdim=True)
        - torch.lgamma(ones.sum(dim=1, keepdim=True))
    )
    second_term = (
        (alpha - ones)
        .mul(torch.digamma(alpha) - torch.digamma(S))
        .sum(dim=1, keepdim=True)
    )
    kl = first_term + second_term
    return kl.mean()    # / batch_size


class EDL_Loss(nn.Module):
    def __init__(self, prior, kl_weight=0.01, annealing_step=10):
        super(EDL_Loss, self).__init__()
        self.prior = prior
        self.kl_weight = kl_weight
        self.annealing_step = annealing_step

    def forward(self, logit, label, epoch_num):
        K = logit.shape[1]
        alpha = F.relu(logit) + 1
        S = torch.sum(alpha, dim=1, keepdim=True) 

        one_hot_y = torch.eye(K).cuda()
        one_hot_y = one_hot_y[label]
        one_hot_y.requires_grad = False

        loss_ce = torch.sum((2-self.prior) * one_hot_y * (torch.digamma(S) - torch.digamma(alpha))) / logit.shape[0]

        annealing_coef = torch.min(
            torch.tensor(1.0, dtype=torch.float32),
            torch.tensor(epoch_num / self.annealing_step, dtype=torch.float32),
        ) 

        kl_alpha = (alpha - 1) * (1 - one_hot_y) + 1
        loss_kl = annealing_coef * kl_divergence(kl_alpha)

        loss_cor = torch.sum(- K/S.detach() * one_hot_y * logit) / logit.shape[0]      
        # print('ce loss: {}, kl loss: {}, cor loss: {}'.format(loss_ce.item(), loss_kl.item(), loss_cor.item()))

        return loss_ce + self.kl_weight * (loss_kl+loss_cor)
    

class EDL_Dice_Loss(nn.Module):
    def __init__(self, kl_weight=0.001, annealing_step=10):
        super(EDL_Dice_Loss, self).__init__()
        self.smooth = 1e-5
        self.kl_weight = kl_weight
        self.annealing_step = annealing_step

    def forward(self, logit, label, epoch_num):
        K = logit.shape[1]

        alpha = (F.relu(logit)+1)**2
        S = torch.sum(alpha, dim=1, keepdim=True) 

        pred = alpha / S

        one_hot_y = torch.zeros(pred.shape).cuda()
        one_hot_y = one_hot_y.scatter_(1, label, 1.0)
        one_hot_y.requires_grad = False

        dice_score = 0
        for class_idx in range(logit.shape[1]):   
            inter = (pred[:,class_idx,...] * one_hot_y[:,class_idx,...]).sum()
            union = (pred[:,class_idx,...] ** 2).sum() + (one_hot_y[:,class_idx,...] ** 2).sum() + (pred[:,class_idx,...]*(1-pred[:,class_idx,...])/(S[:,0,...]+1)).sum()

            dice_score += (2*inter + self.smooth) / (union + self.smooth)

        loss_dice = 1 - dice_score/logit.shape[1]

        annealing_coef = torch.min(
            torch.tensor(1.0, dtype=torch.float32),
            torch.tensor(epoch_num / self.annealing_step, dtype=torch.float32),
        ) 

        kl_alpha = (alpha - 1) * (1 - one_hot_y) + 1
        loss_kl = annealing_coef * kl_divergence(kl_alpha)

        loss_cor = torch.sum(- K/S.detach() * one_hot_y * logit) / (logit.shape[0] * logit.shape[2] * logit.shape[3])
        # print('dice loss: {}, kl loss: {}, cor loss: {}'.format(loss_dice.item(), loss_kl.item(), loss_cor.item()))

        return loss_dice + self.kl_weight * (loss_kl+loss_cor)
