import torch

def cnt_sample_num(labeled_loader, num_classes):
    num = torch.zeros(num_classes).cuda()
    for _, (_, data) in enumerate(labeled_loader):
        label = data['label']
        num += torch.tensor([(label==i).sum() for i in range(num_classes)]).cuda()

    return num

    
