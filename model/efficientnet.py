import torch
import torch.nn as nn
import timm
import torch.nn.functional as F
from torchsummary import summary
import pdb

class EfficientNetB0(nn.Module):
    def __init__(self, num_classes = 8, pretrained = True):
        super(EfficientNetB0, self).__init__()

        self.efficientnet = timm.create_model('efficientnet_b0', pretrained=False, checkpoint_path='../FAL/model/efficientnet_b0_ra-3dd342df.pth')
        num_ftrs = self.efficientnet.classifier.in_features
        self.efficientnet.reset_classifier(0)
        self.fc = nn.Linear(num_ftrs, num_classes)

        self.block_features = []
        self.handle1 = None
        self.handle2 = None
        self.handle3 = None
        self.handle4 = None
        self.register_hook(self.efficientnet)

    def register_hook(self, model):
        self.handle1 = model.blocks[1].register_forward_hook(hook=self.hook_block_forward)
        self.handle2 = model.blocks[2].register_forward_hook(hook=self.hook_block_forward)
        self.handle3 = model.blocks[3].register_forward_hook(hook=self.hook_block_forward)
        self.handle4 = model.blocks[4].register_forward_hook(hook=self.hook_block_forward) 

    def hook_block_forward(self, module, input, output):
        self.block_features.append(output)

    def forward(self, x, embedding=False):
        self.block_features = []

        if embedding:
            outf = x
        else:
            outf = self.efficientnet(x)
            
        out = self.fc(outf)

        self.handle1 = None
        self.handle2 = None
        self.handle3 = None
        self.handle4 = None

        return out, F.softmax(out, dim=-1), outf, self.block_features


