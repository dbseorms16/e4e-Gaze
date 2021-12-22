import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import math
import numpy as np
#from Torch.ntools import VGG16

class model(nn.Module):
    def __init__(self):
        super(model, self).__init__()
        # 36 * 60
        # 18 * 30
        # 9  * 15
        
        vgg16ForLeft = torchvision.models.vgg16(pretrained=True)
        vgg16ForRight = torchvision.models.vgg16(pretrained=True)

        self.leftEyeNet = vgg16ForLeft.features
        self.rightEyeNet = vgg16ForRight.features

        self.leftPool = nn.AdaptiveAvgPool2d(1)
        self.rightPool = nn.AdaptiveAvgPool2d(1)

        self.leftFC = nn.Sequential(
            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024, momentum=0.99, eps=1e-3),
            nn.ReLU(inplace=True)
        )
        self.rightFC = nn.Sequential(
            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024, momentum=0.99, eps=1e-3),
            nn.ReLU(inplace=True)
        )

        self.totalFC1 = nn.Sequential(
            nn.Linear(2048, 512),
            nn.BatchNorm1d(512, momentum=0.99, eps=1e-3),
            nn.ReLU(inplace=True)
        )

        self.totalFC2 = nn.Sequential(
            nn.Linear(514, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 2)
        )

    def forward(self, le_c_list, re_c_list, face, head_batch_label):
        leftFeature = self.leftEyeNet(le_c_list)
        rightFeature = self.rightEyeNet(re_c_list)

        leftFeature = self.leftPool(leftFeature)
        rightFeature = self.rightPool(rightFeature)

        leftFeature = leftFeature.view(leftFeature.size(0), -1)
        rightFeature = rightFeature.view(rightFeature.size(0), -1)

        leftFeature = self.leftFC(leftFeature)
        rightFeature = self.rightFC(rightFeature)

        feature = torch.cat((leftFeature, rightFeature), 1)

        feature = self.totalFC1(feature)
        feature = torch.cat((feature,  head_batch_label.unsqueeze(0)), 1)

        gaze = self.totalFC2(feature)

        return gaze

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, mode="fan_in", nonlinearity="relu")
                nn.init.zeros_(m.bias)

def make_model():
    return model()

class GazeModel(nn.Module):
    def __init__(self, opt):
        super(GazeModel, self).__init__()
        print('\nMaking Gaze model...')
        self.opt = opt
        self.n_GPUs = opt.n_GPUs
        self.device = torch.device('cpu' if opt.cpu else 'cuda')
        self.model = make_model().to(self.device)
        self.load(opt.pre_gaze, cpu=opt.cpu)

    def load(self, pre_gaze, cpu=False):
        
        #### load gaze model ####

        if pre_gaze != '.':
            weihgt = dict((key.replace('model.', ''), value) for (key, value) in torch.load(pre_gaze).items())
            print('Loading gaze model from {}'.format(pre_gaze))
            self.model.load_state_dict(
                weihgt,
                strict=True
            )
            print("Complete loading Gaze estimation model weight")
        
        num_parameter = self.count_parameters(self.model)

        print(f"The number of parameters is {num_parameter / 1000 ** 2:.2f}M \n")

    def forward(self, le_c_list, re_c_list, face, head_batch_label):
        return self.model(le_c_list, re_c_list, face, head_batch_label)

    def count_parameters(self, model):
        param_sum = sum(p.numel() for p in model.parameters() if p.requires_grad)
        return param_sum