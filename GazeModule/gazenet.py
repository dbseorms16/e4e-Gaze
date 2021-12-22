import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torchvision.utils import save_image
import os
import imageio

class model(nn.Module):
    def __init__(self):
        super(model, self).__init__()
        
        vgg16 = models.vgg16(pretrained=True)

        self.convNet = vgg16.features

        self.FC = nn.Sequential(
            nn.Linear(512*4*7, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5)
        )

        self.output = nn.Sequential(
            nn.Linear(4096+2, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, 2),
        )

        # replace the maxpooling layer in VGG
        self.convNet[4] = nn.MaxPool2d(kernel_size=2, stride=1)
        self.convNet[9] = nn.MaxPool2d(kernel_size=2, stride=1)
      

    def forward(self, le_c_list, re_c_list, face, head_batch_label):
        feature = self.convNet(le_c_list)
        feature = torch.flatten(feature, start_dim=1)
        feature = self.FC(feature)

        feature = torch.cat((feature, head_batch_label), 1)
        gaze = self.output(feature)

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