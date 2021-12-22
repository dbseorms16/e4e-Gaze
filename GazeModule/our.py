import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torchvision.utils import save_image
from GazeModule import dct
import os
import imageio

class GazeEstimationAbstractModel(nn.Module):

    def __init__(self):
        super(GazeEstimationAbstractModel, self).__init__()
        self.dct = dct.DCT_2D()
        self.idct = dct.IDCT_2D()

    @staticmethod
    def _create_fc_layers(in_features, out_features):
        xl_dct = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.BatchNorm1d(512, momentum=0.99, eps=1e-3),
            nn.ReLU(inplace=True)
        )

        xr_dct = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.BatchNorm1d(512, momentum=0.99, eps=1e-3),
            nn.ReLU(inplace=True)
        )
        
        x_l = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.BatchNorm1d(512, momentum=0.99, eps=1e-3),
            nn.ReLU(inplace=True)
        )
        x_r = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.BatchNorm1d(512, momentum=0.99, eps=1e-3),
            nn.ReLU(inplace=True)
        )

        face = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.BatchNorm1d(512, momentum=0.99, eps=1e-3),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2)
        )

        concat = nn.Sequential(
            nn.Linear(2560, 512),
            nn.BatchNorm1d(512, momentum=0.99, eps=1e-3),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2)
        )

        fc = nn.Sequential(
            nn.Linear(514, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, out_features),
            nn.Dropout(0.2)
        )

        return x_l, x_r, xl_dct, xr_dct, face, concat, fc
        # return x_l, x_r, face, concat, fc

    def forward(self, left_eye, right_eye, face, headpose):

        _,_, h, w = left_eye.size()
        mask = torch.ones((h, w), dtype=torch.int64, device = torch.device('cuda:0'))
        diagonal = w-(w//6)
        hf_mask = torch.fliplr(torch.triu(mask, diagonal)) != 1
        hf_mask = hf_mask.unsqueeze(0).expand(left_eye.size())

        ## left_eye_dct
        left_eye_dct = self.dct(left_eye)
        left_eye_dct = left_eye_dct * hf_mask
        left_eye_dct = self.idct(left_eye_dct)

        left_eye_dct = self.left_dct_features(left_eye_dct)
        left_eye_dct = torch.flatten(left_eye_dct, 1)
        left_eye_dct = self.xl_dct(left_eye_dct)

        ## left_eyendarr
        left_x = self.left_features(left_eye)
        left_x = torch.flatten(left_x, 1)
        left_x = self.xl(left_x)
        # torch.Size([7, 1024])z
        
        # left_eye_dct
        right_eye_dct = self.dct(right_eye)
        right_eye_dct = right_eye_dct * hf_mask
        right_eye_dct = self.idct(right_eye_dct)
        
        # for i in range(1):
        #     ndarr = right_eye_dct[i]

        #     n = torch.argmin(ndarr) 
        #     min = torch.flatten(ndarr)[n]

            
        #     m = torch.argmax(ndarr) 
        #     max = torch.flatten(ndarr)[m]
        #     ndarr = ndarr - min
        #     print(ndarr)
        #     ndarr = ndarr.byte().permute(1, 2, 0).cpu().numpy()
        #     imageio.imwrite('{}.jpg'.format(i), ndarr)


        right_eye_dct = self.left_dct_features(right_eye_dct)
        right_eye_dct = torch.flatten(right_eye_dct, 1)
        right_eye_dct = self.xr_dct(right_eye_dct)


        right_x = self.right_features(right_eye)
        right_x = torch.flatten(right_x, 1)
        right_x = self.xr(right_x)

        # torch.Size([7, 1024]) 
        #face dct\

        _,_, h, w = face.size()
        mask = torch.ones((h, w), dtype=torch.int64, device = torch.device('cuda:0'))
        diagonal = w-(w//6)
        hf_mask = torch.fliplr(torch.triu(mask, diagonal)) != 1
        hf_mask = hf_mask.unsqueeze(0).expand(face.size())

        face_dct = self.dct(face)
        face_dct = face_dct * hf_mask
        face_dct = self.idct(face_dct)

        # b,c,h,w = face_dct.size()
        
        face_dct = self.face_dct_features(face)
        face_dct = torch.flatten(face_dct, 1)
        face_dct = self.face(face_dct)

        features = torch.cat((left_x, left_eye_dct, right_x, right_eye_dct, face_dct ), dim=1)
        # features = torch.cat((left_x, left_eye_dct, right_x, right_eye_dct ), dim=1)
        # features = torch.cat((left_x, right_x ), dim=1)
        # print(eyes_x.shape)
        # torch.Size([7, 2048])

        features = self.concat(features)
        # self.concat은 FC layer
        # torch.Size([7, 512])

        features_headpose = torch.cat((features, headpose.unsqueeze(0)), dim=1)
        # torch.Size([7, 514])

        fc_output = self.fc(features_headpose)
        # torch.Size([7, 2])

        return fc_output


    @staticmethod
    def _init_weights(modules):
        for m in modules:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, mode="fan_in", nonlinearity="relu")
                nn.init.zeros_(m.bias)


class GazeEstimationModelResnet18(GazeEstimationAbstractModel):

    def __init__(self, num_out=2):
        super(GazeEstimationModelResnet18, self).__init__()
        _left_model = models.resnet18(pretrained=True)
        _right_model = models.resnet18(pretrained=True)
        _left_dct_model = models.resnet18(pretrained=True)
        _right_dct_model = models.resnet18(pretrained=True)
        _face_dct_model = models.resnet18(pretrained=True)
        # remove the last ConvBRelu layer
        self.left_features = nn.Sequential(
            _left_model.conv1,
            _left_model.bn1,
            _left_model.relu,
            _left_model.maxpool,
            _left_model.layer1,
            _left_model.layer2,
            _left_model.layer3,
            _left_model.layer4,
            _left_model.avgpool
        )

        self.right_features = nn.Sequential(
            _right_model.conv1,
            _right_model.bn1,
            _right_model.relu,
            _right_model.maxpool,
            _right_model.layer1,
            _right_model.layer2,
            _right_model.layer3,
            _right_model.layer4,
            _right_model.avgpool
        )
        self.left_dct_features = nn.Sequential(
            _left_dct_model.conv1,
            _left_dct_model.bn1,
            _left_dct_model.relu,
            _left_dct_model.maxpool,
            _left_dct_model.layer1,
            _left_dct_model.layer2,
            _left_dct_model.layer3,
            _left_dct_model.layer4,
            _left_dct_model.avgpool
        )
        self.right_dct_features = nn.Sequential(
            _right_dct_model.conv1,
            _right_dct_model.bn1,
            _right_dct_model.relu,
            _right_dct_model.maxpool,
            _right_dct_model.layer1,
            _right_dct_model.layer2,
            _right_dct_model.layer3,
            _right_dct_model.layer4,
            _right_dct_model.avgpool
        )
        self.face_dct_features = nn.Sequential(
            _face_dct_model.conv1,
            _face_dct_model.bn1,
            _face_dct_model.relu,
            _face_dct_model.maxpool,
            _face_dct_model.layer1,
            _face_dct_model.layer2,
            _face_dct_model.layer3,
            _face_dct_model.layer4,
            _face_dct_model.avgpool
        )
        for param in self.left_features.parameters():
            param.requires_grad = True
        for param in self.right_features.parameters():
            param.requires_grad = True
        for param in self.left_dct_features.parameters():
            param.requires_grad = True
        for param in self.right_dct_features.parameters():
            param.requires_grad = True
        for param in self.face_dct_features.parameters():
            param.requires_grad = True
    
        self.xl, self.xr, self.xl_dct, self.xr_dct, self.face, self.concat, self.fc = GazeEstimationAbstractModel._create_fc_layers(in_features=_left_model.fc.in_features, out_features=num_out)
        # self.xl, self.xr, self.concat, self.fc = GazeEstimationAbstractModel._create_fc_layers(in_features=_left_model.fc.in_features, out_features=num_out)
        # self.xl, self.xr = GazeEstimationAbstractModel._create_fc_layers(in_features=_left_model.fc.in_features, out_features=num_out)
        GazeEstimationAbstractModel._init_weights(self.modules())

def make_model():
    return GazeEstimationModelResnet18(num_out=2)

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