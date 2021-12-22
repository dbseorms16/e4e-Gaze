from matplotlib.pyplot import contour
import torch
from functools import partial
import os
import sys
import cv2
from torchvision import datasets
from torchvision import transforms
import torch.nn as nn
import numpy as np


_loss_fn = {
            "mse": partial(torch.nn.MSELoss, reduction="sum")
        }
_param_num = {
    "mse": 2
}


# def generateEyePatches(sr_batch_size):
#     le_c_list = None
#     re_c_list = None
#     detected_list = []
#     eye_patch_dest = []
#     for i in range(len(sr_batch_size)):
#         faceboxes = [[0,0,224,224]]

#         minim_l, maxim_l, minim_r, maxim_r = eye_detection.eye_detector(sr_batch_size[i])
#         if(minim_l ==None or minim_r ==None):
#             # eye_detector가 양쪽 눈을 검출하지 못한다면, 해당 데이터셋을 skip 한다.
#             continue

#         # r,g,b 채널에 대하여 각각 crop한 후에, 병합한다.
#         left_r = sr_batch_size[i][0][minim_l[1]:maxim_l[1] + 1, minim_l[0]:maxim_l[0] + 1]
#         left_g = sr_batch_size[i][1][minim_l[1]:maxim_l[1] + 1, minim_l[0]:maxim_l[0] + 1]
#         left_b = sr_batch_size[i][2][minim_l[1]:maxim_l[1] + 1, minim_l[0]:maxim_l[0] + 1]

#         right_r = sr_batch_size[i][0][minim_r[1]:maxim_r[1] + 1, minim_r[0]:maxim_r[0] + 1]
#         right_g = sr_batch_size[i][1][minim_r[1]:maxim_r[1] + 1, minim_r[0]:maxim_r[0] + 1]
#         right_b = sr_batch_size[i][2][minim_r[1]:maxim_r[1] + 1, minim_r[0]:maxim_r[0] + 1]

#         le_c = torch.stack([left_r,left_g,left_b])
#         re_c = torch.stack([right_r,right_g,right_b])

#         # Tensor.cat은 해당 dimention을 증가시키며 tensor를 합한다.
#         # Tensor.stack은 새로운 dimention을 생성하여 tensor를 합한다.
#         if le_c.shape != (3,36,60) or re_c.shape != (3,36,60):
#             continue

#         if i ==0 or le_c_list == None:
#             le_c_list = le_c
#             le_c_list = torch.reshape(le_c_list, (1, 3, 36, 60))
#             re_c_list = re_c
#             re_c_list = torch.reshape(re_c_list, (1, 3, 36, 60))
#         else:
#             le_c = torch.reshape(le_c, (1, 3, 36, 60))
#             re_c = torch.reshape(re_c, (1, 3, 36, 60))
#             le_c_list = torch.cat([le_c_list, le_c], dim=0)
#             re_c_list = torch.cat([re_c_list, re_c], dim=0)

#         detected_list.append(i)
#         eye_patch_dest.append([i, minim_l[1], minim_l[0], minim_r[1], minim_r[0]])

#         if len(detected_list) == 0:
#             return None,None,None

#     return le_c_list, re_c_list, detected_list, eye_patch_dest


def computeGazeLoss(angular_out, gaze_batch_label):
    _criterion = _loss_fn.get("mse")()
    gaze_loss = _criterion(angular_out, gaze_batch_label).cuda()
    
    pi_angular_error = 0
    theta_angular_error = 0
    detected_number = angular_out.shape[0]
    for i in range(detected_number):

        # COMPUTE PI ANGULAR ERROR
        EST_pi = angular_out[i][0].cpu().detach().numpy()
        GT_pi = gaze_batch_label[i][0].cpu().detach().numpy()


        if EST_pi > GT_pi:
            pi_angular_error += np.abs(EST_pi - GT_pi)
        else:
            pi_angular_error += np.abs(GT_pi - EST_pi)
    
         # COMPUTE THETA ANGULAR ERROR
        EST_theta = angular_out[i][1].cpu().detach().numpy()
        GT_theta = gaze_batch_label[i][1].cpu().detach().numpy()

        if EST_theta > GT_theta:
            theta_angular_error += np.abs(EST_theta - GT_theta)
        else:
            theta_angular_error += np.abs(GT_theta - EST_theta)
    
    # 추후에 인식된 모든 사람 수를 고려하여 평균을 내주어야 한다.
    # angular_error는 Pi 와 Theta각도만을 고려하여 계산되었다.
    angular_error = (pi_angular_error + theta_angular_error)*45
        
    return gaze_loss, angular_error





def loadLabel(lines, names, type):
    gaze_batch_label = []
    head_batch_label = []
    flag = False

    # image_name : P02_000008.png,82,91,134,88
    # line : P02_000008.png,82,91,134,88

    # Batch_Image_Person_number : int(image_name[i].split("_")[1:])
    # Batch_Image_number : int(image_name[i].split("_")[:5])

    # Person_number : int(line.split("_")[1:])
    # Image_number : int(line.split("_")[:5])

    for name in names:

        batch_image_person_number = int(name.split("_")[0][1:])
        batch_image_number = int(name.split("_")[1][:6])


        if type == "train":
            if batch_image_person_number == 0:
                min = 0
                max = 4194
                mid = int((min+max)/2)

            elif batch_image_person_number == 1:
                min = 4195
                max = 8337
                mid = int((min+max)/2)
            
            elif batch_image_person_number == 2:
                min = 8338
                max = 12297
                mid = int((min+max)/2)

            elif batch_image_person_number == 4:
                min = 12298
                max = 15004
                mid = int((min+max)/2)

            elif batch_image_person_number == 6:
                min = 15005
                max = 20261
                mid = int((min+max)/2)

            elif batch_image_person_number == 7:
                min = 20262
                max = 24206
                mid = int((min+max)/2)

            elif batch_image_person_number == 8:
                min = 24207
                max = 26490
                mid = int((min+max)/2)

            elif batch_image_person_number == 9:
                min = 26491
                max = 28558
                mid = int((min+max)/2)

            elif batch_image_person_number == 10:
                min = 28559
                max = 32315
                mid = int((min+max)/2)

            elif batch_image_person_number == 11:
                min = 32316
                max = 37109
                mid = int((min+max)/2)

            elif batch_image_person_number == 12:
                min = 37110
                max = 37326
                mid = int((min+max)/2)

    
        if type == "validation":
            if batch_image_person_number == 0:
                min = 0
                max = 1149
                mid = int((min+max)/2)

            elif batch_image_person_number == 1:
                min = 1150
                max = 2265
                mid = int((min+max)/2)

            elif batch_image_person_number == 2:
                min = 2266
                max = 3125
                mid = int((min+max)/2)

            elif batch_image_person_number == 4:
                min = 3126
                max = 3989
                mid = int((min+max)/2)

            elif batch_image_person_number == 6:
                min = 3990
                max = 4842
                mid = int((min+max)/2)

            elif batch_image_person_number == 7:
                min = 4843
                max = 5657
                mid = int((min+max)/2)

            elif batch_image_person_number == 8:
                min = 5658
                max = 6419
                mid = int((min+max)/2)

            elif batch_image_person_number == 9:
                min = 6420
                max = 7109
                mid = int((min+max)/2)

            elif batch_image_person_number == 10:
                min = 7110
                max = 7862
                mid = int((min+max)/2)

            elif batch_image_person_number == 11:
                min = 7863
                max = 8781
                mid = int((min+max)/2)

            elif batch_image_person_number == 12:
                min = 8782
                max = 8999
                mid = int((min+max)/2)

        image_number = int(lines[mid].split("_")[1][:6]) 

        while True:
            batch_image_number = int(name.split("_")[1][:6])
            image_number = int(lines[mid].split("_")[1][:6])  
            # print(name.split("_")[1][:6])
            # print(batch_image_number)
            # print(lines[mid].split("_")[1][:6])
            # print(image_number)
            if batch_image_number == image_number:
                label = lines[mid].split(",")
                head_batch_label.append([float(label[1]),float(label[2])])
                gaze_batch_label.append([float(label[3]),float(label[4])])
                break

            elif batch_image_number < image_number:
                max = mid-1

            elif batch_image_number > image_number:
                min = mid + 1

            mid = int((min+max)/2)  


    head_batch_label= torch.FloatTensor(head_batch_label)
    gaze_batch_label = torch.FloatTensor(gaze_batch_label)

    return head_batch_label, gaze_batch_label


def generateEyePatches_fast(sr_batch_size, image_names, type='train'):
    le_c_list = None
    re_c_list = None
    flag = False
    detected_list = []

    if type == "train":
        labels = open("./dataset/Training_eye_coordinate(new).txt", "r")
    else:
        labels = open("./dataset/Validation_eye_coordinate(new).txt", "r")

    lines = labels.readlines()

    # image_name : P02_000008.png,82,91,134,88
    # line : P02_000008.png,82,91,134,88

    # Batch_Image_Person_number : int(image_name[i].split("_")[1:])
    # Batch_Image_number : int(image_name[i].split("_")[:5])

    # Person_number : int(line.split("_")[1:])
    # Image_number : int(line.split("_")[:5])

    for i in range(len(sr_batch_size)):
        try_num = 0
        # print("load patch ...", i)

        if type == "validation":
            image_names = [image_names]
        
        batch_image_person_number = int(image_names[i].split("_")[0][1:])
        batch_image_number = int(image_names[i].split(".")[0].split("_")[1][:6])  

        if type == "train":
            if batch_image_person_number == 0:
                min = 0
                max = 4194
                mid = int((min+max)/2)

            elif batch_image_person_number == 1:
                min = 4195
                max = 8337
                mid = int((min+max)/2)
            
            elif batch_image_person_number == 2:
                min = 8338
                max = 12297
                mid = int((min+max)/2)

            elif batch_image_person_number == 4:
                min = 12298
                max = 15004
                mid = int((min+max)/2)

            elif batch_image_person_number == 6:
                min = 15005
                max = 20261
                mid = int((min+max)/2)

            elif batch_image_person_number == 7:
                min = 20262
                max = 24206
                mid = int((min+max)/2)

            elif batch_image_person_number == 8:
                min = 24207
                max = 26490
                mid = int((min+max)/2)

            elif batch_image_person_number == 9:
                min = 26491
                max = 28558
                mid = int((min+max)/2)

            elif batch_image_person_number == 10:
                min = 28559
                max = 32315
                mid = int((min+max)/2)

            elif batch_image_person_number == 11:
                min = 32316
                max = 37109
                mid = int((min+max)/2)

            elif batch_image_person_number == 12:
                min = 37110
                max = 37326
                mid = int((min+max)/2)

    
        if type == "validation":
            if batch_image_person_number == 0:
                min = 0
                max = 1149
                mid = int((min+max)/2)

            elif batch_image_person_number == 1:
                min = 1150
                max = 2265
                mid = int((min+max)/2)

            elif batch_image_person_number == 2:
                min = 2266
                max = 3125
                mid = int((min+max)/2)

            elif batch_image_person_number == 4:
                min = 3126
                max = 3989
                mid = int((min+max)/2)

            elif batch_image_person_number == 6:
                min = 3990
                max = 4842
                mid = int((min+max)/2)

            elif batch_image_person_number == 7:
                min = 4843
                max = 5657
                mid = int((min+max)/2)

            elif batch_image_person_number == 8:
                min = 5658
                max = 6419
                mid = int((min+max)/2)

            elif batch_image_person_number == 9:
                min = 6420
                max = 7109
                mid = int((min+max)/2)

            elif batch_image_person_number == 10:
                min = 7110
                max = 7862
                mid = int((min+max)/2)

            elif batch_image_person_number == 11:
                min = 7863
                max = 8781
                mid = int((min+max)/2)

            elif batch_image_person_number == 12:
                min = 8782
                max = 8999
                mid = int((min+max)/2)

        person_number = int(lines[mid].split("_")[0][1:])
        image_number = int(lines[mid].split(".")[0].split("_")[1])  
        while True:
            batch_image_number = int(image_names[i].split(".")[0].split("_")[1])  
            image_number = int(lines[mid].split(".")[0].split("_")[1])  
            # print(image_names[i])
            # print("min : ",min,"\tmid : ",mid,"\tmax : ",max)
            # print("batch_image_number : ",batch_image_number)
            # print("image_number : ",image_number)

            if batch_image_number == image_number:
                break

            elif batch_image_number < image_number:
                max = mid-1
                try_num += 1

            elif batch_image_number > image_number:
                min = mid + 1
                try_num += 1
            

            mid = int((min+max)/2)  

            if try_num == 20:
                print(image_number)
                print(batch_image_number)
                flag == False
                continue

        if image_names[i] in lines[mid]:
            left_y = int(lines[mid].split(",")[2]) // 2
            left_x = int(lines[mid].split(",")[4]) // 2
            right_y =int(lines[mid].split(",")[1]) // 2
            right_x =int(lines[mid].split(",")[3]) // 2
            flag = True
    
        if flag == False:
            print("I Couldn't find eye coordinate")
            continue
        # r,g,b 채널에 대하여 각각 crop한 후에, 병합한다.
        # print(left_y, left_x, right_y, right_x)
        detected_list.append(i)
        # print(sr_batch_size.shape)
        left_r = sr_batch_size[i][0][left_y-18:left_y+18, left_x-30:left_x + 30]
        left_g = sr_batch_size[i][1][left_y-18:left_y+18, left_x-30:left_x + 30]
        left_b = sr_batch_size[i][2][left_y-18:left_y+18, left_x-30:left_x + 30]

        right_r = sr_batch_size[i][0][right_y-18:right_y+18, right_x-30:right_x + 30]
        right_g = sr_batch_size[i][1][right_y-18:right_y+18, right_x-30:right_x + 30]
        right_b = sr_batch_size[i][2][right_y-18:right_y+18, right_x-30:right_x + 30]

        if left_r.size() != (36, 60):
            shift = 60 - left_r.size()[1]
            yshift = 36 - left_r.size()[0]
            left_r = sr_batch_size[i][0][(left_y-yshift)-18:(left_y-yshift)+18, (left_x-shift)-30:(left_x-shift) + 30]
            left_g = sr_batch_size[i][0][(left_y-yshift)-18:(left_y-yshift)+18, (left_x-shift)-30:(left_x-shift) + 30]
            left_b = sr_batch_size[i][0][(left_y-yshift)-18:(left_y-yshift)+18, (left_x-shift)-30:(left_x-shift) + 30]
        
        if right_r.size() != (36, 60):
            shift = 60 - right_r.size()[1]
            yshift = 36 - right_r.size()[0]
            right_r = sr_batch_size[i][0][(right_y-yshift)-18:(right_y-yshift)+18, (right_x-shift)-30:(right_x-shift) + 30]
            right_g = sr_batch_size[i][0][(right_y-yshift)-18:(right_y-yshift)+18, (right_x-shift)-30:(right_x-shift) + 30]
            right_b = sr_batch_size[i][0][(right_y-yshift)-18:(right_y-yshift)+18, (right_x-shift)-30:(right_x-shift) + 30]

        le_c = torch.stack([left_r,left_g,left_b])
        re_c = torch.stack([right_r,right_g,right_b])
        # print(le_c.shape)


        if i ==0 or le_c_list == None:
            le_c_list = le_c
            le_c_list = torch.reshape(le_c_list, (1, 3, 36, 60))
            re_c_list = re_c
            re_c_list = torch.reshape(re_c_list, (1, 3, 36, 60))
        else:
            le_c = torch.reshape(le_c, (1, 3, 36, 60))
            re_c = torch.reshape(re_c, (1, 3, 36, 60))
            le_c_list = torch.cat([le_c_list, le_c], dim=0)
            re_c_list = torch.cat([re_c_list, re_c], dim=0)
    
    return le_c_list, re_c_list, detected_list