import torch
import numpy as np


import utility
from decimal import Decimal
from tqdm import tqdm
from option import args

from torchvision import transforms 
from PIL import Image
import matplotlib
import matplotlib.pyplot as plt
import copy
import os
import cv2
import numpy as np
from torchvision.utils import save_image
import torchvision

import torch.nn.functional as nnf

# from getGazeLoss_RT_GENE import *
from getGazeLoss import *

matplotlib.use("Agg")
class Trainer():
    def __init__(self, opt, loader, gaze_model, my_loss,ckp):
        self.opt = opt
        self.scale = opt.scale
        self.ckp = ckp
        self.loader_train = loader.loader_train
        self.loader_test = loader.loader_test
        self.model = gaze_model
        self.optimizer = utility.make_optimizer(opt, self.model)
        self.gaze_model_scheduler = utility.make_gaze_model_scheduler(opt, self.optimizer)
        self.loss = my_loss
        self.loss2 = my_loss
        self.scheduler = utility.make_scheduler(opt, self.optimizer)
        self.error_last = 1e8
        self.iteration_count = 0
        
        self.endPoint_flag = True
    
    def train(self):

        DETECTION_COUNT = 0
        TOTAL_gaze_loss = 0
        TOTAL_angular_error = 0

        epoch = self.scheduler.last_epoch + 1
        lr = self.scheduler.get_last_lr()[0]
        self.ckp.set_epoch(epoch)

        # label_txt = open("./dataset/Training_GT.txt" , "r")
        label_txt = open("./dataset/Training_GT(new).txt" , "r")
        labels = label_txt.readlines()

        self.ckp.write_log(
            '[Epoch {}]\tLearning rate: {:.2e}'.format(epoch, Decimal(lr))
        )

        self.loss.start_log()

        self.model.train()

        timer_data, timer_model = utility.timer(), utility.timer()


        for batch, (lr, hr, image_names) in enumerate(self.loader_train):
            self.iteration_count += 1
            lr, hr = self.prepare(lr, hr)

            # SET GAZE ESTIMATOR'S GRADIENT TO ZERO.
            self.optimizer.zero_grad() 
            
            timer_data.hold()
            timer_model.tic()
            
            # GET EYE PATCHES FOR TRAIN IMAGES.

            le_c_list , re_c_list, detected_list = generateEyePatches_fast(hr, image_names, "train")
            # IF THERE ARE NOT DETECTED IMAGES, SKIP THAT BATCH.
            # image_root_path = "./test_img"

            # face_path = os.path.join(image_root_path, "Left_eye(org)")
            # face_path_rcab = os.path.join(image_root_path, "Left_eye(RCAB)")
            # face_path_dct = os.path.join(image_root_path, "Left_eye(DCT)")

            # os.makedirs(face_path, exist_ok = True)
            # os.makedirs(face_path_rcab, exist_ok = True)
            # os.makedirs(face_path_dct, exist_ok = True)

            # for i in range(len(le_c_list)):
            #     save_image(le_c_list[i]/255, (os.path.join(face_path, image_names[i]+'.jpg')))
            
            # for i in range(len(re_c_list)):
            #     save_image(re_c_list[i]/255, (os.path.join(face_path_rcab, image_names[i]+'.jpg')))

            if len(detected_list) < 3:
                continue
            DETECTION_COUNT += len(detected_list)

            # IF THERE ARE NON DETECTED IMAGE, REMOVE ITS NAME FOR IMAGE_NAMES(LIST).
            new_image_names = []
            # index = torch.tensor(()).int().cuda()
            for i in detected_list:
            #     a = torch.tensor([i]).int().cuda()

            #     index = torch.cat((index, a),0)
                new_image_names.append(image_names[i])
            # hr = torch.index_select(hr, 0, index)

            # LOAD HEAD & GAZE VECTORS CORRESPOND TO IMAGES. AND SET CUDA
            head_batch_label, gaze_batch_label = loadLabel(labels, new_image_names)
            # head_batch_label, gaze_batch_label = loadLabel(labels, new_image_names, 'train' )

            head_batch_label = head_batch_label.cuda()
            gaze_batch_label = gaze_batch_label.cuda()

            # FORWARD GAZE ESTIMATOR
            angular_out = self.model(le_c_list, re_c_list, hr, head_batch_label)

            # print("angular_out : ",angular_out)
            gaze_loss, angular_error = computeGazeLoss(angular_out, gaze_batch_label)
            # print("Gaze loss : ",gaze_loss.item()/len(detected_list), "\tAngular Error : ",angular_error.item()/len(detected_list))

            TOTAL_gaze_loss += gaze_loss
            TOTAL_angular_error += angular_error

            # GAZE ESTIMATOR UPDATE
            gaze_loss.backward()
            self.optimizer.step()

            # SAZE IMAGES
            
                
            timer_model.hold()

            if (batch + 1) % self.opt.print_every == 0:
                self.ckp.write_log('[{}/{}]\t[Average Gaze Loss : {:.4f}]\t{:.1f}+{:.1f}s\t[Average Angular Error:{:.3f}]'.format(
                    (batch + 1) * self.opt.batch_size,
                    len(self.loader_train.dataset),
                    TOTAL_gaze_loss / DETECTION_COUNT,
                    timer_model.release(),
                    timer_data.release(),TOTAL_angular_error / DETECTION_COUNT))
            timer_data.tic()

        # TRACKING VARIABLES AND WRITE LOSSES FOR LOGS
        AVG_gaze_loss = TOTAL_gaze_loss / DETECTION_COUNT
        AVG_angular_error = TOTAL_angular_error / DETECTION_COUNT
        print('Train gaze loss : ', float(AVG_gaze_loss.item()))
        print('Train Angular loss : ', float(AVG_angular_error.item()))
        
        path_list = ["./experiments/gaze_train/Train_gaze_loss.txt", "./experiments/gaze_train/Train_angular_error.txt"]
        log_list = [float(AVG_gaze_loss.item()), float(AVG_angular_error.item())]

        # for i in range(2):
        #     txt = open(path_list[i], 'a')
        #     log = str(log_list[i]) + "\n"
        #     txt.write(log)
        #     txt.close()


        # ENDPOINT OF EPOCH
        self.loss.end_log(len(self.loader_train))
        self.error_last = self.loss.log[-1, -1]
        self.step()
      
    def test(self):
        self.loss2.start_log()

        epoch = self.scheduler.last_epoch
        self.ckp.write_log('\nEvaluation:')
        self.ckp.add_log(torch.zeros(1, 1))
        self.model.eval()
        timer_test = utility.timer()

        DETECTION_COUNT = 0
        TOTAL_gaze_loss = 0
        TOTAL_angular_error = 0
        # label_txt = open("./dataset/Validation_GT(new).txt" , "r")
        label_txt = open("./dataset/Validation_GT.txt" , "r")
        data_txt = open("./dataset/demo_bic.txt" , "w")
        labels = label_txt.readlines()
        eval_ssim = 0
        eval_psrn = 0
        head_pose = open("./dataset/head.txt" , "r")
        with torch.no_grad():
            scale = max(self.scale)

            for si, s in enumerate([scale]):
                tqdm_test = tqdm(self.loader_test, ncols=80)
                for _, (lr, hr, filename) in enumerate(tqdm_test):
                    filename = filename[0]
                    no_eval = (hr.nelement() == 1)
                    if not no_eval:
                        lr, hr = self.prepare(lr, hr)
                    else:
                        lr, = self.prepare(lr)

                    N,C,H,W = hr.size()

                    # inH , inW =  int(H / total_scale), int(W / total_scale)
                    # intH, intW = int(inH * int_scale), int(inW * int_scale)
                    # outH, outW = int(intH * res_scale), int(intW * res_scale)
                    # hr = hr[:, :, : int(outH//2*2),  :int(outW//2*2)]
                    # hr = hr[:, :, : outH,  :outW]
                    hr = nnf.interpolate(lr[0], size=(112, 112), mode='bicubic', align_corners=False).to('cuda:0')
                    hr = utility.quantize(hr, self.opt.rgb_range)
                    
                    # GET EYE PATCHES FOR TRAIN IMAGES.

                    le_c_list , re_c_list, detected_list = generateEyePatches_fast(hr, filename, "validation")
                    # IF THERE ARE NOT DETECTED IMAGES, SKIP THAT IMAGE.
                    if (len(detected_list) == 0) or (type(le_c_list) != torch.Tensor) :
                        continue

                    
                    DETECTION_COUNT += 1

                    # head_batch_label, gaze_batch_label = loadLabel(labels,[filename], 'validation' )
                    # head_batch_label, gaze_batch_label = loadLabel(labels,[filename] )
                    # print(gaze_batch_label)
                    string_ =head_pose.readline()
                    head_the = string_.split(" ")[0]
                    head_pi = string_.split(" ")[1]

                    head_batch_label = []
                    head_batch_label.append(float(head_the))
                    head_batch_label.append(float(head_pi))
                    head_batch_label = torch.FloatTensor(head_batch_label)
                    head_batch_label = head_batch_label.cuda()

                    # gaze_batch_label = gaze_batch_label.cuda()
                    angular_out = self.model(le_c_list, re_c_list, hr, head_batch_label)
                    data = '{} {} {}\n'.format(filename, angular_out[0][0], angular_out[0][1])
                    data_txt.write(data)
                    # gaze_loss, angular_error = computeGazeLoss(angular_out, gaze_batch_label)
                    # if angular_error > 10 :
                    #     print('{} {:4f}'.format(filename, angular_error))
                    # psrn = utility.calc_psnr(
                    #         hr, hr, scale, self.opt.rgb_range,
                    #         benchmark=self.loader_test.dataset.benchmark
                    #     )
                    # eval_psrn += psrn
                    # hr_numpy = hr[0].cpu().numpy().transpose(1, 2, 0)
                    # sr_numpy = lr[0][0].cpu().numpy().transpose(1, 2, 0)
                    # ssim = utility.SSIM(hr_numpy, sr_numpy)
                    # eval_ssim += ssim

                    self.ckp.save_results_nopostfix(filename, hr, 4)
                    # TOTAL_gaze_loss += gaze_loss
                    # TOTAL_angular_error += angular_error
                self.ckp.log[-1, si] = TOTAL_gaze_loss / len(self.loader_test)
                best = self.ckp.log.min(0)
                self.ckp.write_log(
                    '[{} x{}]\tGaze Loss: {:.4f} (Best: {:.4f} @epoch {})'.format(
                        self.opt.data_test, s,
                        self.ckp.log[-1, si],
                        best[0][si],
                        best[1][si] + 1
                    )
                )
                # print('eval_psrn :', eval_psrn / len(self.loader_test) )
                # print('eval_ssim :', eval_ssim / len(self.loader_test) )

        if not self.opt.test_only:
            Gaze_model_save(self.opt, self.model, epoch, is_best=(best[1][0] + 1 == epoch))

        self.ckp.write_log(
            'Total time: {:.2f}s\n'.format(timer_test.toc()), refresh=True
        )

        AVG_gaze_loss = TOTAL_gaze_loss / DETECTION_COUNT  
        AVG_angular_error = TOTAL_angular_error / DETECTION_COUNT

        print('Validation gaze loss : ', float(AVG_gaze_loss.item()))
        print('Validation Angular loss : ', float(AVG_angular_error.item()))
        
        path_list = ["./experiments/gaze_train/Validation_gaze_loss.txt", "./experiments/gaze_train/Validation_angular_error.txt"]
        log_list = [float(AVG_gaze_loss.item()), float(AVG_angular_error.item())]

        # for i in range(2):
        #     txt = open(path_list[i], 'a')
        #     log = str(log_list[i]) + "\n"
        #     txt.write(log)
        #     txt.close()



    def step(self):
        self.scheduler.step()

    def prepare(self, *args):
        device = torch.device('cpu' if self.opt.cpu else 'cuda')

        if len(args) > 1:
            return [a.to(device) for a in args[0]], args[-1].to(device)
        return [a.to(device) for a in args[0]],

    def terminate(self):
        if self.opt.test_only:
            self.test()
            return True
        else:
            epoch = self.scheduler.last_epoch
            return epoch >= self.opt.epochs

def Gaze_model_save(opt, gaze_model, epoch, is_best=False):
        path = opt.gaze_model_save_path
        torch.save(
            gaze_model.state_dict(), 
            os.path.join(path, 'dlnet_latest.pt')
        )
        if is_best:
                torch.save(
                    gaze_model.state_dict(),
                    os.path.join(path, 'dlnet_best.pt')
                )