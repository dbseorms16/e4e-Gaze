import torch
import numpy as np
from torch.nn.modules import module


import utility
from decimal import Decimal
from tqdm import tqdm
from option import args

from torchvision import transforms 
from PIL import Image
import matplotlib
import matplotlib.pyplot as plt
import copy
import h5py
import os
import cv2
import numpy as np
from torchvision.utils import save_image
import torchvision
from getGazeLoss import *
# from getGazeLoss_RT_GENE import *


matplotlib.use("Agg")
class Trainer():
    def __init__(self, opt, loader, SR_model, gaze_model, my_loss,ckp):
        self.opt = opt
        self.scale = opt.scale
        self.ckp = ckp
        self.loader_train = loader.loader_train
        self.loader_test = loader.loader_test

        self.model = SR_model
        self.optimizer = utility.make_optimizer(opt, self.model)
        self.scheduler = utility.make_scheduler(opt, self.optimizer)

        self.gaze_model = gaze_model
        self.gaze_optimizer = utility.make_optimizer(opt, self.gaze_model)
        self.gaze_model_scheduler = utility.make_gaze_model_scheduler(opt, self.gaze_optimizer)
        self.loss = my_loss
        self.loss2 = my_loss
        self.scheduler = utility.make_scheduler(opt, self.optimizer)
        self.error_last = 1e8
        self.iteration_count = 0
        
        self.endPoint_flag = True
    
    def train(self):

        DETECTION_COUNT = 0
        Epoch_gaze_loss = 0
        Epoch_angular_error = 0

        Epoch_L1_loss = 0

        epoch = self.scheduler.last_epoch + 1
        lr = self.scheduler.get_last_lr()[0]
        self.ckp.set_epoch(epoch)

        label_txt = open("./dataset/Training_GT.txt" , "r")
        # label_txt = open("./dataset/Training_GT(new).txt" , "r")

        labels = label_txt.readlines()

        self.ckp.write_log(
            '[Epoch {}]\tLearning rate: {:.2e}'.format(epoch, Decimal(lr))
        )

        self.loss.start_log()

        # FOR PHASE TRAIN
        if self.opt.freeze == 'sr' :
            for name, param in self.model.named_parameters():
                param.requires_grad = False

        elif self.opt.freeze == 'gaze':
            for name, param in self.gaze_model.named_parameters():
                    param.requires_grad = False

        self.model.train(self.opt.freeze != 'sr')
        self.gaze_model.train(self.opt.freeze != 'gaze')

        timer_data, timer_model = utility.timer(), utility.timer()


        for batch, (lr, hr, image_names) in enumerate(self.loader_train):
            self.iteration_count += 1
            lr, hr = self.prepare(lr, hr)

            # SET GAZE ESTIMATOR'S GRADIENT TO ZERO.
            self.optimizer.zero_grad() 
            self.gaze_optimizer.zero_grad()
            timer_data.hold()
            timer_model.tic()

            # SR FORWARD ------------------------------------------------------------------------
            sr = self.model(lr[0])

            # 1. Compute primary loss
            loss_primary = self.loss(sr[-1], hr)
            # 3. Compute PSNR
            s = np.max(self.scale)
            if isinstance(sr, list): sr_for_psnr = sr[-1]
            sr_for_psnr = utility.quantize(sr_for_psnr, self.opt.rgb_range)

            # for i in range(hr.shape[0]):
            #             sr_one = torch.reshape(sr_for_psnr[i],(1, sr_for_psnr[i].shape[0],sr_for_psnr[i].shape[1],sr_for_psnr[i].shape[2]))
            #             hr_one = torch.reshape(hr[i],(1, hr[i].shape[0],hr[i].shape[1],hr[i].shape[2]))
            #             psnr = utility.calc_psnr(
            #                 sr_one, hr_one, s, self.opt.rgb_range,
            #                 benchmark=self.loader_test.dataset.benchmark
            #             )
            #             psnr +=  psnr/(float(hr.shape[0]) * self.opt.test_every)

            # 4. Compute SR loss for backward
            batch_L1_loss = (loss_primary)/ self.opt.batch_size
            # batch_loss_primary = loss_primary / self.opt.batch_size
            Epoch_L1_loss += batch_L1_loss.clone() / self.opt.test_every

            """
            SAVE SR IMAGE
            if self.opt.save_SR_image :
                image_root_path = "./train_SRImage"
                face_path = os.path.join(image_root_path, "face")
                os.makedirs(face_path, exist_ok = True)
                for i in range(len(sr[-1])):
                    save_image(sr[-1][i]/255, (os.path.join(face_path, image_names[i]+'.jpg')))

                if int(self.scheduler.last_epoch)% 1 == 0: 
                    face_path = os.path.join(image_root_path, "face/%d_epoch/"%self.scheduler.last_epoch)
                    os.makedirs(face_path, exist_ok = True)
                    for i in range(len(sr[-1])):
                        save_image(sr[-1][i]/255, (os.path.join(face_path, image_names[i]+'.jpg')))
            """
            
            # GET EYE PATCHES FOR TRAIN IMAGES.
            
            le_c_list , re_c_list, detected_list = generateEyePatches_fast(sr[-1], image_names, "train")

            # IF THERE ARE NOT DETECTED IMAGES, SKIP THAT BATCH.
            if len(detected_list) < 2:
                continue
            DETECTION_COUNT += len(detected_list)

            # IF THERE ARE NON DETECTED IMAGE, REMOVE ITS NAME FOR IMAGE_NAMES(LIST).
            new_image_names = []
            for i in detected_list:
                new_image_names.append(image_names[i])
            
            # LOAD HEAD & GAZE VECTORS CORRESPOND TO IMAGES. AND SET CUDA
            # head_batch_label, gaze_batch_label = loadLabel(labels, new_image_names)
            head_batch_label, gaze_batch_label = loadLabel(labels, new_image_names, 'train' )

            head_batch_label = head_batch_label.cuda()
            gaze_batch_label = gaze_batch_label.cuda()

            # GAZE FORWARD ------------------------------------------------------------------------
            angular_out = self.gaze_model(le_c_list, re_c_list, sr[-1], head_batch_label)
            gaze_loss, angular_error = computeGazeLoss(angular_out, gaze_batch_label)
            batch_gaze_loss = gaze_loss.clone()
            batch_angular_error = angular_error / self.opt.batch_size

            Epoch_gaze_loss += batch_gaze_loss / self.opt.test_every
            Epoch_angular_error += batch_angular_error / self.opt.test_every

            # DEFINE TOTAL LOSS FOR BACKWARD


            # SR BACKWARD ------------------------------------------------------------------------
            if self.opt.freeze == 'gaze':
                # print(loss_primary, gaze_loss)
                backward_loss = loss_primary + gaze_loss*30
                backward_loss.backward()
                self.optimizer.step()

            # GAZE BACKWARD ------------------------------------------------------------------------
            elif self.opt.freeze == 'sr':
                backward_loss = gaze_loss
                backward_loss.backward()
                self.gaze_optimizer.step()
                
            timer_model.hold()

            if (batch + 1) % self.opt.print_every == 0:
                self.ckp.write_log('[{}/{}]\t[Average Gaze Loss : {:.4f}]\t{:.1f}+{:.1f}s\t[Average Angular Error : {:.3f}]\t, [L1 loss : {:.3f}]'.format(
                    (batch + 1) * self.opt.batch_size,
                    len(self.loader_train.dataset),
                    batch_gaze_loss,
                    timer_model.release(),
                    timer_data.release(), batch_angular_error, batch_L1_loss))
            timer_data.tic()

        # TRACKING VARIABLES AND WRITE LOSSES FOR LOGS

        print('Train L1 loss', float(Epoch_L1_loss.item()))
        # print('Train PSNR', float(psnr))
        print('Train gaze loss : ', float(Epoch_gaze_loss.item()))
        print('Train Angular loss : ', float(Epoch_angular_error.item()))
        
        path_list = ["./experiments/Phase1(x4)/Train_L1_loss.txt","./experiments/Phase1(x4)/Train_gaze_loss.txt", "./experiments/Phase1(x4)/Train_angular_error.txt"]
        log_list = [float(Epoch_L1_loss.item()),  float(Epoch_gaze_loss.item()), float(Epoch_angular_error.item())]

        for i in range(3):
            txt = open(path_list[i], 'a')
            log = str(log_list[i]) + "\n"
            txt.write(log)
            txt.close()


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
        self.gaze_model.eval()
        timer_test = utility.timer()

        DETECTION_COUNT = 0
        Validation_gaze_loss = 0
        Validation_angular_error = 0
        Validation_L1_loss = 0
        total_gaze = 0
        loss_primary = 0

        label_txt = open("./dataset/Validation_GT.txt" , "r")
        # label_txt = open("./dataset/Validation_GT(new).txt" , "r")

        labels = label_txt.readlines()

        with torch.no_grad():
            scale = max(self.scale)

            for si, s in enumerate([scale]):
                eval_ssim = 0
                eval_psrn = 0
                tqdm_test = tqdm(self.loader_test, ncols=80)

                for _, (lr, hr, filename) in enumerate(self.loader_test):
                    filename = filename[0]
                    no_eval = (hr.nelement() == 1)
                    if not no_eval:
                        lr, hr = self.prepare(lr, hr)
                    else:
                        lr, = self.prepare(lr)

                    # SR FORWARD -------------------------------------------------------------
                    sr = self.model(lr[0])

                    if isinstance(sr, list): sr = sr[-1]

                    sr = utility.quantize(sr, self.opt.rgb_range)

                    # COMPUTE L1 LOSS
                    Validation_L1_loss += self.loss2(sr, hr).clone() / len(self.loader_test)

                    
                    # GET EYE PATCHES FOR TRAIN IMAGES.
                    le_c_list , re_c_list, detected_list = generateEyePatches_fast(sr, filename, "validation")
                    
                    # IF THERE ARE NOT DETECTED IMAGES, SKIP THAT IMAGE.
                    if (len(detected_list) == 0) or (type(le_c_list) != torch.Tensor):
                        continue
                    
                    DETECTION_COUNT += 1

                    # head_batch_label, gaze_batch_label = loadLabel(labels,[filename])
                    head_batch_label, gaze_batch_label = loadLabel(labels,[filename], 'validation' )

                    head_batch_label = head_batch_label.cuda()
                    gaze_batch_label = gaze_batch_label.cuda()

                    angular_out = self.gaze_model(le_c_list, re_c_list, sr, head_batch_label)
                    gaze_loss, angular_error = computeGazeLoss(angular_out, gaze_batch_label)
                    total_gaze += gaze_loss
                    Validation_gaze_loss += gaze_loss.clone() / len(self.loader_test)
                    Validation_angular_error += angular_error / len(self.loader_test)
                    # if angular_error > 15:
                    # ormat(filename, angular_error))

                    psrn = utility.calc_psnr(
                            sr, hr, scale, self.opt.rgb_range,
                            benchmark=self.loader_test.dataset.benchmark
                        )
                    eval_psrn += psrn
                    hr_numpy = hr[0].cpu().numpy().transpose(1, 2, 0)
                    sr_numpy = sr[0].cpu().numpy().transpose(1, 2, 0)
                    ssim = utility.SSIM(hr_numpy, sr_numpy)
                    eval_ssim += ssim

                # SAVE SR RESULTS
                    if self.endPoint_flag:
                        if self.opt.save_results:
                            self.ckp.save_results_nopostfix(filename, sr, s)

                avg_gaze_loss_val = total_gaze/DETECTION_COUNT

                # SR TRAIN
                # self.ckp.log[-1, si] = eval_psnr / len(self.loader_test)
                # best = self.ckp.log.max(0)
                # self.ckp.write_log(
                #     '[{} x{}]\tL1 LOSS: {:.4f} (Best: {:.4f} @epoch {})'.format(
                #         self.opt.data_test, s,
                #         self.ckp.log[-1, si],
                #         best[0][si],
                #         best[1][si] + 1
                #     )
                # )

                # GAZE TRAIN
                self.ckp.log[-1, si] = avg_gaze_loss_val
                best = self.ckp.log.min(0)  
                self.ckp.write_log(
                    '[{} x{}]\GAZE LOSS: {:.4f} (Best: {:.4f} @epoch {})'.format(
                        self.opt.data_test, s,
                        self.ckp.log[-1, si],
                        best[0][si],
                        best[1][si] + 1
                    )
                )
                    

        if not self.opt.test_only:
            self.ckp.save(self, epoch, is_best=(best[1][0] + 1 == epoch))
            Gaze_model_save(self.opt, self.gaze_model, epoch, is_best=(best[1][0] + 1 == epoch))

        self.ckp.write_log(
            'Total time: {:.2f}s\n'.format(timer_test.toc()), refresh=True
        )

        print('Validation L1 loss', float(Validation_L1_loss.item()))
        print('Validation PSNR', float(eval_psrn / len(self.loader_test)))
        print('Validation SSIM', float(eval_ssim / len(self.loader_test)))
        print('Validation gaze loss : ', float(Validation_gaze_loss.item()))
        print('Validation Angular loss : ', float(Validation_angular_error.item()))
        
        path_list = ["./experiments/Phase1(x4)/Validation_L1_loss.txt","./experiments/Phase1(x4)/Validation_PSNR.txt","./experiments/Phase1(x4)/Validation_gaze_loss.txt", "./experiments/phase1(x4)/Validation_angular_error.txt"]
        log_list = [float(Validation_L1_loss.item()), float(eval_psrn), float(Validation_gaze_loss.item()), float(Validation_angular_error.item())]

        for i in range(4):
            txt = open(path_list[i], 'a')
            log = str(log_list[i]) + "\n"
            txt.write(log)
            txt.close()



    def step(self):
        self.scheduler.step()
        self.gaze_model_scheduler.step()

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
            os.path.join(path, 'end_x3_latest.pt')
        )
        if is_best:
                torch.save(
                    gaze_model.state_dict(),
                    os.path.join(path, 'end_x3_best.pt')
                )