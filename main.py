import argparse
import re
import os, glob, datetime, time
import numpy as np
import torch
import torch.nn as nn
from torch.nn.modules.loss import _Loss
import torch.nn.init as init
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR,ExponentialLR,CosineAnnealingLR
from dataset import *
from utils import *
from model import *

import random

QT_FACTOR = 40
# Params
parser = argparse.ArgumentParser(description='PyTorch ARCNN/FastARCNN/DnCNN')
parser.add_argument('--model_name', default='DnCNN', type=str, help='define the name of model')
parser.add_argument('--model_choice', default='DnCNN', type=str, help='choose the model type')
parser.add_argument('--QT_FACTOR', default=40, type=int, help='quantization factor')
parser.add_argument('--train_epoch', default=60, type=int, help='number of train epoches')  # 180
parser.add_argument('--lr', default=1e-4, type=float, help='initial learning rate for Adam')  # 1e-3
parser.add_argument('--train_batchsize', default=4, type=int, help='training batches')
parser.add_argument('--test_batchsize', default=80, type=int, help='training batches')
parser.add_argument('--test_interval', default=100, type=int, help='testing interval')
parser.add_argument('--decayrate', default=0.1, type=float, help='decay rate')
parser.add_argument('--InputFolder', default='data/JPEG', type=str, help='inputfolder')
parser.add_argument('--LabelFolder', default='data/GT', type=str, help='LabelFolder')
parser.add_argument('--Trainlist', default='data/train_list.txt', type=str,
                    help='trainlist')  # 64612 *7
parser.add_argument('--Testlist', default='data/val_list.txt', type=str,
                    help='testlist')  # 7824

if __name__ == '__main__':
    args = parser.parse_args()

    print('args:++++++++++++++++++++++')
    for k, v in sorted(vars(args).items()):
        print(str(k) + ": " + str(v))
    print('+++++++++++++++++++++++++++')

    QT_FACTOR = args.QT_FACTOR

    batch_size = args.train_batchsize
    batch_size_test = args.test_batchsize
    test_interval = args.test_interval
    # gpu_num = args.GPU_NUM
    full_batch_size = batch_size   # * gpu_num
    cuda = torch.cuda.is_available()
    n_epoch = args.train_epoch

    save_dir = args.model_name + '_' + 'q' + str(QT_FACTOR)

    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # model selection
    print('===> Building model')
    if args.model_choice == 'ARCNN':
        model = ARCNN()
    elif args.model_choice == 'FastARCNN':
        model = FastARCNN()
    elif args.model_choice == 'DnCNN':
        model = DnCNN()
    elif args.model_choice =='DRN':
        model = DenseResNet()        # new model
    else:
        assert False, 'ERROR: no specific model choice !!!'

        
    # build criterion
    criterion = mean_squared_error()
    
    # initial the performance recorded during the validation
    psnr_pre = 0
    psnr_input = 0
    psnr_best  = 0
        
    # multi-GPU
    if cuda :
        print("use", torch.cuda.device_count(), "GPUs for training")
        if  (torch.cuda.device_count()) > 1:
            device_ids = list(range(torch.cuda.device_count()))
            print("multiple GPU detected, use parallel")
            model = nn.DataParallel(model, device_ids=device_ids).cuda()
        model.to(device)
        criterion = criterion.cuda()
    else:
        print("use CPU for training")

    model.train()
    
    # build optimizer and scheduler
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    #scheduler = MultiStepLR(optimizer, milestones=[10, 20, 30,40,50,60], gamma=0.2)  # learning rates milestones=[30, 60, 90]
    #scheduler = ExponentialLR(optimizer,gamma=0.1)
    scheduler = CosineAnnealingLR(optimizer,60,0)
    
    # first get all train label name list & test label name list
    stage_flag = 'train'
    full_train_image_list = get_full_name_list(args.Trainlist, stage_flag)
    stage_flag = 'test'
    full_test_image_list = get_full_name_list(args.Testlist, stage_flag)

    n_iterations = int((len(full_train_image_list) / full_batch_size) * args.train_epoch)

    print('start loading batch and training...')
    origin_epoch = 0
    origin_epoch_test = 0
    
    # record the loss and predicted psnr during the validation
    f = open("loss.txt","w+")
    f.close()
    f = open("psnr_pre.txt","w+")
    f.close()
    
    for iteration in range(1, n_iterations):
        epoch = int((iteration * full_batch_size) / len(full_train_image_list))

        # shuffle images per epoch
        if epoch > origin_epoch:
            origin_epoch = epoch
            print('shuffle train image list...')
            random.shuffle(full_train_image_list)

        if (iteration * full_batch_size) > len(full_train_image_list):
            index_end = int((iteration * full_batch_size) - (epoch * len(full_train_image_list)))
            index_start = int(index_end - full_batch_size)
        else:
            index_start = int((iteration - 1) * full_batch_size)
            index_end = int(iteration * full_batch_size)

        batch_paths = []
        for index in range(index_start, index_end):
            batch_paths.append(full_train_image_list[index])

        scheduler.step(epoch)  # step to the learning rate in this epoch
        epoch_loss = 0
        start_time = time.time()

        xs = generate_single_batch(train_input_dir=args.InputFolder, train_label_dir=args.LabelFolder,
                                      batch_paths=batch_paths, crop_flag=True)
        
        for i in range(0, len(xs)):  # (7, 256, 448, 3)
            xs[i] = (xs[i].astype('float32') / 255.0)
            xs[i] = torch.from_numpy(xs[i].transpose((0, 3, 1, 2)))  # torch.Size([7, 3, 256, 448])
        DDataset = ArtifactDataset(xs)
        DLoader = DataLoader(dataset=DDataset, num_workers=4, drop_last=True, batch_size=batch_size, shuffle=True)

        loss_single_batch = 0
        
        model.train()
        
        for n_count, batch_yx in enumerate(DLoader):  # batch_size = 7 n_count = 0
            optimizer.zero_grad()
            # batch_x: label batch_y:input
            batch_x, batch_y = batch_yx[1], batch_yx[0]
            if cuda:
                batch_x, batch_y = batch_x.cuda(), batch_y.cuda()
                batch_x.to(device)
                batch_y.to(device)

            loss = criterion(model(batch_y), batch_x)
            loss_single_batch += loss.item()
            epoch_loss += loss.item()

            loss.backward()
            optimizer.step()

        elapsed_time = time.time() - start_time
        if (iteration % 50 == 0) and (iteration > 0):
            
            # record the loss
            f = open("loss.txt","a+")
            f.write("%4d epoch %4d loss: %2.6f \n" % (iteration, epoch+1, loss_single_batch))  
            f.close()
            
            print('%4d %4d / %4d loss = %2.6f , time = %4.2f s' % (
            epoch + 1, n_iterations, iteration, loss_single_batch, elapsed_time))  # /(batch_size*iteration)
            
        if iteration % test_interval == 0:
            # test
            print('start testing..model.eval()')
            model.eval()
            test_iter_num = iteration / test_interval
            full_batch_size_test = batch_size_test   #* gpu_num
            epoch_test = int((test_iter_num * full_batch_size_test) / len(full_test_image_list))

            if (test_iter_num * full_batch_size_test) > len(full_test_image_list):
                index_end = int((test_iter_num * full_batch_size_test) - (epoch_test * len(full_test_image_list)))
                index_start = int(index_end - full_batch_size_test)
            else:
                index_start = int((test_iter_num - 1) * full_batch_size_test)
                index_end = int(test_iter_num * full_batch_size_test)
            
            # fix the index range so that each validation has same input psnr, better to track the improvement
            index_start = 0
            index_end = 80
            
            batch_paths = []
            for index in range(index_start, index_end):
                batch_paths.append(full_test_image_list[index])

            xs_test = generate_single_batch(train_input_dir=args.InputFolder, train_label_dir=args.LabelFolder,
                                               batch_paths=batch_paths, crop_flag=False)

            for i in range(0, len(xs_test)):
                xs_test[i] = (xs_test[i].astype('float32') / 255.0)
                xs_test[i] = torch.from_numpy(xs_test[i].transpose((0, 3, 1, 2)))  # torch.Size([7, 3, 256, 448])

            DDataset_test = ArtifactDataset(xs_test)
            DLoader_test = DataLoader(dataset=DDataset_test, num_workers=4, drop_last=True, batch_size=batch_size_test,
                                      shuffle=False)
            
            for n_count, batch_yx in enumerate(DLoader_test):
                if cuda:
                    batch_x, batch_y = batch_yx[1].cuda(), batch_yx[0].cuda()  # batch_x: label batch_y:input
                    batch_x.to(device)
                    batch_y.to(device)
                psnrs = calculate_psnr(batch_y, model(batch_y), batch_x)
                psnr_pre = psnrs[0]
                psnr_input = psnrs[1]
                
                # save the best model
                if psnr_pre > psnr_best:
                    psnr_best = psnr_pre
                    print('saving model ' + str(epoch + 1))
                    torch.save({
                        'epoch': epoch + 1,
                        'state_dict': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                    }, os.path.join(save_dir, 'epoch' + '_%03d.pth.tar' % (epoch + 1)))
                
                # record the predicted psnr
                f = open("psnr_pre.txt","a+")
                f.write("%4d psnr_pre = %2.4f, psnr_input = %2.4f \n" % (iteration, psnr_pre, psnr_input))
                f.close()
                
                print('Test: psnr_pre = %2.4f, psnr_input = %2.4f, psnr_diff = %2.4f ' % (
                psnr_pre, psnr_input, abs(psnr_pre - psnr_input)))
