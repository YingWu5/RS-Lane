# -*- encoding: utf-8 -*-
import os
import time

import numpy as np
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from loss.loss import discriminative_loss,SAD_loss
from tools.dataset import TuSimpleDataset
from model.model import Resnest_LaneNet
from loss.dice_loss import GDiceLossV2

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

def init_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, help='path to CULane dataset')
    parser.add_argument('--ckpt_path', type=str, help='path to parameter file (.pth); If None, training from the beginning')
    parser.add_argument('--save_ckpt', type=str, default='save_ckpt', help='path to parameter file (.pth) while training')
    parser.add_argument('--epoch', type=int, default=10, help='training epoch number')
    parser.add_argument('--label', type=str, help='label to denote details of training')

    return parser.parse_args()

def init_weights(model):
    if type(model) in [nn.Conv2d, nn.ConvTranspose2d, nn.Linear]:
        torch.nn.init.xavier_uniform_(model.weight)
        if model.bias is not None:
            model.bias.data.fill_(0.01)

def print_info(summerwriter,epoch,step, on_val ,loss,bin_loss,bin_perception,bin_recall,bin_F1):

    if on_val:
        summerwriter.add_scalar('total_val_loss', loss, step)
        summerwriter.add_scalar('bin_val_loss', bin_loss, step)
        summerwriter.add_scalar('val_perception', bin_perception, step)
        summerwriter.add_scalar('val_recall', bin_recall, step)
        summerwriter.add_scalar('val_F1', bin_F1, step)
    else:
        summerwriter.add_scalar('total_train_loss', loss, step)
        summerwriter.add_scalar('bin_train_loss', bin_loss, step)
        summerwriter.add_scalar('train_perception', bin_perception, step)
        summerwriter.add_scalar('train_recall', bin_recall, step)
        summerwriter.add_scalar('train_F1', bin_F1, step)

    print('Epoch:{}  Step:{}  TrainLoss:{:.5f}  Bin_Loss:{:.5f} '
            'bin_perception:{:.5f}  bin_recall:{:.5f}  bin_F1:{:.5f}'
            .format(epoch, step, loss, bin_loss,
                    bin_perception, bin_recall, bin_F1))

def train(data_dir,ckpt_path,save_path,epoch_num,label):

    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        device_count = torch.cuda.device_count()
        print("Let's use", torch.cuda.device_count(),"GPUs!")
    else:
        device = torch.device("cpu")
        print("Using CPU!")

    batch_size = 4 * device_count
    learning_rate = 0.001  # 
    epoch_num = epoch_num
    num_workers = 4
    ckpt_epoch_interval = 50  # save a model checkpoint every X epochs 
    val_step_interval = 50  # perform a validation step every X traning steps
    train_start_time = time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime(time.time()))

    train_set = TuSimpleDataset(data_dir, 'train')
    val_set = TuSimpleDataset(data_dir, 'val')


    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers,drop_last=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers,drop_last=True)
    print('Finish loading data from %s' % data_dir)


    writer = SummaryWriter(log_dir='summary/lane-detect-%s' % (label))

    net = Resnest_LaneNet()
    net = nn.DataParallel(net)
    net.to(device)
    optimizer = optim.Adam(net.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer,T_0=50,T_mult=2)
    MSELoss = nn.MSELoss()

    if ckpt_path is not None:
        checkpoint = torch.load(ckpt_path)
        net.load_state_dict(checkpoint['model_state_dict'], strict=False) 
        # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        # epoch = checkpoint['epoch']
        # step = checkpoint['step']
        step = 0  # by default, we reset step and epoch value
        epoch = 1
        loss = checkpoint['loss']
        print('Checkpoint loaded.')

    else:
        net.apply(init_weights)
        step = 0
        epoch = 1
        print('Network parameters initialized.')
    
   
    for epoch in range(epoch_num):

        for batch_idx, batch in enumerate(train_loader):
            start_time = time.time()        
            net.train()
            
            inputs = batch['input_tensor']
            labels_bin = batch['binary_tensor'] 
            labels_inst = batch['instance_tensor']

            inputs = inputs.to(device)
            labels_bin = labels_bin.to(device)
            labels_inst = labels_inst.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward
            embeddings ,logit, layer = net(inputs)

            # compute loss
            preds_bin = torch.argmax(logit, dim=1, keepdim=True)
            preds_bin_expand = preds_bin.view(preds_bin.shape[0] * preds_bin.shape[1] * preds_bin.shape[2] * preds_bin.shape[3])
            labels_bin_expand = labels_bin.view(labels_bin.shape[0] * labels_bin.shape[1] * labels_bin.shape[2])

            '''DICE Loss'''
            logit = F.softmax(logit, dim=1)
            dice = GDiceLossV2()
            bin_loss = 1 + dice(logit,labels_bin)

            # discriminative loss
            embedding_loss = discriminative_loss(embeddings,
                                                labels_inst,
                                                delta_v=0.2,
                                                delta_d=1,
                                                param_var=.5,
                                                param_dist=.5,
                                                param_reg=0.001)

            loss = bin_loss  + embedding_loss * 0.1

            '''SAD Loss'''
            sadloss = torch.tensor(0)
            if epoch > 5:

                sadloss1 = SAD_loss(layer[2],layer[3])
                sadloss2 = SAD_loss(layer[3],layer[4])

                sadloss = sadloss1  + sadloss2 
                loss = loss + sadloss

            loss.backward()
            optimizer.step()
            scheduler.step()
            
            # Statistics
            bin_TP = torch.sum((preds_bin_expand.detach() == labels_bin_expand.detach()) & (preds_bin_expand.detach() == 1))
            bin_perception = bin_TP.double() / (torch.sum(preds_bin_expand.detach() == 1).double() + 1e-6)
            bin_recall = bin_TP.double() / (torch.sum(labels_bin_expand.detach() == 1).double() + 1e-6)
            bin_F1 = 2 * bin_perception * bin_recall / (bin_perception + bin_recall)

            
            step = epoch * len(train_loader) + batch_idx
            if step % ckpt_epoch_interval == 0:
                ckpt_dir = save_path
                if os.path.exists(ckpt_dir) is False:
                    os.makedirs(ckpt_dir)
                ckpt_path = os.path.join(ckpt_dir, 'ckpt_%s_epoch-%d.pth' % (label, batch_idx))
                torch.save({
                    'epoch': epoch,
                    'step': step,
                    'model_state_dict': net.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss,
                }, ckpt_path)

            print_info(writer, epoch, step, False , loss.item(), bin_loss.item(), 
                        bin_perception.item(), bin_recall.item(), bin_F1.item())
                      
        with torch.no_grad():
            val(net,val_loader,epoch,device,writer,label)
          
def val(model, data_loader, epoch_idx, device, summerwriter):

    for batch_idx, batch in enumerate(data_loader):
                    
        model.train()
        
        inputs = batch['input_tensor']
        labels_bin = batch['binary_tensor'] 
        labels_inst = batch['instance_tensor']

        inputs = inputs.to(device)
        labels_bin = labels_bin.to(device)
        labels_inst = labels_inst.to(device)

        # forward
        embeddings ,logit, layer = model(inputs)
        # compute loss
        preds_bin = torch.argmax(logit, dim=1, keepdim=True)
        preds_bin_expand = preds_bin.view(preds_bin.shape[0] * preds_bin.shape[1] * preds_bin.shape[2] * preds_bin.shape[3])
        labels_bin_expand = labels_bin.view(labels_bin.shape[0] * labels_bin.shape[1] * labels_bin.shape[2])

        '''DICE Loss'''
        logit = F.softmax(logit, dim=1)
        dice = GDiceLossV2()
        bin_loss = 1 + dice(logit,labels_bin)

        # discriminative loss
        embedding_loss = discriminative_loss(embeddings,
                                            labels_inst,
                                            delta_v=0.2,
                                            delta_d=1,
                                            param_var=.5,
                                            param_dist=.5,
                                            param_reg=0.001)

        loss = bin_loss  + embedding_loss * 0.1

        # Statistics
        bin_TP = torch.sum((preds_bin_expand.detach() == labels_bin_expand.detach()) & (preds_bin_expand.detach() == 1))
        bin_perception = bin_TP.double() / (torch.sum(preds_bin_expand.detach() == 1).double() + 1e-6)
        bin_recall = bin_TP.double() / (torch.sum(labels_bin_expand.detach() == 1).double() + 1e-6)
        bin_F1 = 2 * bin_perception * bin_recall / (bin_perception + bin_recall)

        step = epoch_idx * len(data_loader) + batch_idx
        print_info(summerwriter, epoch_idx, step, True , loss.item(), bin_loss.item(), 
                    bin_perception.item(), bin_recall.item(), bin_F1.item())
            


if __name__ == '__main__':
    
    args = init_args()
    train(args.data_dir,args.ckpt_path,args.save_path,args.epoch,args.label)