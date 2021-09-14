import os
import time
import argparse
import ujson as json
import numpy as np
import cv2

import torch 
from torch.utils.data import DataLoader
import torch.nn as nn

from tools.utils import cluster_embed,CULanefit,generate_CULane,split_path
from tools.dataset import CULane
from model.model import Resnest_LaneNet

def init_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, help='path to CULane dataset')
    parser.add_argument('--ckpt_path', type=str, help='path to parameter file (.pth)')
    parser.add_argument('--save_path', type=str, default='output', help='path to save dir')
    parser.add_argument('--show', action='store_true', help='whether to show visualization images')
    parser.add_argument('--save_img', action='store_true', help='whether to save visualization images')
    parser.add_argument('--label', type=str, help='label to denote details of experiments')

    return parser.parse_args()

def test_culane(data_dir,ckpt_path,save_path,show,save,label):


    '''Test config'''
    batch_size = 1
    num_workers = 4
    test_start_time = time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime(time.time()))

    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using GPU!")
    else:
        device = torch.device("cpu")
        print("Using CPU!")
    print("Batch size: %d" % batch_size)

    output_dir = '%s/Test-%s' % (save_path,label)
    if os.path.exists(output_dir) is False:
        os.makedirs(output_dir)

    
    test_set = CULane(path = data_dir, image_set='test')

    testset_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    print('Finish loading data from %s' % data_dir)

    '''Constant variables'''
    VGG_MEAN = np.array([103.939, 116.779, 123.68]).astype(np.float32)
    _, h, w = test_set[0]['input_tensor'].shape
    
    # y_start, y_stop and y_num is calculated according to TuSimple Benchmark's setting
    y_sample = np.linspace( 590, 10 , 59 , dtype=np.int16)
    
    '''Forward propogation'''
    with torch.no_grad():
        
        net = Resnest_LaneNet()
        net = nn.DataParallel(net)

        net.to(device)
        net.eval()

        assert ckpt_path is not None, 'Checkpoint Error.'

        checkpoint = torch.load(ckpt_path)
        net.load_state_dict(checkpoint['model_state_dict'], strict=True)

        step = 0

        time_run_avg = 0
        time_fp_avg = 0
        time_clst_avg = 0
        time_fit_avg = 0
        time_ct = 0
        jsonlist = list()
        
        for batch_idx, batch in enumerate(testset_loader):
            
            time_run = time.time()
            time_fp = time.time()

            input_batch = batch['input_tensor']
            raw_file_batch = batch['raw_file']

            # generate prediction save path
            path_tree = split_path(raw_file_batch[0])
            save_dir, save_name = path_tree[-3:-1], path_tree[-1]
            save_dir = os.path.join(output_dir, *save_dir)
            save_name = save_name[:-3] + "lines.txt"
            save_frame_name = save_name[:-3]
            save_name = os.path.join(save_dir, save_name)
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)

            # forward
            input_batch = input_batch.to(device)
            embeddings,logit,_ = net(input_batch)
            pred_bin_batch = torch.argmax(logit, dim=1, keepdim=True)
            

            if pred_bin_batch.sum() < 100:
                with open(save_name, "w") as f:
                    pass
                step += 1
                continue

            time_fp = time.time() - time_fp

            '''sklearn mean_shift'''
            time_clst = time.time()
            try:
                pred_insts = cluster_embed(embeddings, pred_bin_batch, band_width=0.5)
            except:
                with open(save_name, "w") as f:
                    pass
                step += 1
                continue
            time_clst = time.time() - time_clst
            
            '''Curve Fitting'''
            for idx in range(batch_size):
                time_fit = time.time()
                input_rgb = input_batch[idx]  # for each image in a batch
                pred_inst = pred_insts[idx]
                       
                xy_list = CULanefit(pred_inst, y_sample)
                time_fit = time.time() - time_fit

                '''Visualization'''
                if save or show :
                    rgb = (input_rgb.cpu().numpy().transpose(1, 2, 0) * 255 ).astype(np.uint8)
                   
                    for idx, inst in enumerate(xy_list):
                        # ndim: number of dim
                        if inst.ndim == 2:
                            index = np.nonzero(inst[:, 0] != -2)
                            inst = inst[index]
                            pts = inst.transpose((1, 0))
                            rgb=cv2.polylines(cv2.UMat(rgb), [inst.astype(np.int32)], False, (0, 0, 255), 2)
                    
                    if save:
                        cv2.imwrite(os.path.join(save_dir, f'{save_frame_name}rgb_fit.jpg'), rgb)

                    if show:
                        cv2.imshow('rgb_fit', rgb)
                        cv2.waitKey(0)
                        cv2.destroyAllWindows()
                
                lanes = generate_CULane(xy_list, save_name, (288,800))
                with open(save_name, "w") as f:
                    for l in lanes:
                        for (x, y) in l:
                            print("{} {}".format(x, y), end=" ", file=f)
                        print(file=f)

                time_run = time.time() - time_run
                time_run_avg = (time_ct * time_run_avg + time_run) / (time_ct + 1)
                time_fp_avg = (time_ct * time_fp_avg + time_fp) / (time_ct + 1)
                time_clst_avg = (time_ct * time_clst_avg + time_clst) / (time_ct + 1)
                time_fit_avg = (time_ct * time_fit_avg + time_fit) / (time_ct + 1)
                time_ct += 1

                if step % 50 == 0:  # Change the coefficient to filter the value
                    time_ct = 0
                    
                print('{}  Step:{}  Time:{:5.1f}  '
                      'time_run_avg:{:5.1f}  time_fp_avg:{:5.1f}  time_clst_avg:{:5.1f}  time_fit_avg:{:5.1f}  fps_avg:{:d}'
                      .format(test_start_time, step, time_run*1000,
                              time_run_avg*1000, time_fp_avg*1000 , time_clst_avg*1000, time_fit_avg*1000,
                              int(1/(time_run_avg + 1e-9))))

                step += 1

         
if __name__ == '__main__':

    # args = init_args()
    # test_culane(args.data_dir,args.ckpt_path,args.save_path,args.show,args.save,args.label)
    
    data_dir='D:\Luna\SYSU\Dataset\CULane'
    ckpt_path = 'D:\Luna\SYSU\code\Lane_Detection\ckpt\culane\ckpt_2021-01-21_10-27-47_step-47000.pth'
    label = '0915_23800_merge'
    save_path='output'
    show=True
    save = False 
    test_culane(data_dir,ckpt_path,save_path,show,save,label)