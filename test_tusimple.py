import os
import time
import argparse
import ujson as json
import numpy as np
import cv2

import torch 
from torch.utils.data import DataLoader
import torch.nn as nn

from tools.utils import cluster_embed, get_color,generate_tusimple_json,fit_tusimple
from tools.dataset import TuSimpleDataset
from model.model import Resnest_LaneNet

def init_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, help='path to tusimple dataset')
    parser.add_argument('--ckpt_path', type=str, help='path to parameter file (.pth)')
    parser.add_argument('--save_path', type=str, default='output', help='path to save dir')
    parser.add_argument('--show', action='store_true', help='whether to show visualization images')
    parser.add_argument('--save_img', action='store_true', help='whether to save visualization images')
    parser.add_argument('--label', type=str, help='label to denote details of experiments')


    return parser.parse_args()


def test_tusimple(data_dir,ckpt_path,save_path,show,save,label):

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

    test_set = TuSimpleDataset(data_dir, 'test')

    num_test = len(test_set)
    testset_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    print('Finish loading data from %s' % data_dir)

    '''Constant variables'''
    _, h, w = test_set[0]['input_tensor'].shape
       
    # y_start, y_stop and y_num is calculated according to TuSimple Benchmark's setting
    y_start = np.round(160*h/ 720.)
    y_stop = np.round(710*h/720. )
    y_num = 56
    y_sample = np.linspace(y_start, y_stop, y_num, dtype=np.int16)

    

    '''Forward propogation'''
    with torch.no_grad():
        
        net = Resnest_LaneNet()
        net = nn.DataParallel(net)
        net.to(device)
        net.eval()

        assert ckpt_path is not None, 'Checkpoint Error.'

        checkpoint = torch.load(ckpt_path)
        net.load_state_dict(checkpoint['model_state_dict'], strict=True)
   
        time_run_avg = 0
        time_fp_avg = 0
        time_clst_avg = 0
        time_fit_avg = 0
        time_ct = 0
        jsonlist = list()
        
        for batch_idx, batch in enumerate(testset_loader):
            time_run = time.time()
            time_fp = time.time()

            '''load dataset'''
            input_batch = batch['input_tensor']
            raw_file_batch = batch['raw_file']
            path_batch = batch['path']

            input_batch = input_batch.to(device)
            # forward
            embeddings,logit,_ = net(input_batch)

            pred_bin_batch = torch.argmax(logit, dim=1, keepdim=True)
            preds_bin_expand_batch = pred_bin_batch.view(pred_bin_batch.shape[0] * pred_bin_batch.shape[1] * pred_bin_batch.shape[2] * pred_bin_batch.shape[3])

            time_fp = time.time() - time_fp

            '''sklearn mean_shift'''
            time_clst = time.time()
            try:
                pred_insts = cluster_embed(embeddings, pred_bin_batch, band_width=0.5)
            except:
                break
            time_clst = time.time() - time_clst
            
            '''Curve Fitting'''
            for idx in range(batch_size):
                time_fit = time.time()
                input_rgb = input_batch[idx]  # for each image in a batch
                raw_file = raw_file_batch[idx]
                pred_inst = pred_insts[idx]
                path = path_batch[idx]
                      
                xy_list = fit_tusimple(pred_inst, y_sample)
                time_fit = time.time() - time_fit

                '''Visualization'''
                if save or show :
                    rgb = (input_rgb.cpu().numpy().transpose(1, 2, 0) * 255 ).astype(np.uint8)
                    pred_bin_rgb = pred_bin_batch[idx].repeat(3,1,1).cpu().numpy().transpose(1, 2, 0).astype(np.uint8) * 255
                    pred_inst_rgb = pred_inst.repeat(3, 1, 1).cpu().numpy().astype(np.uint8).transpose(1, 2, 0)  # color
                    
                    for i in np.unique(pred_inst_rgb):
                        if i == 0:
                            continue
                        index = np.where(pred_inst_rgb[:, :, 0] == i)
                        pred_inst_rgb[index] = get_color(i)

                    fg_mask = (pred_bin_rgb[:, :, 0] == 255).astype(np.uint8)
                    bg_mask = (pred_bin_rgb[:, :, 0] == 0).astype(np.uint8)

                    rgb_bg = cv2.bitwise_and(rgb, rgb, mask=bg_mask)
                    rgb_fg = cv2.bitwise_and(rgb, rgb, mask=fg_mask)
                    pred_inst_rgb_fg = cv2.bitwise_and(pred_inst_rgb, pred_inst_rgb, mask=fg_mask)
                    fg_align = cv2.addWeighted(rgb_fg, 0.3, pred_inst_rgb_fg, 0.7, 0)
                    rgb_align = rgb_bg + fg_align

                    for idx, inst in enumerate(xy_list):
                        if inst.ndim == 2:
                            index = np.nonzero(inst[:, 0] != -2)
                            inst = inst[index]
                            pts = inst.transpose((1, 0))
                            rgb=cv2.polylines(cv2.UMat(rgb), [inst.astype(np.int32)], False, (0, 0, 255), 2)

                    if save:
                        clip, seq, frame = path.split('/')
                        output_seq_dir = os.path.join(output_dir, seq)
                        if os.path.exists(output_seq_dir) is False:
                            os.makedirs(output_seq_dir, exist_ok=True)
                        frame = frame.split('.')[0]
                        cv2.imwrite(os.path.join(output_seq_dir, f'{frame}_align.jpg'), rgb_align)
                        cv2.imwrite(os.path.join(output_seq_dir, f'{frame}_fit.jpg'), rgb)

                    if show:
                        #显示图片
                        cv2.imshow('align', rgb_align)
                        cv2.imshow('rgb_fit', rgb)
                        cv2.waitKey(0)
                        cv2.destroyAllWindows()

                outjson=generate_tusimple_json(xy_list,y_sample, raw_file, (288,512), time_clst)
                jsonlist.append(outjson)

                time_run = time.time() - time_run

                time_run_avg = (time_ct * time_run_avg + time_run) / (time_ct + 1)
                time_fp_avg = (time_ct * time_fp_avg + time_fp) / (time_ct + 1)
                time_clst_avg = (time_ct * time_clst_avg + time_clst) / (time_ct + 1)
                time_fit_avg = (time_ct * time_fit_avg + time_fit) / (time_ct + 1)
                time_ct += 1

                if batch_idx % 50 == 0:  # Change the coefficient to filter the value
                    time_ct = 0


                print('{}  Step:{}  Time:{:5.1f}  '
                      'time_run_avg:{:5.1f}  time_fp_avg:{:5.1f}  time_clst_avg:{:5.1f}  time_fit_avg:{:5.1f}  fps_avg:{:d}'
                      .format(test_start_time, batch_idx, time_run*1000,
                              time_run_avg*1000, time_fp_avg*1000 , time_clst_avg*1000, time_fit_avg*1000,
                              int(1/(time_run_avg + 1e-9))))
                
                batch_idx += 1

        with open(f'{output_dir}/test_pred-{label}.json', 'w') as f:
            for item in jsonlist:
                json.dump(item, f)  # , indent=4, sort_keys=True
                f.write('\n')
        
            
if __name__ == '__main__':

    args = init_args()
    test_tusimple(args.data_dir,args.ckpt_path,args.save_path,args.show,args.save,args.label)
