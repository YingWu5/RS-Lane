#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/5/10 19:07
# @Author  : Shenhan Qian
# @Site    : 
# @File    : utils.py
# @Software: PyCharm
import os
import numpy as np
import cv2 as cv2
import torch
from sklearn.cluster import MeanShift
from scipy.interpolate import interp1d

def fit_tusimple(inst_pred,y_sample):
    h, w = inst_pred.shape
    inst_pred_expand = inst_pred.view(-1)
    inst_unique = torch.unique(inst_pred_expand)

    lanes = []
    curves_pts = []
    for inst_idx in inst_unique:
        if inst_idx != 0:
            lanes.append(torch.nonzero(inst_pred == inst_idx).cpu().numpy())
    
    for lane in lanes:
        x_sample = []
        for i in range(len(y_sample)):
            index = np.where(lane[:,0]==y_sample[i])
            xlist = lane[:,1][index]
            if len(xlist) == 0:
                x = -2
            else:
                x = xlist.mean()

            if np.isnan(x):
                x = -2
            x_sample.append(x)
        xy_sample = np.vstack((x_sample, y_sample)).transpose((1, 0)).astype(np.int32)

        curves_pts.append(xy_sample)

    return curves_pts

def CULanefit(inst_pred,y_sample):
    h, w = inst_pred.shape
    inst_pred_expand = inst_pred.view(-1)
    inst_unique = torch.unique(inst_pred_expand)

    lanes = []
    curves_pts = []
    for inst_idx in inst_unique:
        if inst_idx != 0:
            lanes.append(torch.nonzero(inst_pred == inst_idx).cpu().numpy())
    
    for lane in lanes:
        x_sample = []
        y_list = []
        for i in range(len(y_sample)):
            y = (y_sample[i]* 288./590.).astype(np.int32)
            index = np.where(lane[:,0]==y)
            xlist = lane[:,1][index]
            if len(xlist) == 0:
                x = -2
            else:
                x = xlist.mean()

            if np.isnan(x):
                x = -2
            x_sample.append(x)
            y_list.append(y)
        xy_sample = np.vstack((x_sample, y_list)).transpose((1, 0)).astype(np.int32)

        curves_pts.append(xy_sample)
    
    lanes = []
    for curve in curves_pts:
        index = np.where(curve[:, 0] > 0)
        curve = curve[index]
        if curve.shape[0]<5:
            # lanes.append(curve)
            continue
        x = curve[:, 0] 
        y = curve[:, 1]
        fit = np.polyfit(y, x, 1)
        if fit[0]<-7:
            continue
        fy = np.poly1d(fit)
        y_min = np.min(y)
        y_list = (y_sample* 288./591.).astype(np.int32)
        y = y_list[np.where(y_list>=y_min)]
        x = fy(y)
        index = np.where(x<800)
        x = x[index]
        y = y[index]
        index = np.where(x>=0)
        x = x[index]
        y = y[index]
        xy = np.vstack((x,y)).transpose((1, 0)).astype(np.int32)
        lanes.append(xy)

    return lanes
    # return curves_pts

def generate_tusimple_json(curves_pts_pred, y_sample, raw_file, size, run_time):
    h, w = size

    lanes = []
    for curve in curves_pts_pred:
        index = np.where(curve[:, 0] > 0)
        curve[index, 0] = curve[index, 0] * 720. / h
        # curve[index, 0] = (curve[index, 0] * 0.7 /h +0.3) *720

        x_list = np.round(curve[:, 0]).astype(np.int32).tolist()
        lanes.append(x_list)

    entry_dict = dict()

    entry_dict['lanes'] = lanes
    entry_dict['h_sample'] = np.round(y_sample * 720. / h).astype(np.int32).tolist()
    entry_dict['run_time'] = int(np.round(run_time * 1000))
    entry_dict['raw_file'] = raw_file

    return entry_dict

def generate_CULane(curves_pts_pred, save_name, size):
    h, w = size

    lanes = []
    for curve in curves_pts_pred:
        index = np.where(curve[:, 0] > 0)
        curve = curve[index]
        curve[:, 0] = curve[:, 0] * 1640. / w
        curve[:, 1] = curve[:, 1] * 590. / h
        curve = np.round(curve).astype(np.int32)
        lanes.append(curve)
    return lanes

def cluster_embed(embeddings, preds_bin, band_width):
    c = embeddings.shape[1]
    n, _, h, w = preds_bin.shape
    preds_bin = preds_bin.view(n, h, w)
    preds_inst = torch.zeros_like(preds_bin)
    for idx, (embedding, bin_pred) in enumerate(zip(embeddings, preds_bin)):
       
        embedding_fg = torch.transpose(torch.masked_select(embedding, bin_pred.bool()).view(c, -1), 0, 1)
        clustering = MeanShift(bandwidth=band_width, bin_seeding=True, min_bin_freq=200,cluster_all=False).fit(embedding_fg.cpu().detach().numpy())
        labels= clustering.labels_.astype(np.int64)
        preds_inst[idx][bin_pred.bool()] = torch.from_numpy(labels).cuda() + 1

    return preds_inst


color_set = [(0, 0, 0),
    (60, 76, 231), (113, 204, 46), (219, 152, 52), (182, 89, 155),(156, 188, 26), 
    (173, 68, 142), (141, 140, 127), (43, 57, 192), (34, 126, 230), (96, 174, 39),
     (18, 156, 243), (94, 73, 52),(0, 84, 211), (15, 196, 241), (185, 128, 41),
    (241, 240, 236), (166, 165, 149), (199, 195, 189), (80, 62, 44), (133, 160, 22),
]

def split_path(path):
    """split path tree into list"""
    folders = []
    while True:
        path, folder = os.path.split(path)
        if folder != "":
            folders.insert(0, folder)
        else:
            if path != "":
                folders.insert(0, path)
            break
    return folders

def get_color(idx):
    return color_set[idx]


if __name__ == '__main__':
    pass






