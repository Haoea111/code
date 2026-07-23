import os
import pickle as pkl
import sys

import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset, DataLoader, TensorDataset

curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)
from lib.utils import Scaler_NYC, Scaler_Chi

# high frequency time
high_fre_hour = [6, 7, 8, 15, 16, 17, 18]


def split_and_norm_data(all_data, train_rate=0.6, valid_rate=0.2, recent_prior=3, week_prior=4, one_day_period=24,
                        days_of_week=7, pre_len=1):
    num_of_time, channel, _, _ = all_data.shape
    train_line, valid_line = int(num_of_time * train_rate), int(num_of_time * (train_rate + valid_rate))
    for index, (start, end) in enumerate(((0, train_line), (train_line, valid_line), (valid_line, num_of_time))):
        if index == 0:
            if channel == 48:  # NYC
                scaler = Scaler_NYC(all_data[start:end, :, :, :])
            if channel == 41:  # Chicago
                scaler = Scaler_Chi(all_data[start:end, :, :, :])
        norm_data = scaler.transform(all_data[start:end, :, :, :])
        X, Y = [], []
        high_X, high_Y = [], []
        for i in range(len(norm_data) - week_prior * days_of_week * one_day_period - pre_len + 1):
            t = i + week_prior * days_of_week * one_day_period
            label = norm_data[t:t + pre_len, 0, :, :]
            period_list = []
            for week in range(week_prior):
                period_list.append(i + week * days_of_week * one_day_period)
            for recent in list(range(1, recent_prior + 1))[::-1]:
                period_list.append(t - recent)
            feature = norm_data[period_list, :, :, :]
            X.append(feature)
            Y.append(label)
            # NYC/Chicago hour_of_day feature index is [1:25]
            if list(norm_data[t, 1:25, 0, 0]).index(1) in high_fre_hour:
                high_X.append(feature)
                high_Y.append(label)
        yield np.array(X), np.array(Y), np.array(high_X), np.array(high_Y), scaler

def generate_aux(rx, wx, Y):
    project_num = 0
    data_rx=np.array(rx)
    data_wx=np.array(wx)
    if data_rx.shape[2]==48:
        data=np.concatenate((data_rx[:,:,:,:,:],data_wx[:,:,:,:,:]),axis=1)
        project_num = 14
        print("generate_aux NYC")
    else:
        data=np.concatenate((data_rx[:,:,:,:,:],data_wx[:,:,:,:,:]),axis=1)
        project_num = 10
        print("generate_aux Chicago")
    batch_size, T, D, W, H = data.shape
    data = np.transpose(data.reshape(batch_size,-1,W,H), (0, 2 ,3 ,1))
    T, W, H, D = data.shape
    label=np.array(Y).reshape(batch_size,W,H)
    to_int = [2**i for i in range(project_num)]
    
    random_projection = np.random.normal(size=(D, project_num))
    for a in range(W):
        for b in range(H):
            bucket = np.array([0 for i in range(2**project_num)])
            label[label[:,a,b]>0,a,b]=2
            if len(label[label[:,a,b]>0]) == 0:
                continue
            
            hash = (np.matmul(np.squeeze(data[np.nonzero(label[:,a,b]), a, b, :],axis=0), random_projection) >= 0).astype(int)
            bucket_num = np.matmul(hash, to_int).tolist()
            bucket[bucket_num]=1
            if len(label[label[:,a,b]>0]) == 0:
                continue
            cur_data = data[:, a, b, :]  
            hash = (np.matmul(cur_data, random_projection) >= 0).astype(int)
            bucket_num = np.matmul(hash, to_int)

            pos_2 = (label[:,a,b] == 2)
            label[np.nonzero(bucket[bucket_num]),a,b] = 1
            label[pos_2,a,b] = 2

    return np.expand_dims(label, axis=1)


def split_and_norm_data_time(all_data, train_rate=0.6, valid_rate=0.2, recent_prior=3, week_prior=4, one_day_period=24,
                             days_of_week=7, pre_len=1):
    num_of_time, channel, data_w, data_h = all_data.shape
    start_rate = 0
    start_line = int(num_of_time * start_rate)
    train_line, valid_line = int(num_of_time * train_rate), int(num_of_time * (train_rate + valid_rate))
    scal_line = int(train_line)
    print(scal_line,train_line,valid_line)
    for index, (start, end) in enumerate(((start_line, scal_line), (train_line, valid_line), (valid_line, num_of_time))):
        if index == 0:
            if channel == 48:
                scaler = Scaler_NYC(all_data[start:end, :, :, :])
            if channel == 41:
                scaler = Scaler_Chi(all_data[start:end, :, :, :])
        norm_data = scaler.transform(all_data[start:end, :, :, :])
        wX, rX, Y, target_time = [], [], [],[]
        high_wX, high_rX, high_Y, high_target_time = [], [], [], []
        for i in range(len(norm_data) - week_prior * days_of_week * one_day_period - pre_len + 1):
            t = i + week_prior * days_of_week * one_day_period
            label = norm_data[t:t + pre_len, 0, :, :]
            recent_list = []
            week_list=[]
            for week in range(week_prior):
                week_list.append(i + week * days_of_week * one_day_period)
            for recent in list(range(1, recent_prior + 1))[::-1]:
                recent_list.append(t - recent)
            r_feature = norm_data[recent_list, :, :, :]
            rX.append(r_feature)
            w_feature = norm_data[week_list, :, :, :]
            wX.append(w_feature)
            Y.append(label)
            target_time.append(norm_data[t, 1:33, 0, 0])
            if list(norm_data[t, 1:25, 0, 0]).index(1) in high_fre_hour:
                high_wX.append(w_feature)
                high_rX.append(r_feature)
                high_Y.append(label)
                high_target_time.append(norm_data[t, 1:33, 0, 0])
        
        if data_h==20:
            aux_Y = generate_aux(rX, wX, Y)
        else:
            aux_Y=Y

        yield np.array(rX),np.array(wX), np.array(Y), np.array(aux_Y), np.array(target_time), np.array(high_rX), np.array(high_wX), np.array(high_Y), np.array(
            high_target_time), scaler


def normal_and_generate_dataset_time(all_data_filename, train_rate=0.6, valid_rate=0.2, recent_prior=3, week_prior=4,
                                     one_day_period=24, days_of_week=7, pre_len=1):
    all_data = pkl.load(open(all_data_filename, 'rb')).astype(np.float32)

    for i in split_and_norm_data_time(all_data,
                                      train_rate=train_rate,
                                      valid_rate=valid_rate,
                                      recent_prior=recent_prior,
                                      week_prior=week_prior,
                                      one_day_period=one_day_period,
                                      days_of_week=days_of_week,
                                      pre_len=pre_len):
        yield i


def get_mask(mask_path):
    """
    Arguments:
        mask_path {str} -- mask filename
    
    Returns:
        {np.array} -- mask matrix，shape(W,H)
    """
    mask = pkl.load(open(mask_path, 'rb')).astype(np.float32)
    return mask


def get_adjacent(adjacent_path):
    """
    Arguments:
        adjacent_path {str} -- adjacent matrix path
    
    Returns:
        {np.array} -- shape:(N,N)
    """
    adjacent = pkl.load(open(adjacent_path, 'rb')).astype(np.float32)
    return adjacent


def get_grid_node_map_maxtrix(grid_node_path):
    """
    Arguments:
        grid_node_path {str} -- filename
    
    Returns:
        {np.array} -- shape:(W*H,N)
    """
    grid_node_map = pkl.load(open(grid_node_path, 'rb')).astype(np.float32)
    return grid_node_map


def get_trans(trans_path):
    """
    Arguments:
        trans_path {str} -- filename

    Returns:
        {np.array} -- shape:(N_f,N_c)
    """
    trans = pkl.load(open(trans_path, 'rb')).astype(np.float32)
    return trans

def generate_dataloader(
        all_data_filename,
        grid_node_data_filename,
        batch_size,
        train_rate=0.6,
        valid_rate=0.2,
        recent_prior=3,
        week_prior=4,
        one_day_period=24,
        days_of_week=7,
        pre_len=1,
        test=False,
        north_south_map=20,
        west_east_map=20):
    loaders = []
    high_test_loader = []
    scaler_r = ""
    train_data_shape_r = ""
    time_shape_r = ""
    graph_feature_shape_r = ""
    for idx, (rx, wx, y, aux_y, target_times, high_rx, high_wx, high_y, high_target_times, scaler) in enumerate(
            normal_and_generate_dataset_time(
                all_data_filename,
                train_rate=train_rate,
                valid_rate=valid_rate,
                recent_prior=recent_prior,
                week_prior=week_prior,
                one_day_period=one_day_period,
                days_of_week=days_of_week,
                pre_len=pre_len)):

        grid_node_map = get_grid_node_map_maxtrix(grid_node_data_filename)
        if 'nyc' in all_data_filename:
            graph_rx = rx[:, :, [0, 46, 47], :, :].reshape((rx.shape[0], rx.shape[1], -1, north_south_map * west_east_map))
            graph_wx = wx[:, :, [0, 46, 47], :, :].reshape((wx.shape[0], wx.shape[1], -1, north_south_map * west_east_map))
            high_graph_rx = high_rx[:, :, [0, 46, 47], :, :].reshape(
                (high_rx.shape[0], high_rx.shape[1], -1, north_south_map * west_east_map))
            high_graph_wx = high_wx[:, :, [0, 46, 47], :, :].reshape(
                (high_wx.shape[0], high_wx.shape[1], -1, north_south_map * west_east_map))
            graph_rx = np.dot(graph_rx, grid_node_map)
            graph_wx = np.dot(graph_wx, grid_node_map)
            high_graph_rx = np.dot(high_graph_rx, grid_node_map)
            high_graph_wx = np.dot(high_graph_wx, grid_node_map)
        if 'chicago' in all_data_filename:
            graph_rx = rx[:, :, [0, 39, 40], :, :].reshape((rx.shape[0], rx.shape[1], -1, north_south_map * west_east_map))
            graph_wx = wx[:, :, [0, 39, 40], :, :].reshape((wx.shape[0], wx.shape[1], -1, north_south_map * west_east_map))
            high_graph_rx = high_rx[:, :, [0, 39, 40], :, :].reshape(
                (high_rx.shape[0], high_rx.shape[1], -1, north_south_map * west_east_map))
            high_graph_wx = high_wx[:, :, [0, 39, 40], :, :].reshape(
                (high_wx.shape[0], high_wx.shape[1], -1, north_south_map * west_east_map))
            graph_rx = np.dot(graph_rx, grid_node_map)
            graph_wx = np.dot(graph_wx, grid_node_map)
            high_graph_rx = np.dot(high_graph_rx, grid_node_map)
            high_graph_wx = np.dot(high_graph_wx, grid_node_map)

        print("rx feature:", str(graph_rx.shape), "wx feature:", str(graph_wx.shape), "label:", str(y.shape), "time:", str(target_times.shape),
              "high rx feature:", str(high_rx.shape), "high wx feature:", str(high_wx.shape), "high label:", str(high_y.shape))

        if idx == 0:  # train
            scaler_r = scaler
            train_data_shape_r = rx.shape
            time_shape_r = target_times.shape
            graph_feature_shape_r = graph_rx.shape

        loaders.append(DataLoader(
            TensorDataset(
                torch.from_numpy(graph_rx),
                torch.from_numpy(graph_wx),
                torch.from_numpy(target_times),
                torch.from_numpy(y),
                torch.from_numpy(aux_y)
            ),
            batch_size=batch_size,
            shuffle=(idx == 0)
        ))

        if idx == 2:  # test
            high_test_loader = DataLoader(
                TensorDataset(
                    torch.from_numpy(high_graph_rx),
                    torch.from_numpy(high_graph_wx),
                    torch.from_numpy(high_target_times),
                    torch.from_numpy(high_y),
                    torch.from_numpy(high_y)#no use
                ),
                batch_size=batch_size,
                shuffle=(idx == 0)
            )
    return scaler_r, train_data_shape_r, time_shape_r, graph_feature_shape_r, loaders, high_test_loader
