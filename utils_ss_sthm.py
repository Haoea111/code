import os
import sys

import numpy as np
import torch

curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)

from lib.metrics import mask_evaluation_np


def mask_loss(
    predicts,
    classify_predicts,
    labels,
    region_mask,
    bfc,
    ssl_loss,
    aux_label,
    data_type="nyc",
    ssl_loss_weight=1e-2,
):
    """
    Arguments:
        predicts {Tensor} -- predict，(batch_size, pre_len, W, H)
        classify_predicts {Tensor} -- classify_predicts，(batch_size, pre_len * W * H)
        labels {Tensor} -- label，(batch_size, pre_len, W, H)
        region_mask {np.array} -- mask matrix，(W, H)
        bfc {Tensor} -- bfc，(W * H, W/c * W/c)
        ssl_loss {Tensor} -- ST-SSL auxiliary loss
        data_type {str} -- nyc/chicago

    Returns:
        {Tensor} -- total loss
    """
    region_mask_list = []
    mse_list = []
    for i in range(4):
        mask = torch.from_numpy(region_mask[i]).to(predicts[i].device)
        mask /= mask.mean()
        region_mask_list.append(mask)
        mse_list.append(((predicts[i] - labels[i]) * mask) ** 2)

    predicts_f = predicts[0].reshape(predicts[0].shape[0], predicts[0].shape[1], 400)
    predicts_fc = torch.matmul(predicts_f, bfc).to()
    predicts_fc = predicts_fc.reshape(predicts_fc.shape[0], predicts_fc.shape[1], 10, 10).to(predicts[0].device)
    mse_fc = torch.mean(((predicts_fc - labels[1]) * region_mask_list[1]) ** 2)

    bce_list = []
    for i in range(4):
        classify_labels = labels[i].view(labels[i].shape[0], -1)
        bce_list.append(bce(classify_predicts[i], classify_labels))

    if data_type == "nyc":
        ratio_mask = torch.zeros(labels[0].shape).to(predicts[0].device)
        index_1 = labels[0] <= 0
        index_2 = (labels[0] > 0) & (labels[0] <= 0.04)
        index_3 = (labels[0] > 0.04) & (labels[0] <= 0.08)
        index_4 = labels[0] > 0.08

        ratio_mask[index_1] = 0.05
        ratio_mask[index_2] = 0.2
        ratio_mask[index_3] = 0.25
        ratio_mask[index_4] = 0.5

        mse_list[0] *= ratio_mask
    elif data_type == "chicago":
        ratio_mask = torch.zeros(labels[0].shape).to(predicts[0].device)
        index_1 = labels[0] <= 0
        index_2 = (labels[0] > 0) & (labels[0] <= 1 / 17)
        index_3 = (labels[0] > 1 / 17) & (labels[0] <= 2 / 17)
        index_4 = labels[0] > 2 / 17

        ratio_mask[index_1] = 0.05
        ratio_mask[index_2] = 0.2
        ratio_mask[index_3] = 0.25
        ratio_mask[index_4] = 0.5
        mse_list[0] *= ratio_mask

    if data_type == "nyc":
        ratio_mask = torch.zeros(aux_label.shape).to(predicts[0].device)
        index_1 = aux_label == 0
        index_2 = aux_label == 1
        index_3 = aux_label == 2

        ratio_mask[index_1] = 1
        ratio_mask[index_2] = 2
        ratio_mask[index_3] = 1

        mse_list[0] *= ratio_mask
    elif data_type == "chicago":
        ratio_mask = torch.zeros(aux_label.shape).to(predicts[0].device)
        index_1 = aux_label == 0
        index_2 = aux_label == 1
        index_3 = aux_label == 2

        ratio_mask[index_1] = 1
        ratio_mask[index_2] = 1.5
        ratio_mask[index_3] = 1

        mse_list[0] *= ratio_mask

    if data_type == "nyc":
        ssl_loss_sum = ssl_loss_weight * ssl_loss
        base_loss = (
            torch.mean(mse_list[0])
            + torch.mean(mse_list[1])
            + torch.mean(mse_list[2])
            + torch.mean(mse_list[3])
            + 3e-4 * bce_list[0]
            + 3e-4 * bce_list[1]
            + 1e-5 * bce_list[2]
            + 1e-5 * bce_list[3]
            + mse_fc
        )
        print("ssl_loss_sum", ssl_loss_sum.item(), "base_loss", base_loss.item())
        return base_loss + ssl_loss_sum

    if data_type == "chicago":
        ssl_loss_sum = ssl_loss_weight * ssl_loss
        base_loss = (
            torch.mean(mse_list[0])
            + 3e-4 * torch.mean(mse_list[1])
            + 1e-4 * torch.mean(mse_list[2])
            + 3e-5 * torch.mean(mse_list[3])
            + 1e-3 * bce_list[0]
            + 1e-3 * bce_list[1]
            + 1e-5 * bce_list[2]
            + 1e-5 * bce_list[3]
            + 3e-4 * mse_fc
        )
        print("ssl_loss_sum", ssl_loss_sum.item(), "base_loss", base_loss.item())
        return base_loss + ssl_loss_sum

    raise ValueError(f"Unsupported data_type: {data_type}")


def bce(y_pred, y_true, padded_value_indicator=0):
    device = y_pred.device

    y_pred = y_pred.clone()
    y_true = y_true.clone()

    mask = y_true == padded_value_indicator
    valid_mask = y_true != padded_value_indicator

    y_pred = torch.clamp(y_pred, min=0.0, max=1.0)

    y_true = torch.where(y_true != 0, torch.tensor(1.0).to(device), y_true)
    ls = torch.nn.BCELoss(reduction="none")(y_pred, y_true)

    ls[mask] = 0.0

    document_loss = torch.sum(ls, dim=-1)
    sum_valid = torch.sum(valid_mask, dim=-1).type(torch.float32) > torch.tensor(
        0.0, dtype=torch.float32, device=device
    )
    loss_output = torch.sum(document_loss) / torch.sum(sum_valid)

    return loss_output


@torch.no_grad()
def compute_loss(
    net,
    dataloader,
    risk_mask,
    road_adj,
    risk_adj,
    poi_adj,
    grid_node_map,
    trans,
    device,
    bfc,
    data_type="nyc",
    ssl_loss_weight=1e-2,
):
    net.eval()
    temp = []
    for batch_1, batch_2, batch_3, batch_4 in zip(dataloader[0], dataloader[1], dataloader[2], dataloader[3]):
        batch = [batch_1, batch_2, batch_3, batch_4]
        feature = []
        target_time = []
        graph_feature = []
        label = []
        aux_label = None
        for i in range(4):
            t_feature, t_target_time, t_graph_feature, t_label, t_aux_label = batch[i]
            t_feature = t_feature.to(device)
            t_target_time = t_target_time.to(device)
            t_graph_feature = t_graph_feature.to(device)
            t_label = t_label.to(device)
            t_aux_label = t_aux_label.to(device)
            if i == 0:
                aux_label = t_aux_label
            feature.append(t_feature)
            target_time.append(t_target_time)
            graph_feature.append(t_graph_feature)
            label.append(t_label)

        final_output, classification_output, ssl_loss = net(
            feature, target_time, graph_feature, road_adj, risk_adj, poi_adj, grid_node_map, trans
        )
        loss = mask_loss(
            final_output,
            classification_output,
            label,
            risk_mask,
            bfc,
            ssl_loss,
            aux_label,
            data_type,
            ssl_loss_weight=ssl_loss_weight,
        )
        temp.append(loss.cpu().item())
    return sum(temp) / len(temp)


@torch.no_grad()
def predict_and_evaluate(net, dataloader, risk_mask, road_adj, risk_adj, poi_adj, grid_node_map, trans, scaler, device):
    net.eval()
    prediction_list = []
    label_list = []
    for batch_1, batch_2, batch_3, batch_4 in zip(dataloader[0], dataloader[1], dataloader[2], dataloader[3]):
        batch = [batch_1, batch_2, batch_3, batch_4]
        feature = []
        target_time = []
        graph_feature = []
        label = []
        for i in range(4):
            t_feature, t_target_time, t_graph_feature, t_label, t_aux_label = batch[i]
            t_feature = t_feature.to(device)
            t_target_time = t_target_time.to(device)
            t_graph_feature = t_graph_feature.to(device)
            t_label = t_label.to(device)
            feature.append(t_feature)
            target_time.append(t_target_time)
            graph_feature.append(t_graph_feature)
            label.append(t_label)

        final_output, classification_output, ssl_loss = net(
            feature, target_time, graph_feature, road_adj, risk_adj, poi_adj, grid_node_map, trans
        )
        prediction_list.append(final_output[0].cpu().numpy())
        label_list.append(label[0].cpu().numpy())
    prediction = np.concatenate(prediction_list, 0)
    label = np.concatenate(label_list, 0)

    inverse_trans_pre = scaler[0].inverse_transform(prediction)
    inverse_trans_label = scaler[0].inverse_transform(label)

    rmse_, recall_, map_ = mask_evaluation_np(inverse_trans_label, inverse_trans_pre, risk_mask[0], 0)
    return rmse_, recall_, map_, inverse_trans_pre, inverse_trans_label
