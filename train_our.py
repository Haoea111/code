import os
import torch
import torch.optim as optim

import numpy as np
import json
from time import time
import argparse
import random

import sys

if not hasattr(np, "Inf"):
    np.Inf = np.inf

from lib.dataloader import get_mask, get_adjacent, get_grid_node_map_maxtrix, \
    get_trans, generate_dataloader
from lib.early_stop import EarlyStopping
from model.our import STTAR
from lib.utils_ss_sthm import mask_loss, compute_loss, predict_and_evaluate

curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)

parser = argparse.ArgumentParser()
parser.add_argument(
    "--config",
    type=str,
    help="configuration file",
    default="config/nyc/NYC_Config.json",
)
parser.add_argument("--gpus", type=str, help="test program", default="0")
parser.add_argument("--test", action="store_true", help="test program")
parser.add_argument("--infer", action="store_true", help="load model from file and infer")
parser.add_argument("--export-npz", type=str, default="", help="save test predictions to npz for visualization")
parser.add_argument(
    "--run-tag",
    type=str,
    default="",
    help="checkpoint filename tag (e.g. wo_cvdr -> pth/nyc/nyc_wo_cvdr_full_model.pth)",
)
parser.add_argument(
    "--checkpoint",
    type=str,
    default="",
    help="full checkpoint path for save/load (overrides --run-tag and default)",
)

args = parser.parse_args()
if args.gpus and args.gpus.strip().lower() != "cpu":
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus.strip()

MODEL_TAG = "sttar_tapm_ss_sthm"


def resolve_checkpoint_path(data_type, config):
    if args.checkpoint:
        path = args.checkpoint
    else:
        tag = args.run_tag or config.get("run_tag", "") or MODEL_TAG
        city_dir = os.path.join("pth", data_type)
        path = os.path.join(city_dir, f"{data_type}_{tag}_full_model.pth")
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    return path


config_filename = args.config
with open(config_filename, 'r') as f:
    config = json.loads(f.read())
print(json.dumps(config, sort_keys=True, indent=4))

CKPT_PATH = resolve_checkpoint_path(config["data_type"], config)
print("Checkpoint path:", CKPT_PATH, flush=True)


def resolve_device(gpus):
    if gpus and gpus.strip().lower() == "cpu":
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    return torch.device("cpu")


device = resolve_device(args.gpus)
print("Using device:", device, flush=True)


def evaluate_test_sets(net, test_loader, high_test_loader, risk_mask, road_adj, risk_adj, poi_adj,
                       grid_node_map, trans, scaler, device):
    test_rmse, test_recall, test_map, test_inverse_trans_pre, test_inverse_trans_label = \
        predict_and_evaluate(net, test_loader, risk_mask, road_adj, risk_adj, poi_adj,
                             grid_node_map, trans, scaler, device)

    high_test_rmse, high_test_recall, high_test_map, _, _ = \
        predict_and_evaluate(net, high_test_loader, risk_mask, road_adj, risk_adj, poi_adj,
                             grid_node_map, trans, scaler, device)

    return {
        "test_rmse": test_rmse,
        "test_recall": test_recall,
        "test_map": test_map,
        "high_test_rmse": high_test_rmse,
        "high_test_recall": high_test_recall,
        "high_test_map": high_test_map,
        "test_inverse_trans_pre": test_inverse_trans_pre,
        "test_inverse_trans_label": test_inverse_trans_label,
    }

if config['seed'] is not None:
    seed = config['seed']
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed)
    random.seed(seed)


def loadConfig(config):
    patience = config['patience']
    delta = config['delta']

    train_rate = config['train_rate']
    valid_rate = config['valid_rate']
    recent_prior = config['recent_prior']
    week_prior = config['week_prior']
    one_day_period = config['one_day_period']
    days_of_week = config['days_of_week']
    pre_len = config['pre_len']
    seq_len = recent_prior + week_prior
    training_epoch = config['training_epoch']
    batch_size = config['batch_size']
    learning_rate = config['learning_rate']
    weight_dacay = config['weight_dacay']
    transformer_hidden_size = config['transformer_hidden_size']
    gcn_num_filter = config['gcn_num_filter']
    num_of_heads = config['num_of_heads']

    all_data_filename = []
    mask_filename = []
    road_adj_filename = []
    risk_adj_filename = []
    poi_adj_filename = []
    grid_node_filename = []
    north_south_map = []
    west_east_map = []

    trans20_10_filename = config['trans20_10_filename']
    trans10_5_filename = config['trans10_5_filename']
    trans5_2_filename = config['trans5_2_filename']

    trans = [trans20_10_filename, trans10_5_filename, trans5_2_filename]
    for i in range(4):
        all_data_filename.append(config['all_data_filename_{}'.format(i + 1)])
        mask_filename.append(config['mask_filename_{}'.format(i + 1)])
        road_adj_filename.append(config['road_adj_filename_{}'.format(i + 1)])
        risk_adj_filename.append(config['risk_adj_filename_{}'.format(i + 1)])
        poi_adj_filename.append(config['poi_adj_filename_{}'.format(i + 1)])
        grid_node_filename.append(config['grid_node_filename_{}'.format(i + 1)])
        north_south_map.append(config['north_south_map_{}'.format(i + 1)])
        west_east_map.append(config['west_east_map_{}'.format(i + 1)])

    bfc_20_10_filename = config['bfc_20_10_filename']

    return (patience, delta, train_rate, valid_rate, recent_prior, week_prior, one_day_period, days_of_week, pre_len,
            seq_len, training_epoch, batch_size, learning_rate, weight_dacay, transformer_hidden_size,
            gcn_num_filter,
            all_data_filename, mask_filename, road_adj_filename, risk_adj_filename, poi_adj_filename, grid_node_filename,
            north_south_map, west_east_map, trans, bfc_20_10_filename, num_of_heads)


def training(net,
             training_epoch,
             train_loader,
             val_loader,
             test_loader,
             high_test_loader,
             road_adj,
             risk_adj,
             poi_adj,
             risk_mask,
             grid_node_map,
             trainer,
             early_stop,
             device,
             scaler,
             trans,
             bfc,
             data_type='nyc',
             ssl_loss_weight=1e-2,
             ):

    global_step = 1
    best_epoch = None
    for epoch in range(1, training_epoch + 1):
        net.train()
        batch_num = 1
        train_loss = []
        for batch_1, batch_2, batch_3, batch_4 in zip(train_loader[0], train_loader[1], train_loader[2], train_loader[3]):
            batch = [batch_1, batch_2, batch_3, batch_4]
            r_train_feature = []
            w_train_feature = []
            target_time = []
            train_label = []
            aux_label = None
            for i in range(4):
                rt_graph_feature, wt_graph_feature, t_target_time, t_train_label, t_aux_label = batch[i]
                rt_graph_feature, wt_graph_feature, t_target_time, t_train_label, t_aux_label = rt_graph_feature.to(device), wt_graph_feature.to(device), \
                    t_target_time.to(device), t_train_label.to(device), t_aux_label.to(device)
                if i == 0:
                    aux_label = t_aux_label
                r_train_feature.append(rt_graph_feature)
                w_train_feature.append(wt_graph_feature)
                target_time.append(t_target_time)
                train_label.append(t_train_label)

            start_time = time()
            final_output, classification_output, ssl_loss = net(
                r_train_feature, w_train_feature, target_time, road_adj, risk_adj, poi_adj, grid_node_map, trans
            )
            l = mask_loss(
                final_output,
                classification_output,
                train_label,
                risk_mask,
                bfc,
                ssl_loss,
                aux_label,
                data_type,
                ssl_loss_weight=ssl_loss_weight,
            )
            trainer.zero_grad()
            l.backward()
            trainer.step()
            training_loss = l.cpu().item()
            train_loss.append(training_loss)

            print('global step: %s, epoch: %s, batch: %s, training loss: %.6f, time: %.2fs'
                  % (global_step, epoch, batch_num, training_loss, time() - start_time), flush=True)

            batch_num += 1
            global_step += 1

        print('mean train loss:', np.mean(train_loss))
        net.eval()
        val_loss = compute_loss(net, val_loader, risk_mask, road_adj, risk_adj, poi_adj,
                                grid_node_map, trans, device, bfc, data_type, ssl_loss_weight=ssl_loss_weight)
        print('global step: %s, epoch: %s,val loss：%.6f' % (global_step - 1, epoch, val_loss), flush=True)

        if epoch == 1 or val_loss < early_stop.best_score:
            metrics = evaluate_test_sets(
                net,
                test_loader,
                high_test_loader,
                risk_mask,
                road_adj,
                risk_adj,
                poi_adj,
                grid_node_map,
                trans,
                scaler,
                device,
            )
            best_epoch = epoch
            print('global step: %s, epoch: %s, test RMSE: %.4f,test Recall: %.2f%%,test MAP: %.4f,'
                  'high test RMSE: %.4f,high test Recall: %.2f%%,high test MAP: %.4f'
                  % (
                      global_step - 1,
                      epoch,
                      metrics["test_rmse"],
                      metrics["test_recall"],
                      metrics["test_map"],
                      metrics["high_test_rmse"],
                      metrics["high_test_recall"],
                      metrics["high_test_map"],
                  ), flush=True)
        else:
            metrics = {
                "test_rmse": early_stop.best_rmse,
                "test_recall": early_stop.best_recall,
                "test_map": early_stop.best_map,
                "high_test_rmse": early_stop.best_high_rmse,
                "high_test_recall": early_stop.best_high_recall,
                "high_test_map": early_stop.best_high_map,
                "test_inverse_trans_pre": early_stop.best_pre,
                "test_inverse_trans_label": early_stop.best_label,
            }

        flag = early_stop(
            val_loss,
            metrics["test_rmse"],
            metrics["test_recall"],
            metrics["test_map"],
            metrics["high_test_rmse"],
            metrics["high_test_recall"],
            metrics["high_test_map"],
            metrics["test_inverse_trans_pre"],
            metrics["test_inverse_trans_label"],
        )
        if epoch == 1 or flag:
            torch.save(net.state_dict(), CKPT_PATH)
            print("model saved in epoch: %s -> %s" % (epoch, CKPT_PATH), flush=True)

        if early_stop.early_stop:
            print("Early Stopping in global step: %s, epoch: %s" % (global_step, epoch), flush=True)
            print('best test RMSE: %.4f,best test Recall: %.2f%%,best test MAP: %.4f'
                  % (early_stop.best_rmse, early_stop.best_recall, early_stop.best_map), flush=True)
            print('best test high RMSE: %.4f,best test high Recall: %.2f%%,best high test MAP: %.4f'
                  % (early_stop.best_high_rmse, early_stop.best_high_recall, early_stop.best_high_map), flush=True)
            break

    if not early_stop.early_stop:
        print("Reached max epochs: %s" % training_epoch, flush=True)
        if best_epoch is not None:
            print("Best tracked test metrics last updated at epoch: %s" % best_epoch, flush=True)
        print('best test RMSE: %.4f,best test Recall: %.2f%%,best test MAP: %.4f'
              % (early_stop.best_rmse, early_stop.best_recall, early_stop.best_map), flush=True)
        print('best test high RMSE: %.4f,best test high Recall: %.2f%%,best high test MAP: %.4f'
              % (early_stop.best_high_rmse, early_stop.best_high_recall, early_stop.best_high_map), flush=True)

    return early_stop.best_rmse, early_stop.best_recall, early_stop.best_map


def main(config):
    patience, delta, train_rate, valid_rate, recent_prior, week_prior, one_day_period, days_of_week, pre_len, \
        seq_len, training_epoch, batch_size, learning_rate, weight_dacay, transformer_hidden_size, \
        gcn_num_filter, \
        all_data_filename, mask_filename, road_adj_filename, risk_adj_filename, poi_adj_filename, grid_node_filename, \
        north_south_map, west_east_map, trans_filename, bfc_filename, num_of_heads = loadConfig(config)

    bfc = get_trans(bfc_filename)
    nums_of_filter = []
    for _ in range(2):
        nums_of_filter.append(gcn_num_filter)

    scaler = []
    train_data_shape = []
    time_shape = []
    graph_feature_shape = []
    high_test_loader = []
    train_loader = []
    val_loader = []
    test_loader = []
    grid_node_map = []

    for i in range(4):
        grid_node_map.append(get_grid_node_map_maxtrix(grid_node_filename[i]))
        t_scaler, t_train_data_shape, t_time_shape, t_graph_feature_shape, t_loaders, t_high_test_loader = \
            generate_dataloader(
                all_data_filename=all_data_filename[i],
                grid_node_data_filename=grid_node_filename[i],
                batch_size=batch_size,
                train_rate=train_rate,
                valid_rate=valid_rate,
                recent_prior=recent_prior,
                week_prior=week_prior,
                one_day_period=one_day_period,
                days_of_week=days_of_week,
                pre_len=pre_len,
                test=args.test,
                north_south_map=north_south_map[i],
                west_east_map=west_east_map[i]
            )
        scaler.append(t_scaler)
        train_data_shape.append(t_train_data_shape)
        time_shape.append(t_time_shape)
        graph_feature_shape.append(t_graph_feature_shape)
        high_test_loader.append(t_high_test_loader)

        t_train_loader, t_val_loader, t_test_loader = t_loaders
        train_loader.append(t_train_loader)
        val_loader.append(t_val_loader)
        test_loader.append(t_test_loader)

    trans = []
    for i in trans_filename:
        trans.append(get_trans(i))

    node_num_list = [grid_map.shape[1] for grid_map in grid_node_map]
    ss_sthm_nmb_prototype = config.get("ss_sthm_nmb_prototype", 32)
    ss_sthm_tau = config.get("ss_sthm_tau", 0.5)
    ss_sthm_temporal_weight = config.get("ss_sthm_temporal_weight", 1.0)
    ss_sthm_spatial_weight = config.get("ss_sthm_spatial_weight", 1.0)
    ssl_loss_weight = config.get("ssl_loss_weight", 1e-2)
    tapm_te_dim = config.get("tapm_te_dim", 16)
    tapm_patch_num = config.get("tapm_patch_num", 4)
    tapm_dropout = config.get("tapm_dropout", 0.1)

    risk_mask = []
    road_adj = []
    risk_adj = []
    poi_adj = []
    for i in range(4):
        risk_mask.append(get_mask(mask_filename[i]))
        road_adj.append(get_adjacent(road_adj_filename[i]))
        risk_adj.append(get_adjacent(risk_adj_filename[i]))
        if poi_adj_filename[0] == "":
            poi_adj.append(None)
        else:
            poi_adj.append(get_adjacent(poi_adj_filename[i]))

    model = STTAR(
        transformer_hidden_size,
        pre_len,
        time_shape[0][1],
        graph_feature_shape[0][2],
        nums_of_filter,
        north_south_map,
        west_east_map,
        num_of_heads,
        node_num_list=node_num_list,
        tapm_te_dim=tapm_te_dim,
        tapm_patch_num=tapm_patch_num,
        tapm_dropout=tapm_dropout,
        ss_sthm_batch_size=batch_size,
        ss_sthm_device=str(device),
        ss_sthm_nmb_prototype=ss_sthm_nmb_prototype,
        ss_sthm_tau=ss_sthm_tau,
        ss_sthm_temporal_weight=ss_sthm_temporal_weight,
        ss_sthm_spatial_weight=ss_sthm_spatial_weight,
    )

    model.to(device)
    print(model)

    num_of_parameters = 0
    for name, parameters in model.named_parameters():
        num_of_parameters += np.prod(parameters.shape)
    print("Number of Parameters: {}".format(num_of_parameters), flush=True)

    trainer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_dacay)
    early_stop = EarlyStopping(patience=patience, delta=delta)

    for i in range(len(trans)):
        trans[i] = torch.from_numpy(trans[i]).unsqueeze(0).to(device)
    bfc = torch.from_numpy(bfc).unsqueeze(0).to(torch.float32).to(device)

    if args.infer:
        checkpoint = torch.load(CKPT_PATH, map_location=device)
        if isinstance(checkpoint, torch.nn.Module):
            model = checkpoint.to(device)
        else:
            model.load_state_dict(checkpoint)
            model.to(device)
        test_rmse, test_recall, test_map, test_inverse_trans_pre, test_inverse_trans_label = \
            predict_and_evaluate(model, test_loader, risk_mask, road_adj, risk_adj, poi_adj,
                                 grid_node_map, trans, scaler, device)

        high_test_rmse, high_test_recall, high_test_map, _, _ = \
            predict_and_evaluate(model, high_test_loader, risk_mask, road_adj, risk_adj, poi_adj,
                                 grid_node_map, trans, scaler, device)

        print('test RMSE: %.4f,test Recall: %.2f%%,test MAP: %.4f,'
              'high test RMSE: %.4f,high test Recall: %.2f%%,high test MAP: %.4f'
              % (test_rmse, test_recall, test_map, high_test_rmse, high_test_recall,
                 high_test_map), flush=True)
        if args.export_npz:
            os.makedirs(os.path.dirname(args.export_npz) or ".", exist_ok=True)
            np.savez_compressed(
                args.export_npz,
                prediction=test_inverse_trans_pre,
                truth=test_inverse_trans_label,
            )
            print("Saved visualization npz -> %s" % args.export_npz, flush=True)
    else:
        training(
            model,
            training_epoch,
            train_loader,
            val_loader,
            test_loader,
            high_test_loader,
            road_adj,
            risk_adj,
            poi_adj,
            risk_mask,
            grid_node_map,
            trainer,
            early_stop,
            device,
            scaler,
            trans,
            bfc,
            data_type=config['data_type'],
            ssl_loss_weight=ssl_loss_weight,
        )


if __name__ == "__main__":
    main(config)
