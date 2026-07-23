"""Our full model: APN (TAPM on all scales) + SS-STHM."""

import torch
import torch.nn as nn

from model.tapm import ST_Model
from model.ss_sthm import SSSTHMInterlossModel


class STTAR(nn.Module):
    def __init__(
        self,
        transformer_hidden_size,
        pre_len,
        num_of_target_time_feature,
        num_of_graph_feature,
        nums_of_graph_filters,
        north_south_map,
        west_east_map,
        num_of_heads,
        node_num_list=None,
        tapm_te_dim=16,
        tapm_patch_num=4,
        tapm_dropout=0.1,
        ss_sthm_batch_size=1,
        ss_sthm_device="cpu",
        ss_sthm_nmb_prototype=32,
        ss_sthm_tau=0.5,
        ss_sthm_temporal_weight=1.0,
        ss_sthm_spatial_weight=1.0,
    ):
        super(STTAR, self).__init__()
        self.north_south_map = north_south_map
        self.west_east_map = west_east_map
        self.fusion_channel = 16

        self.ST_Model = ST_Model(
            transformer_hidden_size,
            pre_len,
            num_of_target_time_feature,
            num_of_graph_feature,
            nums_of_graph_filters,
            north_south_map,
            west_east_map,
            num_of_heads,
            node_num_list=node_num_list,
            tapm_te_dim=tapm_te_dim,
            tapm_patch_num=tapm_patch_num,
            tapm_dropout=tapm_dropout,
        )

        self.ssl_loss_model = SSSTHMInterlossModel(
            self.fusion_channel,
            north_south_map[0] * west_east_map[0],
            ss_sthm_batch_size,
            ss_sthm_device,
            nmb_prototype=ss_sthm_nmb_prototype,
            tau=ss_sthm_tau,
            temporal_weight=ss_sthm_temporal_weight,
            spatial_weight=ss_sthm_spatial_weight,
        )

        self.output_layer = nn.ModuleList(
            [
                nn.Linear(
                    2 * self.fusion_channel * north_south_map[0] * west_east_map[0],
                    pre_len * north_south_map[0] * west_east_map[0],
                ),
                nn.Linear(
                    2 * self.fusion_channel * north_south_map[1] * west_east_map[1],
                    pre_len * north_south_map[1] * west_east_map[1],
                ),
                nn.Linear(
                    2 * self.fusion_channel * north_south_map[2] * west_east_map[2],
                    pre_len * north_south_map[2] * west_east_map[2],
                ),
                nn.Linear(
                    2 * self.fusion_channel * north_south_map[3] * west_east_map[3],
                    pre_len * north_south_map[3] * west_east_map[3],
                ),
            ]
        )

    def forward(self, r_input, w_input, target_time_feature, road_adj, risk_adj, poi_adj, grid_node_map, trans):
        batch_size = r_input[0].shape[0]

        rr_output = self.ST_Model(r_input, target_time_feature, road_adj, risk_adj, poi_adj, grid_node_map, trans)
        ww_output = self.ST_Model(w_input, target_time_feature, road_adj, risk_adj, poi_adj, grid_node_map, trans)

        r_output = self.ST_Model.time_forward(rr_output, ww_output, target_time_feature, grid_node_map)
        w_output = self.ST_Model.time_forward(ww_output, rr_output, target_time_feature, grid_node_map)

        fusion_output = []
        final_output = []
        classification_output = []

        ssl_loss = self.ssl_loss_model(r_output[0], w_output[0])

        for i in range(4):
            fusion_output.append(torch.cat((r_output[i], w_output[i]), dim=1))

        for i in range(4):
            fusion_output[i] = fusion_output[i].contiguous().view(batch_size, -1)
            final_output.append(
                self.output_layer[i](fusion_output[i]).view(
                    batch_size, -1, self.north_south_map[i], self.west_east_map[i]
                )
            )
            classification_output.append(torch.relu(final_output[i].view(final_output[i].shape[0], -1)))

        return final_output, classification_output, ssl_loss
