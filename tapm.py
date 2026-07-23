"""TAPM / APN building blocks stitched into the ST-TAR backbone."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]


class TemporalAnchorPatching(nn.Module):
    def __init__(self, N, P, S, te_dim, hid_dim, history, feature_dim, dropout_rate=0.1):
        super().__init__()
        self.N = N
        self.P = P
        self.S = max(history / P, 1e-6) if S is None else S
        self.history = history
        self.hid_dim = hid_dim
        self.te_dim = te_dim
        self.feature_dim = feature_dim + te_dim
        self.delta_left_params = nn.Parameter(torch.zeros(N, P))
        self.raw_log_width_params = nn.Parameter(torch.full((N, P), math.log(self.S)))
        self.tau_params = nn.Parameter(torch.zeros(N))
        self.projection_layer = nn.Linear(self.feature_dim, self.hid_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hid_dim, hid_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hid_dim * 2, hid_dim),
        )
        self.norm = nn.LayerNorm(hid_dim)

    def forward(self, t_stacked, x_with_te, mask_stacked):
        current_device = t_stacked.device
        B_N, _, _ = t_stacked.shape
        B = B_N // self.N
        patch_centers = torch.linspace(self.S / 2, self.history - self.S / 2, self.P, device=current_device)
        base_left_boundaries = (patch_centers - self.S / 2).unsqueeze(0)
        t_left_n_p = base_left_boundaries + self.delta_left_params
        width_learned_n_p = torch.exp(self.raw_log_width_params) + 1e-6
        t_right_n_p = t_left_n_p + width_learned_n_p
        current_variable_taus = F.softplus(self.tau_params).unsqueeze(-1) + 1e-6
        t_left_b_n = t_left_n_p.unsqueeze(0).expand(B, -1, -1).reshape(B_N, self.P).unsqueeze(-1)
        t_right_b_n = t_right_n_p.unsqueeze(0).expand(B, -1, -1).reshape(B_N, self.P).unsqueeze(-1)
        taus_b_n = current_variable_taus.unsqueeze(0).expand(B, -1, -1).reshape(B_N, 1).unsqueeze(-1)
        t_raw_b_n = t_stacked.transpose(-1, -2)
        weights_raw = torch.sigmoid((t_right_b_n - t_raw_b_n) / taus_b_n) * torch.sigmoid(
            (t_raw_b_n - t_left_b_n) / taus_b_n
        )
        mask_b_n = mask_stacked.transpose(-1, -2)
        temporal_weights = weights_raw * mask_b_n
        sum_weights = temporal_weights.sum(dim=-1, keepdim=True) + 1e-9
        weighted_features_sum = torch.bmm(temporal_weights, x_with_te)
        h_patches_avg = weighted_features_sum / sum_weights
        h_patches_proj = self.projection_layer(h_patches_avg)
        h_patches = self.norm(h_patches_proj + self.ffn(h_patches_proj))
        return h_patches


class LearnableTimeEmbedding(nn.Module):
    def __init__(self, te_dim):
        super(LearnableTimeEmbedding, self).__init__()
        self.te_scale = nn.Linear(1, 1)
        self.te_periodic = nn.Linear(1, te_dim - 1)

    def forward(self, tt):
        out1 = self.te_scale(tt)
        out2 = torch.sin(self.te_periodic(tt))
        return torch.cat([out1, out2], -1)


class GCN_Layer(nn.Module):
    def __init__(self, num_of_features, num_of_filter):
        super(GCN_Layer, self).__init__()
        self.gcn_layer = nn.Sequential(
            nn.Linear(in_features=num_of_features, out_features=num_of_filter),
            nn.ReLU(),
        )

    def forward(self, input, adj):
        batch_size, _, _ = input.shape
        adj = torch.from_numpy(adj).to(input.device)
        adj = adj.repeat(batch_size, 1, 1)
        input = torch.bmm(adj, input)
        output = self.gcn_layer(input)
        return output


class GraphConv(nn.Module):
    def __init__(self, num_of_graph_feature, nums_of_graph_filters):
        super(GraphConv, self).__init__()
        self.road_gcn = nn.ModuleList()
        for idx, num_of_filter in enumerate(nums_of_graph_filters):
            if idx == 0:
                self.road_gcn.append(GCN_Layer(num_of_graph_feature, num_of_filter))
            else:
                self.road_gcn.append(GCN_Layer(nums_of_graph_filters[idx - 1], num_of_filter))

        self.risk_gcn = nn.ModuleList()
        for idx, num_of_filter in enumerate(nums_of_graph_filters):
            if idx == 0:
                self.risk_gcn.append(GCN_Layer(num_of_graph_feature, num_of_filter))
            else:
                self.risk_gcn.append(GCN_Layer(nums_of_graph_filters[idx - 1], num_of_filter))

        self.poi_gcn = nn.ModuleList()
        for idx, num_of_filter in enumerate(nums_of_graph_filters):
            if idx == 0:
                self.poi_gcn.append(GCN_Layer(num_of_graph_feature, num_of_filter))
            else:
                self.poi_gcn.append(GCN_Layer(nums_of_graph_filters[idx - 1], num_of_filter))

    def forward(self, graph_feature, road_adj, risk_adj, poi_adj):
        batch_size, T, D1, N = graph_feature.shape

        road_graph_output = graph_feature.view(-1, D1, N).permute(0, 2, 1).contiguous()
        for gcn_layer in self.road_gcn:
            road_graph_output = gcn_layer(road_graph_output, road_adj)

        risk_graph_output = graph_feature.view(-1, D1, N).permute(0, 2, 1).contiguous()
        for gcn_layer in self.risk_gcn:
            risk_graph_output = gcn_layer(risk_graph_output, risk_adj)

        graph_output = road_graph_output + risk_graph_output

        if poi_adj is not None:
            poi_graph_output = graph_feature.view(-1, D1, N).permute(0, 2, 1).contiguous()
            for gcn_layer in self.poi_gcn:
                poi_graph_output = gcn_layer(poi_graph_output, poi_adj)
            graph_output += poi_graph_output

        graph_output = (
            graph_output.view(batch_size, T, N, -1)
            .permute(0, 2, 1, 3)
            .contiguous()
            .view(batch_size * N, T, -1)
            .view(batch_size, T, -1, N)
        )
        return graph_output


class TAPMEncoder(nn.Module):
    def __init__(
        self,
        transformer_hidden_size,
        num_of_target_time_feature,
        north_south_map,
        west_east_map,
        num_nodes=None,
        tapm_te_dim=16,
        tapm_patch_num=4,
        tapm_dropout=0.1,
    ):
        super(TAPMEncoder, self).__init__()
        self.hidden_size = transformer_hidden_size
        self.north_south_map = north_south_map
        self.west_east_map = west_east_map
        self.num_nodes = num_nodes
        self.param_nodes = 1 if num_nodes is None else num_nodes
        self.tapm_patch_num = tapm_patch_num
        self.time_embedding = LearnableTimeEmbedding(tapm_te_dim)
        self.patching = TemporalAnchorPatching(
            N=self.param_nodes,
            P=tapm_patch_num,
            S=None,
            te_dim=tapm_te_dim,
            hid_dim=transformer_hidden_size,
            history=1.0,
            feature_dim=transformer_hidden_size,
            dropout_rate=tapm_dropout,
        )
        self.patch_pos_enc = PositionalEncoding(transformer_hidden_size, max_len=tapm_patch_num)
        self.var_queries = nn.Parameter(torch.randn(1, self.param_nodes, 1, transformer_hidden_size))
        self.aggregation_norm = nn.LayerNorm(transformer_hidden_size)
        self.other_proj = nn.Linear(transformer_hidden_size, transformer_hidden_size)
        self.target_time_proj = nn.Linear(num_of_target_time_feature, transformer_hidden_size)
        self.output_norm = nn.LayerNorm(transformer_hidden_size)

    def forward(self, g_output, o_output, target_time_feature, grid_node_map):
        batch_size, T, _, N = g_output.shape
        if self.num_nodes is not None and self.num_nodes != N:
            raise ValueError(f"TAPMEncoder expected {self.num_nodes} nodes, but got {N}.")

        graph_output = (
            g_output.view(batch_size, T, N, -1)
            .permute(0, 2, 1, 3)
            .contiguous()
            .view(batch_size * N, T, -1)
        )

        history_time = torch.linspace(0, 1, T, device=g_output.device, dtype=g_output.dtype).view(1, T, 1)
        history_time = history_time.repeat(batch_size * N, 1, 1)
        history_mask = torch.ones(batch_size * N, T, 1, device=g_output.device, dtype=g_output.dtype)
        te_his = self.time_embedding(history_time)
        x_with_te = torch.cat([graph_output, te_his], dim=-1)

        h_patches = self.patching(history_time, x_with_te, history_mask)
        h_patches = self.patch_pos_enc(h_patches)
        h_patches = h_patches.view(batch_size, N, self.tapm_patch_num, self.hidden_size)

        attn_scores = torch.matmul(self.var_queries, h_patches.transpose(-1, -2)) * (self.hidden_size ** -0.5)
        attn_weights = F.softmax(attn_scores, dim=-1)
        h_final = torch.matmul(attn_weights, h_patches).squeeze(-2)
        h_final = self.aggregation_norm(h_final)

        other_output = o_output.mean(dim=1).permute(0, 2, 1).contiguous()
        target_time_context = self.target_time_proj(target_time_feature).unsqueeze(1).expand(-1, N, -1)
        h_final = self.output_norm(h_final + self.other_proj(other_output) + target_time_context)

        grid_node_map_tmp = torch.from_numpy(grid_node_map).to(h_final.device, dtype=h_final.dtype).repeat(
            batch_size, 1, 1
        )
        h_final = torch.bmm(grid_node_map_tmp, h_final).permute(0, 2, 1).contiguous()
        h_final = h_final.view(batch_size, -1, self.north_south_map, self.west_east_map)
        return h_final


class ST_Model(nn.Module):
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
    ):
        super(ST_Model, self).__init__()
        self.north_south_map = north_south_map
        self.west_east_map = west_east_map
        self.fusion_channel = 16
        if node_num_list is None:
            node_num_list = [None] * 4

        self.GConv = nn.ModuleList(
            [
                GraphConv(num_of_graph_feature, nums_of_graph_filters),
                GraphConv(num_of_graph_feature, nums_of_graph_filters),
                GraphConv(num_of_graph_feature, nums_of_graph_filters),
                GraphConv(num_of_graph_feature, nums_of_graph_filters),
            ]
        )

        self.GTsEncoder = nn.ModuleList(
            [
                TAPMEncoder(
                    transformer_hidden_size,
                    num_of_target_time_feature,
                    north_south_map[0],
                    west_east_map[0],
                    num_nodes=node_num_list[0],
                    tapm_te_dim=tapm_te_dim,
                    tapm_patch_num=tapm_patch_num,
                    tapm_dropout=tapm_dropout,
                ),
                TAPMEncoder(
                    transformer_hidden_size,
                    num_of_target_time_feature,
                    north_south_map[1],
                    west_east_map[1],
                    num_nodes=node_num_list[1],
                    tapm_te_dim=tapm_te_dim,
                    tapm_patch_num=tapm_patch_num,
                    tapm_dropout=tapm_dropout,
                ),
                TAPMEncoder(
                    transformer_hidden_size,
                    num_of_target_time_feature,
                    north_south_map[2],
                    west_east_map[2],
                    num_nodes=node_num_list[2],
                    tapm_te_dim=tapm_te_dim,
                    tapm_patch_num=tapm_patch_num,
                    tapm_dropout=tapm_dropout,
                ),
                TAPMEncoder(
                    transformer_hidden_size,
                    num_of_target_time_feature,
                    north_south_map[3],
                    west_east_map[3],
                    num_nodes=node_num_list[3],
                    tapm_te_dim=tapm_te_dim,
                    tapm_patch_num=tapm_patch_num,
                    tapm_dropout=tapm_dropout,
                ),
            ]
        )

        self.graph_weight = nn.ModuleList(
            [
                nn.Conv2d(in_channels=transformer_hidden_size, out_channels=self.fusion_channel, kernel_size=1),
                nn.Conv2d(in_channels=transformer_hidden_size, out_channels=self.fusion_channel, kernel_size=1),
                nn.Conv2d(in_channels=transformer_hidden_size, out_channels=self.fusion_channel, kernel_size=1),
                nn.Conv2d(in_channels=transformer_hidden_size, out_channels=self.fusion_channel, kernel_size=1),
            ]
        )

    def forward(self, input, target_time_feature, road_adj, risk_adj, poi_adj, grid_node_map, trans):
        graph_output = []
        for i in range(4):
            graph_output.append(self.GConv[i](input[i], road_adj[i], risk_adj[i], poi_adj[i]))

        for i in range(4 - 1):
            f_graph_output = graph_output[i]
            c_graph_output = graph_output[i + 1]

            batch_size, T, _, f_N = f_graph_output.shape
            batch_size1, _, _, c_N = c_graph_output.shape

            c_graph_output = c_graph_output.reshape(batch_size1 * T, -1, c_N)
            cf_out = torch.matmul(c_graph_output, trans[i] / 3)
            f1_graph_output = f_graph_output + 0.2 * cf_out.reshape(batch_size1, T, -1, f_N)

            f_graph_output = f_graph_output.reshape(batch_size * T, -1, f_N)
            fc_out = torch.matmul(f_graph_output, trans[i].permute(0, 2, 1) / 3)

            c_graph_output = c_graph_output.reshape(batch_size1, T, -1, c_N)
            c1_graph_output = c_graph_output + 0.8 * fc_out.reshape(batch_size, T, -1, c_N)

            graph_output[i] = f1_graph_output
            graph_output[i + 1] = c1_graph_output
        return graph_output

    def time_forward(self, graph_output, other_output, target_time_feature, grid_node_map):
        time_output = []
        for i in range(4):
            time_output.append(self.GTsEncoder[i](graph_output[i], other_output[i], target_time_feature[i], grid_node_map[i]))
            time_output[i] = self.graph_weight[i](time_output[i])
        return time_output
