"""SS-STHM: SHM + THM self-supervised losses and SSSTHMInterlossModel wrapper."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class SHM(nn.Module):
    """Spatial Heterogeneity Modeling (SHM) for SS-STHM."""

    def __init__(self, c_in, nmb_prototype, batch_size, tau=0.5):
        super(SHM, self).__init__()
        self.prototypes = nn.Linear(c_in, nmb_prototype, bias=False)

        self.tau = tau
        self.d_model = c_in
        self.batch_size = batch_size

        for module in self.modules():
            self.weights_init(module)

    def weights_init(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight.data)
            if module.bias is not None:
                module.bias.data.fill_(0.0)

    @staticmethod
    def l2norm(x):
        return F.normalize(x, dim=1, p=2)

    def forward(self, z1, z2):
        with torch.no_grad():
            weight = self.prototypes.weight.data.clone()
            weight = self.l2norm(weight)
            self.prototypes.weight.copy_(weight)

        zc1 = self.prototypes(self.l2norm(z1.reshape(-1, self.d_model)))
        zc2 = self.prototypes(self.l2norm(z2.reshape(-1, self.d_model)))
        with torch.no_grad():
            q1 = sinkhorn(zc1.detach())
            q2 = sinkhorn(zc2.detach())
        l1 = -torch.mean(torch.sum(q1 * F.log_softmax(zc2 / self.tau, dim=1), dim=1))
        l2 = -torch.mean(torch.sum(q2 * F.log_softmax(zc1 / self.tau, dim=1), dim=1))
        return l1 + l2


@torch.no_grad()
def sinkhorn(out, epsilon=0.05, sinkhorn_iterations=3):
    q = torch.exp(out / epsilon).t()
    batch_size = q.shape[1]
    num_clusters = q.shape[0]

    q /= torch.sum(q)

    for _ in range(sinkhorn_iterations):
        q /= torch.sum(q, dim=1, keepdim=True)
        q /= num_clusters

        q /= torch.sum(q, dim=0, keepdim=True)
        q /= batch_size

    q *= batch_size
    return q.t()


class THM(nn.Module):
    """Temporal Heterogeneity Modeling (THM) for SS-STHM."""

    def __init__(self, c_in, batch_size, num_nodes, device):
        super(THM, self).__init__()
        self.W1 = nn.Parameter(torch.FloatTensor(num_nodes, c_in))
        self.W2 = nn.Parameter(torch.FloatTensor(num_nodes, c_in))
        nn.init.kaiming_uniform_(self.W1, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.W2, a=math.sqrt(5))

        self.read = AvgReadout()
        self.disc = Discriminator(c_in)
        self.b_xent = nn.BCEWithLogitsLoss()
        self.batch_size = batch_size
        self.num_nodes = num_nodes
        self.device = device

    def forward(self, z1, z2):
        h = (z1 * self.W1 + z2 * self.W2).squeeze(1)
        summary = self.read(h)

        idx = torch.randperm(h.shape[0], device=h.device)
        shuf_h = h[idx]

        logits = self.disc(summary, h, shuf_h)
        lbl_rl = torch.ones(h.shape[0], h.shape[1], device=logits.device)
        lbl_fk = torch.zeros(h.shape[0], h.shape[1], device=logits.device)
        lbl = torch.cat((lbl_rl, lbl_fk), dim=1)
        return self.b_xent(logits, lbl)


class AvgReadout(nn.Module):
    def __init__(self):
        super(AvgReadout, self).__init__()
        self.sigm = nn.Sigmoid()

    def forward(self, h):
        summary = torch.mean(h, dim=1)
        return self.sigm(summary)


class Discriminator(nn.Module):
    def __init__(self, n_h):
        super(Discriminator, self).__init__()
        self.net = nn.Bilinear(n_h, n_h, 1)

        for module in self.modules():
            self.weights_init(module)

    def weights_init(self, module):
        if isinstance(module, nn.Bilinear):
            torch.nn.init.xavier_uniform_(module.weight.data)
            if module.bias is not None:
                module.bias.data.fill_(0.0)

    def forward(self, summary, h_rl, h_fk):
        summary = torch.unsqueeze(summary, dim=1)
        summary = summary.expand_as(h_rl).contiguous()

        sc_rl = torch.squeeze(self.net(h_rl, summary), dim=2)
        sc_fk = torch.squeeze(self.net(h_fk, summary), dim=2)
        return torch.cat((sc_rl, sc_fk), dim=1)


def grid_to_ss_sthm_view(z):
    """Convert [B, C, H, W] grid features into SS-STHM's [B, 1, N, C] view."""

    if z.dim() != 4:
        raise ValueError(f"Expected a 4D tensor [B, C, H, W], but got shape {tuple(z.shape)}.")
    return z.flatten(2).transpose(1, 2).unsqueeze(1).contiguous()


class SSSTHMInterlossModel(nn.Module):
    """SS-STHM auxiliary loss: weighted SHM (spatial) + THM (temporal)."""

    def __init__(
        self,
        c_in,
        num_nodes,
        batch_size,
        device,
        nmb_prototype=32,
        tau=0.5,
        temporal_weight=1.0,
        spatial_weight=1.0,
    ):
        super().__init__()
        self.temporal_model = THM(c_in, batch_size, num_nodes, device)
        self.spatial_model = SHM(c_in, nmb_prototype, batch_size, tau=tau)
        self.temporal_weight = temporal_weight
        self.spatial_weight = spatial_weight

    def forward(self, z1, z2):
        z1_view = grid_to_ss_sthm_view(z1)
        z2_view = grid_to_ss_sthm_view(z2)

        temporal_loss = self.temporal_model(z1_view, z2_view)
        spatial_loss = self.spatial_model(z1_view, z2_view)
        return self.temporal_weight * temporal_loss + self.spatial_weight * spatial_loss


__all__ = [
    "AvgReadout",
    "Discriminator",
    "SHM",
    "SSSTHMInterlossModel",
    "THM",
    "grid_to_ss_sthm_view",
    "sinkhorn",
]
