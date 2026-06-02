"""
Hand gesture GCN with bone features and per-axis normalization.
3-layer GCNConv + LayerNorm + MLP classifier. ~68K parameters.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class GCNConv(nn.Module):
    """Graph convolution: H' = D^(-1/2) A_hat D^(-1/2) H W"""

    def __init__(self, in_ch, out_ch, bias=True):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(in_ch, out_ch))
        self.bias = nn.Parameter(torch.empty(out_ch)) if bias else None
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x, adj):
        s = torch.matmul(x, self.weight)
        o = torch.matmul(adj, s)
        return o + self.bias if self.bias is not None else o


def build_adj_matrix(edges, N=21, normalize=True):
    adj = torch.zeros(N, N)
    for i in range(N): adj[i, i] = 1.0
    for u, v in edges: adj[u, v] = adj[v, u] = 1.0
    if normalize:
        deg = adj.sum(dim=1)
        d_inv_sqrt = torch.diag(torch.pow(deg, -0.5).clamp(min=0))
        adj = d_inv_sqrt @ adj @ d_inv_sqrt
    return adj


def normalize_landmarks(x):
    """Per-axis normalization: center on wrist, divide each axis by its span."""
    h = x - x[:, 0:1, :]
    span = h.max(dim=1, keepdim=True)[0] - h.min(dim=1, keepdim=True)[0]
    return h / span.clamp(min=1e-6)


def compute_bone_features(x, edges):
    """Bone direction (dx,dy,dz) and length, aggregated to target nodes via scatter-mean."""
    B, N, _ = x.shape; dev = x.device
    src = torch.tensor([e[0] for e in edges], device=dev, dtype=torch.long)
    dst = torch.tensor([e[1] for e in edges], device=dev, dtype=torch.long)
    E = len(edges)
    ev = x[:, dst] - x[:, src]
    el = ev.norm(dim=-1, keepdim=True)
    ef = torch.cat([ev / (el + 1e-8), el], dim=-1)
    bone = torch.zeros(B, N, 4, device=dev)
    bone.scatter_add_(1, dst.view(1, E, 1).expand(B, E, 4), ef)
    cnt = torch.zeros(B, N, 1, device=dev)
    cnt.scatter_add_(1, dst.view(1, E, 1).expand(B, E, 1),
                     torch.ones(B, E, 1, device=dev))
    return bone / (cnt + 1e-8)


class HandGCN(nn.Module):
    """3-layer GCN with bone features.

    Input  (B,21,8)  xyz_norm + handedness + bone*4
    GCNConv(8->160) + LN + ReLU + Drop
    GCNConv(160->160) + LN + ReLU + Drop
    GCNConv(160->160) + LN + ReLU
    GlobalMeanPool -> Linear(160->80) + ReLU + Drop -> Linear(80->10)
    """

    def __init__(self, num_classes=10, hidden_dim=160, dropout=0.3):
        super().__init__()

        self.conv1 = GCNConv(8, hidden_dim)
        self.norm1 = nn.LayerNorm(hidden_dim)

        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)

        self.conv3 = GCNConv(hidden_dim, hidden_dim)
        self.norm3 = nn.LayerNorm(hidden_dim)

        self.drop = nn.Dropout(dropout)

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes),
        )

        self._edges = None

    def set_edges(self, edges):
        self._edges = edges

    def forward(self, x, adj, handedness):
        B, N, _ = x.shape

        # Normalize + bone features
        xn = normalize_landmarks(x)
        bone = compute_bone_features(xn, self._edges) if self._edges else \
               torch.zeros(B, N, 4, device=x.device)
        hl = handedness[:, 1:2].unsqueeze(1).expand(B, N, 1)
        h = torch.cat([xn, hl, bone], dim=-1)

        # Three GCN layers
        h = self.conv1(h, adj); h = self.norm1(h); h = F.relu(h); h = self.drop(h)
        h = self.conv2(h, adj); h = self.norm2(h); h = F.relu(h); h = self.drop(h)
        h = self.conv3(h, adj); h = self.norm3(h); h = F.relu(h)

        # Global mean pool + classify
        return self.classifier(h.mean(dim=1))


def create_model(device, num_classes=10, dropout=0.3):
    from dataset_metadata import HAND_EDGES
    m = HandGCN(num_classes=num_classes, dropout=dropout)
    m.set_edges(HAND_EDGES)
    return m.to(device)
