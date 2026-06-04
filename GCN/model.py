"""
Hand gesture GCN with bone features, joint angles, and residual connections.
3-layer ResGCN + MLP classifier. ~120K parameters.
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
    h = x - x[:, 0:1, :]
    span = h.max(dim=1, keepdim=True)[0] - h.min(dim=1, keepdim=True)[0]
    return h / span.clamp(min=1e-6)


def compute_bone_features(x, edges):
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


# Consecutive bone pairs defining joint angles
_ANGLE_PAIRS = [
    (1, 2, 3),    # thumb MCP
    (2, 3, 4),    # thumb IP
    (5, 6, 7),    # index PIP
    (6, 7, 8),    # index DIP
    (9, 10, 11),  # middle PIP
    (10, 11, 12), # middle DIP
    (13, 14, 15), # ring PIP
    (14, 15, 16), # ring DIP
    (17, 18, 19), # pinky PIP
    (18, 19, 20), # pinky DIP
    (0, 5, 9),    # wrist→index_mcp→middle_mcp (spread angle)
]


def compute_joint_angles(x):
    """Compute cosine of joint flexion angles at 11 key joints (vectorized)."""
    B, N, _ = x.shape; dev = x.device
    a = torch.tensor([p[0] for p in _ANGLE_PAIRS], device=dev, dtype=torch.long)
    b = torch.tensor([p[1] for p in _ANGLE_PAIRS], device=dev, dtype=torch.long)
    c = torch.tensor([p[2] for p in _ANGLE_PAIRS], device=dev, dtype=torch.long)
    K = len(_ANGLE_PAIRS)
    u = x[:, b] - x[:, a]; u = u / (u.norm(dim=-1, keepdim=True) + 1e-8)
    v = x[:, c] - x[:, b]; v = v / (v.norm(dim=-1, keepdim=True) + 1e-8)
    cos_all = (u * v).sum(dim=-1, keepdim=True)  # (B, K, 1)
    angles = torch.zeros(B, N, 1, device=dev)
    angles.scatter_(1, b.view(1, K, 1).expand(B, K, 1), cos_all)
    return angles


class ResGCNBlock(nn.Module):
    """Residual GCN: LN → GCN → ReLU → Dropout + skip connection"""

    def __init__(self, ch, dropout):
        super().__init__()
        self.norm = nn.LayerNorm(ch)
        self.conv = GCNConv(ch, ch)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, adj):
        return x + self.drop(F.relu(self.conv(self.norm(x), adj)))


class HandGCN(nn.Module):
    """3-layer ResGCN with bone features and joint angles.

    Input  (B,21,9)  xyz_norm + hand + bone*4 + angle
    GCN(9→160) + LN + ReLU + Drop
    ResGCN(160) → ResGCN(160) → ResGCN(160)
    GlobalMeanPool → Linear(160→80) + Drop → Linear(80→10)
    """

    def __init__(self, num_classes=10, hidden_dim=160, dropout=0.3):
        super().__init__()
        in_ch = 3 + 1 + 4 + 1  # xyz + hand + bone*4 + angle = 9
        self.input_proj = GCNConv(in_ch, hidden_dim)
        self.input_norm = nn.LayerNorm(hidden_dim)

        self.block1 = ResGCNBlock(hidden_dim, dropout)
        self.block2 = ResGCNBlock(hidden_dim, dropout)
        self.block3 = ResGCNBlock(hidden_dim, dropout)

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

        xn = normalize_landmarks(x)
        bone = compute_bone_features(xn, self._edges) if self._edges else \
               torch.zeros(B, N, 4, device=x.device)
        angles = compute_joint_angles(xn)
        hl = handedness[:, 1:2].unsqueeze(1).expand(B, N, 1)
        h = torch.cat([xn, hl, bone, angles], dim=-1)

        h = F.relu(self.input_norm(self.input_proj(h, adj)))
        h = self.drop(h)

        h = self.block1(h, adj)
        h = self.block2(h, adj)
        h = self.block3(h, adj)

        return self.classifier(h.mean(dim=1))


def create_model(device, num_classes=10, dropout=0.3):
    from dataset_metadata import HAND_EDGES
    m = HandGCN(num_classes=num_classes, dropout=dropout)
    m.set_edges(HAND_EDGES)
    return m.to(device)
