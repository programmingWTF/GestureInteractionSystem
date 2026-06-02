"""
GCN 手势识别模型 — 增强版（HandGCN++）

架构特点：
  1. 骨骼向量特征 — 边上显式编码 (dx, dy, dz, length)
  2. 残差 GCN 块 ×6 — Pre-LayerNorm + 双卷积 + 跳跃连接
  3. 多尺度融合（JumpingKnowledge）— 拼接浅/中/深层特征
  4. 注意力读出 — 自动学习哪些节点重要（指尖权重更高）
  5. 左右手门控融合 — 不是简单拼一维向量，而是用 FiLM 调制

输入：
  - x:           (batch, 21, 3)   关键点 (x,y,z)
  - handedness:  (batch, 2)       左右手 one-hot

输出：
  - (batch, num_classes)  logits

纯 PyTorch 实现，无需 PyTorch Geometric。
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ══════════════════════════════════════════════════════════
# 基础组件
# ══════════════════════════════════════════════════════════

class GCNConv(nn.Module):
    """
    图卷积: H' = D̂^(-1/2) Â D̂^(-1/2) H W

    Â = A + I（邻接矩阵 + 自环），D̂ 为 Â 的度矩阵。
    """

    def __init__(self, in_channels: int, out_channels: int, bias: bool = True):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(in_channels, out_channels))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x:    (B, N, C_in)   节点特征
            adj:  (N, N)         归一化邻接矩阵
        Returns:
            (B, N, C_out)
        """
        support = torch.matmul(x, self.weight)          # (B, N, C_out)
        output = torch.matmul(adj, support)             # (B, N, C_out)
        if self.bias is not None:
            output = output + self.bias
        return output


class LayerNorm1d(nn.Module):
    """
    对每个节点的特征做 LayerNorm（沿最后一维）。

    与 nn.LayerNorm 相同，但不用指定 normalized_shape，
    自动适配任意 (B, N, C) 输入。
    """

    def __init__(self, channels: int, eps: float = 1e-5):
        super().__init__()
        self.norm = nn.LayerNorm(channels, eps=eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, N, C) → 对最后一维归一化
        return self.norm(x)


class ResidualGCNBlock(nn.Module):
    """
    残差 GCN 块（Pre-norm 风格）：

        x → LN → GCN → ReLU → Dropout
          → LN → GCN → ReLU → Dropout
          + 残差连接（带可选 1×1 投影对齐维度）
    """

    def __init__(self, channels: int, dropout: float = 0.3):
        super().__init__()
        self.norm1 = LayerNorm1d(channels)
        self.conv1 = GCNConv(channels, channels)
        self.norm2 = LayerNorm1d(channels)
        self.conv2 = GCNConv(channels, channels)
        self.dropout = nn.Dropout(dropout)

        # 残差投影（当 in_ch != out_ch 时用，这里同维度所以是 Identity）
        self.proj = nn.Identity()

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        residual = self.proj(x)

        h = self.norm1(x)
        h = self.conv1(h, adj)
        h = F.relu(h)
        h = self.dropout(h)

        h = self.norm2(h)
        h = self.conv2(h, adj)
        h = F.relu(h)
        h = self.dropout(h)

        return h + residual


class ResidualGCNBlockProj(nn.Module):
    """
    带投影的残差 GCN 块（用于维度变化处）。
    """

    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.3):
        super().__init__()
        self.norm1 = LayerNorm1d(in_channels)
        self.conv1 = GCNConv(in_channels, out_channels)
        self.norm2 = LayerNorm1d(out_channels)
        self.conv2 = GCNConv(out_channels, out_channels)
        self.dropout = nn.Dropout(dropout)

        # 1×1 卷积投影对齐维度（沿特征轴）
        self.proj = nn.Linear(in_channels, out_channels, bias=False) \
            if in_channels != out_channels else nn.Identity()

        self.out_channels = out_channels

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        residual = self.proj(x)

        h = self.norm1(x)
        h = self.conv1(h, adj)
        h = F.relu(h)
        h = self.dropout(h)

        h = self.norm2(h)
        h = self.conv2(h, adj)
        h = F.relu(h)
        h = self.dropout(h)

        return h + residual


class AttentionReadout(nn.Module):
    """
    注意力读出层：

        α_i = softmax( Linear(h_i) )    ← 每个节点的标量注意力权重
        h_out = Σ α_i · h_i             ← 加权求和
    """

    def __init__(self, channels: int):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(channels, channels // 4),
            nn.Tanh(),
            nn.Linear(channels // 4, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, N, C)  节点特征
        Returns:
            (B, C)        加权聚合后的图级特征
        """
        attn_scores = self.attn(x).squeeze(-1)   # (B, N)
        attn_weights = F.softmax(attn_scores, dim=1)  # (B, N)
        out = torch.bmm(attn_weights.unsqueeze(1), x).squeeze(1)  # (B, C)
        return out


class FiLMFusion(nn.Module):
    """
    FiLM（Feature-wise Linear Modulation）融合左右手信息：

        h_out = γ(hand) ⊙ h + β(hand)

    相比简单拼接，FiLM 可以按通道调制图特征，让模型学会
    "左手应该关注 xx 通道，右手应该关注 yy 通道"。
    """

    def __init__(self, hand_dim: int, feature_dim: int):
        super().__init__()
        self.gamma = nn.Linear(hand_dim, feature_dim)
        self.beta = nn.Linear(hand_dim, feature_dim)

    def forward(self, h: torch.Tensor, hand_feat: torch.Tensor) -> torch.Tensor:
        """
        Args:
            h:          (B, C)  图级特征
            hand_feat:  (B, H)  左右手特征
        Returns:
            (B, C)
        """
        gamma = self.gamma(hand_feat)   # (B, C)
        beta = self.beta(hand_feat)     # (B, C)
        return gamma * h + beta


# ══════════════════════════════════════════════════════════
# 邻接矩阵 & 骨骼特征
# ══════════════════════════════════════════════════════════

def build_adj_matrix(edges, num_nodes=21, normalize=True):
    """
    从边列表构建归一化邻接矩阵 D̂^(-1/2) Â D̂^(-1/2)。
    """
    adj = torch.zeros(num_nodes, num_nodes)

    for i in range(num_nodes):
        adj[i, i] = 1.0

    for u, v in edges:
        adj[u, v] = 1.0
        adj[v, u] = 1.0

    if normalize:
        deg = adj.sum(dim=1)
        deg_inv_sqrt = torch.pow(deg, -0.5)
        deg_inv_sqrt[torch.isinf(deg_inv_sqrt)] = 0.0
        d_inv_sqrt = torch.diag(deg_inv_sqrt)
        adj = d_inv_sqrt @ adj @ d_inv_sqrt

    return adj


def compute_bone_features(x: torch.Tensor, edges) -> torch.Tensor:
    """
    计算骨骼（边）特征。

    对每条边 (u, v)：
        bone_direction = (x_v - x_u) / length     — 单位方向向量
        bone_length    = ||x_v - x_u||             — 骨骼长度

    然后聚合到节点：每个节点的入边骨骼特征取平均。

    Args:
        x:      (B, N, 3)  节点坐标
        edges:  [(u,v), ...]  边列表
    Returns:
        (B, N, 4)  每个节点的聚合骨骼特征 (dx, dy, dz, length)
    """
    B, N, _ = x.shape
    device = x.device

    # 每条边的源/目标索引
    src = torch.tensor([e[0] for e in edges], device=device, dtype=torch.long)
    dst = torch.tensor([e[1] for e in edges], device=device, dtype=torch.long)
    E = len(edges)

    # 边向量: (B, E, 3)
    edge_vec = x[:, dst] - x[:, src]              # (B, E, 3)
    edge_len = edge_vec.norm(dim=-1, keepdim=True)  # (B, E, 1)
    edge_dir = edge_vec / (edge_len + 1e-8)        # (B, E, 3)

    edge_feat = torch.cat([edge_dir, edge_len], dim=-1)  # (B, E, 4)

    # 聚合到目标节点: scatter_mean
    # 对每个目标节点，平均所有入边的特征
    bone_feat = torch.zeros(B, N, 4, device=device)
    ones = torch.ones(B, E, 1, device=device)

    # 累加
    bone_feat.scatter_add_(1, dst.view(1, E, 1).expand(B, E, 4), edge_feat)
    # 计数
    count = torch.zeros(B, N, 1, device=device)
    count.scatter_add_(1, dst.view(1, E, 1).expand(B, E, 1), ones)

    # 平均（避免除零，手腕可能没有入边）
    bone_feat = bone_feat / (count + 1e-8)

    return bone_feat


# ══════════════════════════════════════════════════════════
# HandGCN++ 主模型
# ══════════════════════════════════════════════════════════

class HandGCN(nn.Module):
    """
    HandGCN++ — 增强型手部关键点图卷积网络。

    数据流：
        x (B,21,3)  ─┬─→ GCN 骨干 ─→ 多尺度拼接 ─→ 注意力读出 ─┐
                       │                                          ├─→ MLP → 10类
        bone_feat     ─┴─→ 输入增强                               │
        handedness ──────────────────────────→ FiLM 调制 ────────┘
    """

    def __init__(
        self,
        num_classes: int = 10,
        in_channels: int = 3,        # (x, y, z)
        bone_channels: int = 4,      # (dx, dy, dz, len)
        hidden_dims: tuple = (64, 128, 256, 256, 128, 64),
        dropout: float = 0.35,
    ):
        super().__init__()

        total_in = in_channels + bone_channels   # 7
        self.hidden_dims = hidden_dims

        # ── 输入投影 ──
        self.input_proj = nn.Sequential(
            nn.Linear(total_in, hidden_dims[0]),
            LayerNorm1d(hidden_dims[0]),
        )

        # ── 残差 GCN 块 ──
        self.blocks = nn.ModuleList()
        prev_dim = hidden_dims[0]
        for i, dim in enumerate(hidden_dims):
            if dim != prev_dim:
                self.blocks.append(ResidualGCNBlockProj(prev_dim, dim, dropout))
            else:
                self.blocks.append(ResidualGCNBlock(dim, dropout))
            prev_dim = dim

        num_blocks = len(hidden_dims)

        # ── 多尺度融合 ──
        # 拼接 block[1], block[3], block[5]（浅/中/深层）
        self.jk_indices = [1, 3, 5]  # 0-indexed
        jk_total_dim = sum(hidden_dims[i] for i in self.jk_indices)
        self.jk_proj = nn.Sequential(
            nn.Linear(jk_total_dim, hidden_dims[-1]),
            LayerNorm1d(hidden_dims[-1]),
        )

        # ── 注意力读出 ──
        self.readout = AttentionReadout(hidden_dims[-1])

        # ── 左右手编码 ──
        self.hand_encoder = nn.Sequential(
            nn.Linear(2, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, hidden_dims[-1]),
        )

        # ── FiLM 融合 ──
        self.film = FiLMFusion(hidden_dims[-1], hidden_dims[-1])

        # ── 分类器 ──
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dims[-1], hidden_dims[-1] * 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dims[-1] * 2, hidden_dims[-1]),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dims[-1], num_classes),
        )

        self.dropout = dropout

        # 缓存 edges 以便 compute_bone_features 使用
        self._edges = None

    def set_edges(self, edges):
        """设置骨骼边列表，供 bone feature 计算使用。"""
        self._edges = edges

    def forward(self, x: torch.Tensor, adj: torch.Tensor,
                handedness: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x:           (B, 21, 3)   关键点坐标
            adj:         (21, 21)     归一化邻接矩阵
            handedness:  (B, 2)       左右手 one-hot
        Returns:
            (B, num_classes)  logits
        """
        B = x.size(0)

        # ── 1. 骨骼特征 ──
        if self._edges is not None:
            bone = compute_bone_features(x, self._edges)   # (B, 21, 4)
            x_enriched = torch.cat([x, bone], dim=-1)       # (B, 21, 7)
        else:
            # 无 edges 时用零填充
            x_enriched = F.pad(x, (0, 4))                   # (B, 21, 7)

        # ── 2. 输入投影 ──
        h = self.input_proj(x_enriched)                     # (B, 21, 64)

        # ── 3. 残差 GCN 块 + 多尺度收集 ──
        jk_features = []
        for idx, block in enumerate(self.blocks):
            h = block(h, adj)
            if idx in self.jk_indices:
                jk_features.append(h)

        # ── 4. 多尺度融合 ──
        h_jk = torch.cat(jk_features, dim=-1)               # (B, 21, sum_dims)
        h = self.jk_proj(h_jk)                               # (B, 21, 64)

        # ── 5. 注意力读出 ──
        h_graph = self.readout(h)                            # (B, 64)

        # ── 6. 左右手 FiLM 调制 ──
        hand_feat = self.hand_encoder(handedness)            # (B, 64)
        h_fused = self.film(h_graph, hand_feat)              # (B, 64)

        # ── 7. 分类 ──
        out = self.classifier(h_fused)                       # (B, 10)
        return out


def create_model(device: torch.device, num_classes: int = 10,
                 hidden_dims: tuple = (64, 128, 256, 256, 128, 64),
                 dropout: float = 0.35) -> HandGCN:
    """工厂函数：创建 HandGCN++ 并移至目标设备。"""
    model = HandGCN(
        num_classes=num_classes,
        hidden_dims=hidden_dims,
        dropout=dropout,
    )
    # 导入边列表设置到模型
    from dataset_metadata import HAND_EDGES
    model.set_edges(HAND_EDGES)
    return model.to(device)
