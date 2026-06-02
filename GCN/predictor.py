"""
GCN 手势预测器 — 轻量推理封装

用法:
    predictor = GCNPredictor("GCN/best_model.pth")
    gesture, confidence = predictor.predict(landmarks, handedness)
"""

import os
import sys

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

import torch
import numpy as np

from model import HandGCN, build_adj_matrix
from dataset_metadata import HAND_EDGES, GESTURES

# 手势名称映射
IDX_TO_NAME = {g[0]: g[2] for g in GESTURES}   # 0 → "手掌张开"
IDX_TO_EN   = {g[0]: g[1] for g in GESTURES}   # 0 → "open_palm"


class GCNPredictor:
    """GCN 手势预测器 — 加载训练好的模型，对单帧关键点做推理。"""

    def __init__(self, model_path: str, device: str = None,
                 ema_alpha: float = 0.75):
        """
        Args:
            ema_alpha: 时序平滑系数。0=只看历史, 1=只看当前帧。
                       0.6 表示 60% 当前帧 + 40% 历史累积。
        """
        if device is None:
            self.device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )
        else:
            self.device = torch.device(device)

        # 构建邻接矩阵
        self.adj = build_adj_matrix(HAND_EDGES, 21).to(self.device)

        # 创建模型 & 加载权重
        self.model = HandGCN(num_classes=len(GESTURES)).to(self.device)
        self.model.set_edges(HAND_EDGES)   # ← 之前漏了！骨骼特征一直是零
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()

        # 时序平滑 — 为每只手腕维护一个 EMA 概率向量
        self.ema_alpha = ema_alpha
        self._ema_probs = {}   # hand_id → np.array(10,)

        acc = checkpoint.get("val_acc", 0)
        print(f"  ✓ GCN 模型已加载: {os.path.basename(model_path)} "
              f"(val_acc={acc:.1%}, ema={ema_alpha}, device={self.device})")

    def _predict_raw(self, landmarks, handedness: str):
        """单帧原始推理（不做平滑）。"""
        coords = np.zeros((21, 3), dtype=np.float32)
        for i, lm in enumerate(landmarks):
            coords[i, 0] = lm.x
            coords[i, 1] = lm.y
            coords[i, 2] = lm.z

        x = torch.from_numpy(coords).unsqueeze(0).to(self.device)
        hand_vec = torch.zeros(1, 2).to(self.device)
        hand_vec[0, 1 if handedness == "Right" else 0] = 1.0

        with torch.no_grad():
            logits = self.model(x, self.adj, hand_vec)
            probs = torch.softmax(logits, dim=-1).squeeze(0).cpu().numpy()

        return probs

    def predict(self, landmarks, handedness: str,
                hand_id: str = None):
        """
        对单帧手部关键点做预测（含时序平滑）。

        对每只手腕维护一个 EMA 概率向量：
            p_smooth = α * p_current + (1-α) * p_history

        这样相邻帧之间不会因为噪声导致手势反复跳变。

        Args:
            landmarks:  MediaPipe 的 21 个 NormalizedLandmark 列表
            handedness: "Left" 或 "Right"
            hand_id:    手部唯一标识，用于区分左右手的平滑状态。
                        传 None 时退化为单帧预测（不跨帧平滑）。
        Returns:
            (gesture_name_cn: str, confidence: float, probs: list)
        """
        probs_raw = self._predict_raw(landmarks, handedness)

        # 时序平滑
        hid = hand_id
        if hid is not None:
            if hid in self._ema_probs:
                prev = self._ema_probs[hid]
                probs_smooth = self.ema_alpha * probs_raw + (1 - self.ema_alpha) * prev
            else:
                probs_smooth = probs_raw
            self._ema_probs[hid] = probs_smooth
        else:
            probs_smooth = probs_raw

        pred_idx = int(probs_smooth.argmax())
        confidence = float(probs_smooth[pred_idx])
        return IDX_TO_NAME.get(pred_idx, "未知"), confidence, probs_smooth.tolist()

    def predict_top_k(self, landmarks, handedness: str,
                      hand_id: str = None, k: int = 3):
        """返回 top-k 预测结果。"""
        name, conf, probs = self.predict(landmarks, handedness, hand_id=hand_id)
        ranked = sorted(enumerate(probs), key=lambda x: -x[1])[:k]
        return [(IDX_TO_NAME.get(i, "?"), p) for i, p in ranked]
