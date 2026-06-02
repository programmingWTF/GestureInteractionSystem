"""
Lightweight GCN inference wrapper.
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

IDX_TO_NAME = {g[0]: g[2] for g in GESTURES}
IDX_TO_EN   = {g[0]: g[1] for g in GESTURES}


class GCNPredictor:
    """GCN gesture predictor: loads a trained model and runs per-frame inference."""

    def __init__(self, model_path: str, device: str = None,
                 ema_alpha: float = 0.75):
        """
        Args:
            ema_alpha: Temporal smoothing coefficient.
        """
        if device is None:
            self.device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )
        else:
            self.device = torch.device(device)


        self.adj = build_adj_matrix(HAND_EDGES, 21).to(self.device)


        self.model = HandGCN(num_classes=len(GESTURES)).to(self.device)
        self.model.set_edges(HAND_EDGES)
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()

        # Per-hand EMA probability vectors
        self.ema_alpha = ema_alpha
        self._ema_probs = {}   # hand_id → np.array(10,)

        acc = checkpoint.get("val_acc", 0)
        print(f"  ✓ GCN 模型已加载: {os.path.basename(model_path)} "
              f"(val_acc={acc:.1%}, ema={ema_alpha}, device={self.device})")

    def _predict_raw(self, landmarks, handedness: str):
        """Raw single-frame inference (no smoothing)."""
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
        Predict gesture with EMA temporal smoothing.

        Maintains per-hand EMA of probability vector:
            p_smooth = alpha * p_current + (1-alpha) * p_history

        Args:
            landmarks:  21 MediaPipe NormalizedLandmark objects
            handedness: "Left" or "Right"
            hand_id:    Per-hand identifier for smoothing state.
                        Pass None for frame-independent prediction.
        Returns:
            (gesture_name, confidence, probability_distribution)
        """
        probs_raw = self._predict_raw(landmarks, handedness)


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
        """Return top-k predictions."""
        name, conf, probs = self.predict(landmarks, handedness, hand_id=hand_id)
        ranked = sorted(enumerate(probs), key=lambda x: -x[1])[:k]
        return [(IDX_TO_NAME.get(i, "?"), p) for i, p in ranked]
