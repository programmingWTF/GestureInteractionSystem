"""
GCN 数据集加载器

从 GCN/DateSet/ 目录读取 CSV 文件，转为 PyTorch Dataset。
每个样本：21 个手部关键点 (x,y,z) + 左右手标签 + 手势类别标签。
"""

import os
import csv
import numpy as np
import torch
from torch.utils.data import Dataset

# 导入元数据
from dataset_metadata import (
    GESTURES, NUM_CLASSES, NUM_LANDMARKS,
)

DATA_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "DateSet")


class HandGestureDataset(Dataset):
    """
    手部关键点手势数据集。

    每个样本:
        landmarks:   (21, 3)  float32  关键点坐标
        handedness:  (2,)     float32  左右手 one-hot [left, right]
        label:       int                手势类别 0..9
    """

    def __init__(self, samples: list):
        """
        Args:
            samples: [(landmarks_21x3, handedness_idx, label), ...]
        """
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        landmarks, handedness_idx, label = self.samples[idx]

        # landmarks: (21, 3) float32
        landmarks = torch.from_numpy(landmarks).float()

        # handedness: one-hot (2,)
        hand_vec = torch.zeros(2)
        hand_vec[handedness_idx] = 1.0

        # label: long
        label = torch.tensor(label, dtype=torch.long)

        return landmarks, hand_vec, label


def load_all_data(data_root: str = None) -> list:
    """
    从 DateSet 目录加载所有 CSV 数据。

    返回:
        samples: [(landmarks_21x3, handedness_idx, label), ...]
        其中 handedness_idx: 0=left, 1=right

    如果数据目录为空，返回空列表。
    """
    if data_root is None:
        data_root = DATA_ROOT

    samples = []

    if not os.path.isdir(data_root):
        print(f"  [WARN] 数据目录不存在: {data_root}")
        return samples

    for gid, gname, gdisplay in GESTURES:
        csv_path = os.path.join(data_root, gname, f"{gname}.csv")
        if not os.path.exists(csv_path):
            print(f"  [WARN] 缺少数据: {gname} ({csv_path})")
            continue

        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                # 解析关键点坐标
                landmarks = np.zeros((NUM_LANDMARKS, 3), dtype=np.float32)
                for i in range(NUM_LANDMARKS):
                    landmarks[i, 0] = float(row[f"lm{i}_{LANDMARK_NAMES[i]}_x"])
                    landmarks[i, 1] = float(row[f"lm{i}_{LANDMARK_NAMES[i]}_y"])
                    landmarks[i, 2] = float(row[f"lm{i}_{LANDMARK_NAMES[i]}_z"])

                # 解析左右手
                handedness_raw = row.get("handedness", "right").strip().lower()
                handedness_idx = 0 if handedness_raw == "left" else 1

                samples.append((landmarks, handedness_idx, gid))

        n = len([s for s in samples if s[2] == gid])
        # Count from this specific gesture - let's just report total
        pass

    return samples


def get_train_val_split(samples: list, val_ratio: float = 0.2, seed: int = 42):
    """
    按类别**时序**划分训练集和验证集。

    数据是按采集顺序排列的（同一会话中相邻帧高度相似）。
    随机划分会导致几乎相同的帧同时出现在训练集和验证集（数据泄漏），
    造成验证准确率虚高但真实场景表现差。

    改为时序划分：每类前 80% 帧用于训练，后 20% 用于验证。
    验证集来自采集会话的末期，手的姿态/角度已有漂移，更接近真实泛化场景。
    """
    train_samples = []
    val_samples = []

    for c in range(NUM_CLASSES):
        # 同类数据按 CSV 中原始顺序（已按时序排列）
        class_samples = [s for s in samples if s[2] == c]
        if len(class_samples) == 0:
            continue

        n_val = max(1, int(len(class_samples) * val_ratio))
        train_samples.extend(class_samples[:-n_val])       # 前 80%
        val_samples.extend(class_samples[-n_val:])         # 后 20%（末期）

    # 打乱训练集（验证集保持时序，不洗牌）
    rng = np.random.RandomState(seed)
    rng.shuffle(train_samples)

    return HandGestureDataset(train_samples), HandGestureDataset(val_samples)


def print_data_stats(samples: list):
    """打印数据集统计信息。"""
    if not samples:
        print("  数据集为空！请先运行 collect_data.py 采集数据。")
        return

    by_class = {}
    by_hand = {"left": 0, "right": 0}
    for lm, h_idx, label in samples:
        by_class[label] = by_class.get(label, 0) + 1
        hand = "left" if h_idx == 0 else "right"
        by_hand[hand] += 1

    print("\n" + "=" * 55)
    print(f"  数据集统计（总计 {len(samples)} 帧）")
    print("=" * 55)
    for gid, gname, gdisplay in GESTURES:
        n = by_class.get(gid, 0)
        bar = "█" * min(n // 3, 25)
        print(f"  [{gid:02d}] {gname:<20s} {n:>5d}  {bar}")
    print("─" * 55)
    print(f"  左手: {by_hand['left']:>5d}    右手: {by_hand['right']:>5d}")
    print("=" * 55)


# Re-import needed for CSV column access
from dataset_metadata import LANDMARK_NAMES


if __name__ == "__main__":
    # 独立运行时：加载并打印统计
    samples = load_all_data()
    print_data_stats(samples)
