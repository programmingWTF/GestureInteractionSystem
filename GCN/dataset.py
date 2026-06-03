"""
PyTorch Dataset: loads CSV files from GCN/DataSet/ into (landmarks, handedness, label) tuples.
"""

import os
import csv
import numpy as np
import torch
from torch.utils.data import Dataset


from dataset_metadata import (
    GESTURES, NUM_CLASSES, NUM_LANDMARKS,
)

DATA_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "DataSet")


class HandGestureDataset(Dataset):
    """
    PyTorch Dataset for hand gesture landmarks.
    """

    def __init__(self, samples: list):
        """

        """
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        landmarks, handedness_idx, label = self.samples[idx]


        landmarks = torch.from_numpy(landmarks).float()


        hand_vec = torch.zeros(2)
        hand_vec[handedness_idx] = 1.0


        label = torch.tensor(label, dtype=torch.long)

        return landmarks, hand_vec, label


def load_all_data(data_root: str = None) -> list:
    """
    Load all CSV data from DataSet/. Returns list of (landmarks, handedness_idx, label).
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

                landmarks = np.zeros((NUM_LANDMARKS, 3), dtype=np.float32)
                for i in range(NUM_LANDMARKS):
                    landmarks[i, 0] = float(row[f"lm{i}_{LANDMARK_NAMES[i]}_x"])
                    landmarks[i, 1] = float(row[f"lm{i}_{LANDMARK_NAMES[i]}_y"])
                    landmarks[i, 2] = float(row[f"lm{i}_{LANDMARK_NAMES[i]}_z"])


                handedness_raw = row.get("handedness", "right").strip().lower()
                handedness_idx = 0 if handedness_raw == "left" else 1

                samples.append((landmarks, handedness_idx, gid))



    return samples


def get_train_val_split(samples: list, val_ratio: float = 0.2, seed: int = 42,
                        method: str = "stratified"):
    """
    Train/validation split.

    Args:
        method: "stratified" — random split stratified by class + handedness
                "chronological" — first 80% train, last 20% val (no leakage,
                but val may drift if hand position changes during recording)
    """
    from sklearn.model_selection import train_test_split

    if method == "chronological":
        train_samples, val_samples = [], []
        for c in range(NUM_CLASSES):
            class_samples = [s for s in samples if s[2] == c]
            if len(class_samples) == 0:
                continue
            n_val = max(1, int(len(class_samples) * val_ratio))
            train_samples.extend(class_samples[:-n_val])
            val_samples.extend(class_samples[-n_val:])
        rng = np.random.RandomState(seed)
        rng.shuffle(train_samples)
    else:
        # Stratified by (class, handedness)
        groups = [(s[2], s[1]) for s in samples]  # (label, hand_idx)
        train_idx, val_idx = train_test_split(
            range(len(samples)), test_size=val_ratio,
            stratify=groups, random_state=seed)
        train_samples = [samples[i] for i in train_idx]
        val_samples = [samples[i] for i in val_idx]

    return HandGestureDataset(train_samples), HandGestureDataset(val_samples)


def print_data_stats(samples: list):
    """Print per-class and per-hand data statistics."""
    if not samples:
        print("  Dataset is empty. Run collect_data.py first.")
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


from dataset_metadata import LANDMARK_NAMES


if __name__ == "__main__":
    samples = load_all_data()
    print_data_stats(samples)
