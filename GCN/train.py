"""
GCN gesture recognition training script.
Usage: python GCN/train.py
"""

import os
import sys
import time
import math
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, classification_report
from tqdm import tqdm

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

from dataset_metadata import GESTURES, NUM_CLASSES, HAND_EDGES, NUM_LANDMARKS
from dataset import load_all_data, get_train_val_split, print_data_stats
from model import build_adj_matrix, create_model

CONFIG = {
    "batch_size": 128,
    "epochs": 400,
    "lr": 2e-3,
    "weight_decay": 1e-5,
    "dropout": 0.35,
    "val_ratio": 0.2,
    "patience": 60,
    "warmup_epochs": 5,
    "label_smoothing": 0.05,
    "num_workers": 0,
}
OUTPUT_DIR = _THIS_DIR


def get_device():
    if torch.cuda.is_available():
        print(f"  ✓ CUDA: {torch.cuda.get_device_name(0)}")
        return torch.device("cuda")
    print("  ✓ CPU")
    return torch.device("cpu")


def augment_landmarks(x):
    """训练时数据增强：绕z轴旋转±8° + 高斯噪声 + 随机缩放±5%。
    所有变换以手腕为中心，保持手势语义不变。"""
    B, N, _ = x.shape
    device = x.device

    # 以手腕为中心
    wrist = x[:, 0:1, :]
    x = x - wrist

    # 旋转（z轴 = 画面内旋转）
    angle = (torch.rand(B, device=device) * 2 - 1) * 0.087  # ±5°
    cos_a, sin_a = torch.cos(angle).view(B, 1), torch.sin(angle).view(B, 1)
    xr = torch.zeros_like(x)
    xr[..., 0] = x[..., 0] * cos_a - x[..., 1] * sin_a
    xr[..., 1] = x[..., 0] * sin_a + x[..., 1] * cos_a
    xr[..., 2] = x[..., 2]

    # 缩放
    scale = 1.0 + (torch.rand(B, device=device) * 2 - 1) * 0.03
    xr = xr * scale.view(B, 1, 1)

    # 噪声
    noise = torch.randn_like(xr) * 0.004
    xr = xr + noise

    return xr + wrist


def train_epoch(model, loader, optimizer, criterion, adj, device, epoch, total_epochs):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    pbar = tqdm(loader, desc=f"Train {epoch}/{total_epochs}", unit="batch", leave=False)
    for landmarks, hand_vec, labels in pbar:
        landmarks = landmarks.to(device)
        landmarks = augment_landmarks(landmarks)
        hand_vec = hand_vec.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        logits = model(landmarks, adj, hand_vec)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        bc = (logits.argmax(1) == labels).sum().item()
        total_loss += loss.item() * landmarks.size(0)
        correct += bc; total += labels.size(0)
        pbar.set_postfix(loss=f"{loss.item():.3f}", acc=f"{bc/landmarks.size(0):.1%}")
    return total_loss / total, correct / total


@torch.no_grad()
def validate(model, loader, criterion, adj, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []
    pbar = tqdm(loader, desc="Validate", unit="batch", leave=False)
    for landmarks, hand_vec, labels in pbar:
        landmarks = landmarks.to(device)
        hand_vec = hand_vec.to(device)
        labels = labels.to(device)
        logits = model(landmarks, adj, hand_vec)
        loss = criterion(logits, labels)
        total_loss += loss.item() * landmarks.size(0)
        preds = logits.argmax(1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())
    acc = correct / total
    cm = confusion_matrix(all_labels, all_preds, labels=list(range(NUM_CLASSES)))
    return total_loss / total, acc, cm


def plot_curves(history, path):
    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.rcParams['font.sans-serif'] = ['SimHei']
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        axes[0].plot(history["train_loss"], label="Train", color="#2196F3")
        axes[0].plot(history["val_loss"], label="Val", color="#FF5722")
        axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss")
        axes[0].legend(); axes[0].grid(True, alpha=0.3)
        axes[1].plot(history["train_acc"], label="Train", color="#2196F3")
        axes[1].plot(history["val_acc"], label="Val", color="#FF5722")
        axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Accuracy")
        axes[1].legend(); axes[1].grid(True, alpha=0.3)
        best = int(np.argmax(history["val_acc"]))
        ba = history["val_acc"][best]
        axes[1].annotate(f"Best: {ba*100:.1f}% @ epoch {best+1}",
                         xy=(best, ba), xytext=(best + 3, ba - 0.08),
                         arrowprops=dict(arrowstyle="->", color="green"),
                         fontsize=10, color="green")
        plt.tight_layout(); plt.savefig(path, dpi=150); plt.close()
    except ImportError:
        pass


def plot_confusion(cm, path, class_names):
    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.rcParams['font.sans-serif'] = ['SimHei']
        fig, ax = plt.subplots(figsize=(9, 8))
        im = ax.imshow(cm, cmap="Blues")
        ax.set_xticks(range(len(class_names))); ax.set_yticks(range(len(class_names)))
        ax.set_xticklabels(class_names, rotation=45, ha="right", fontsize=8)
        ax.set_yticklabels(class_names, fontsize=8)
        ax.set_xlabel("Predicted"); ax.set_ylabel("True")
        for i in range(len(class_names)):
            for j in range(len(class_names)):
                ax.text(j, i, cm[i, j], ha="center", va="center",
                        fontsize=7, color="white" if cm[i, j] > cm.max()/2 else "black")
        plt.colorbar(im, ax=ax)
        plt.tight_layout(); plt.savefig(path, dpi=150); plt.close()
    except ImportError:
        pass


def main():
    print("=" * 55)
    print("  GCN 手势识别 — 训练 v2.5")
    print("=" * 55)
    device = get_device()
    print()

    samples = load_all_data()
    print_data_stats(samples)
    if len(samples) < 20:
        print("  ❌ 数据不足"); return
    train_set, val_set = get_train_val_split(samples, CONFIG["val_ratio"])
    print(f"\n  训练: {len(train_set)}  验证: {len(val_set)}")
    train_loader = DataLoader(train_set, CONFIG["batch_size"], shuffle=True,
                              num_workers=CONFIG["num_workers"], drop_last=True)
    val_loader = DataLoader(val_set, CONFIG["batch_size"], shuffle=False,
                            num_workers=CONFIG["num_workers"])

    adj = build_adj_matrix(HAND_EDGES, NUM_LANDMARKS).to(device)
    model = create_model(device, NUM_CLASSES)
    print(f"  参数量: {sum(p.numel() for p in model.parameters()):,}")

    criterion = nn.CrossEntropyLoss(label_smoothing=CONFIG["label_smoothing"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG["lr"],
                                  weight_decay=CONFIG["weight_decay"])
    # Cosine annealing with linear warmup (per-step)
    warmup_steps = CONFIG["warmup_epochs"] * len(train_loader)
    total_steps = CONFIG["epochs"] * len(train_loader)

    def lr_fn(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_fn)
    global_step = 0


    print("\n" + "=" * 55 + "\n  开始训练\n" + "=" * 55)
    hdr = f"  {'Ep':>4s}  {'Train Loss':>10s}  {'Train Acc':>9s}  {'Val Loss':>8s}  {'Val Acc':>7s}  {'LR':>8s}  {'Time':>5s}"
    print(hdr + "\n  " + "─" * 62)
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_acc, best_ep, patience_ctr = 0.0, 0, 0
    best_path = os.path.join(OUTPUT_DIR, "best_model.pth")
    t0 = time.time()

    for ep in range(1, CONFIG["epochs"] + 1):
        t_ep = time.time()
        tl, ta = train_epoch(model, train_loader, optimizer, criterion, adj, device, ep, CONFIG["epochs"])
        vl, va, _ = validate(model, val_loader, criterion, adj, device)
        # step-based cosine annealing
        for _ in range(len(train_loader)):
            global_step += 1
            scheduler.step()
        lr = optimizer.param_groups[0]["lr"]

        history["train_loss"].append(tl); history["train_acc"].append(ta)
        history["val_loss"].append(vl); history["val_acc"].append(va)

        print(f"  {ep:>4d}  {tl:>10.4f}  {ta:>8.2%}  {vl:>8.4f}  {va:>7.2%}  {lr:>7.1e}  {time.time()-t_ep:>4.1f}s")
        if va > best_acc:
            best_acc, best_ep, patience_ctr = va, ep, 0
            torch.save({"epoch": ep, "model_state_dict": model.state_dict(),
                        "val_acc": va, "config": CONFIG}, best_path)
            print(f"         ↑ best ({va:.2%})")
        else:
            patience_ctr += 1
        if patience_ctr >= CONFIG["patience"]:
            print(f"\n  ⏹ Early stop @ {ep}")
            break

    total_t = time.time() - t0
    print(f"\n  Done: {total_t:.0f}s  |  Best: {best_acc:.2%} at epoch {best_ep}")

    ckpt = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    _, final_acc, _ = validate(model, val_loader, criterion, adj, device)

    model.eval(); all_preds, all_labels = [], []
    with torch.no_grad():
        for landmarks, hand_vec, labels in val_loader:
            logits = model(landmarks.to(device), adj, hand_vec.to(device))
            all_preds.extend(logits.argmax(1).cpu().tolist())
            all_labels.extend(labels.tolist())

    cm = confusion_matrix(all_labels, all_preds)
    gesture_names = [g[1] for g in GESTURES]  # English: open_palm, fist, ...
    n_params = sum(p.numel() for p in model.parameters())

    # Paper-ready outputs
    plot_curves(history, os.path.join(OUTPUT_DIR, "training_curves.png"))
    plot_confusion(cm, os.path.join(OUTPUT_DIR, "confusion_matrix.png"), gesture_names)

    en_names = [g[1] for g in GESTURES]
    report = classification_report(all_labels, all_preds,
                                   target_names=en_names, zero_division=0)
    with open(os.path.join(OUTPUT_DIR, "classification_report.txt"), "w") as f:
        f.write(report)

    import json
    info = {
        "model": "HandGCN",
        "parameters": n_params,
        "input_features": "8 per node (xyz+hand+bone*4)",
        "hidden_dim": 160,
        "layers": 3,
        "training_time_s": round(total_t, 1),
        "best_val_acc": round(best_acc, 4),
        "best_epoch": best_ep,
        "epochs_trained": ep,
        "classification_report": report,
        "history": history,
        "config": {k: v for k, v in CONFIG.items() if not callable(v)},
    }
    with open(os.path.join(OUTPUT_DIR, "training_log.json"), "w") as f:
        json.dump(info, f, indent=2, ensure_ascii=False)

    print(f"\n  Final accuracy: {final_acc:.2%}")
    print(f"  Outputs saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
