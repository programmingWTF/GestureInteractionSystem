"""
GCN 手势识别 — 训练脚本

自动检测 CUDA，有 GPU 则用 GPU，无则 CPU。

用法:
    python GCN/train.py

输出:
    GCN/best_model.pth    最佳模型权重
    GCN/training_curves.png  训练曲线图
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

# 将 GCN 目录加入 path，确保可以引用同目录模块
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

from dataset_metadata import GESTURES, NUM_CLASSES, HAND_EDGES, NUM_LANDMARKS
from dataset import load_all_data, get_train_val_split, print_data_stats
from model import HandGCN, build_adj_matrix, create_model


# ══════════════════════════════════════════════════════════
# 配置
# ══════════════════════════════════════════════════════════

CONFIG = {
    "batch_size": 64,
    "epochs": 200,
    "lr": 1e-3,
    "weight_decay": 1e-4,
    "hidden_dims": (64, 128, 256, 256, 128, 64),
    "dropout": 0.35,
    "val_ratio": 0.2,
    "patience": 30,           # early stop
    "lr_patience": 15,        # ReduceLROnPlateau 耐心
    "lr_factor": 0.5,
    "num_workers": 0,
}

OUTPUT_DIR = _THIS_DIR


# ══════════════════════════════════════════════════════════
# 工具函数
# ══════════════════════════════════════════════════════════

def get_device():
    """检测可用设备。"""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"  ✓ CUDA 可用: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print(f"  ✓ 使用 CPU 训练")
    return device


def train_epoch(model, loader, optimizer, criterion, adj, device):
    """单个训练 epoch。"""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for landmarks, hand_vec, labels in loader:
        landmarks = landmarks.to(device)     # (B, 21, 3)
        hand_vec = hand_vec.to(device)       # (B, 2)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(landmarks, adj, hand_vec)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * landmarks.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return total_loss / total, correct / total


@torch.no_grad()
def validate(model, loader, criterion, adj, device):
    """验证。"""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    for landmarks, hand_vec, labels in loader:
        landmarks = landmarks.to(device)
        hand_vec = hand_vec.to(device)
        labels = labels.to(device)

        logits = model(landmarks, adj, hand_vec)
        loss = criterion(logits, labels)

        total_loss += loss.item() * landmarks.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())

    acc = correct / total
    cm = confusion_matrix(all_labels, all_preds, labels=list(range(NUM_CLASSES)))
    return total_loss / total, acc, cm


def plot_curves(history, save_path):
    """绘制训练曲线并保存。"""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Loss
        ax = axes[0]
        ax.plot(history["train_loss"], label="Train Loss", color="#2196F3")
        ax.plot(history["val_loss"], label="Val Loss", color="#FF5722")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_title("Training & Validation Loss")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Accuracy
        ax = axes[1]
        ax.plot(history["train_acc"], label="Train Acc", color="#2196F3")
        ax.plot(history["val_acc"], label="Val Acc", color="#FF5722")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Accuracy")
        ax.set_title("Training & Validation Accuracy")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 标注最佳
        best_epoch = np.argmax(history["val_acc"])
        best_acc = history["val_acc"][best_epoch]
        ax.annotate(f"Best: {best_acc*100:.1f}% @ ep{best_epoch+1}",
                    xy=(best_epoch, best_acc),
                    xytext=(best_epoch + 5, best_acc - 0.05),
                    arrowprops=dict(arrowstyle="->", color="green"),
                    fontsize=10, color="green")

        plt.tight_layout()
        plt.savefig(save_path, dpi=120, bbox_inches="tight")
        plt.close()
        print(f"  📈 训练曲线已保存: {save_path}")
    except ImportError:
        print("  [WARN] matplotlib 未安装，跳过曲线绘制")


# ══════════════════════════════════════════════════════════
# 主函数
# ══════════════════════════════════════════════════════════

def main():
    print("=" * 55)
    print("  GCN 手势识别 — 模型训练")
    print("=" * 55)

    # ── 设备 ──
    device = get_device()
    print()

    # ── 加载数据 ──
    print("  加载数据...")
    samples = load_all_data()
    print_data_stats(samples)

    if len(samples) < 20:
        print("\n  ❌ 数据量太少（<20 帧），无法训练。")
        print("  请先运行 GCN/collect_data.py 采集数据。")
        print("  建议每类手势至少采集 100 帧以上。")
        return

    # ── 划分数据集 ──
    train_set, val_set = get_train_val_split(samples, val_ratio=CONFIG["val_ratio"])
    print(f"\n  训练集: {len(train_set)} 帧 | 验证集: {len(val_set)} 帧")

    train_loader = DataLoader(
        train_set, batch_size=CONFIG["batch_size"], shuffle=True,
        num_workers=CONFIG["num_workers"], drop_last=True,
    )
    val_loader = DataLoader(
        val_set, batch_size=CONFIG["batch_size"], shuffle=False,
        num_workers=CONFIG["num_workers"],
    )

    # ── 构建邻接矩阵 ──
    adj = build_adj_matrix(HAND_EDGES, num_nodes=NUM_LANDMARKS).to(device)
    print(f"  邻接矩阵: {adj.shape}  (21 节点, {len(HAND_EDGES)} 条边)")

    # ── 模型 ──
    model = create_model(
        device,
        num_classes=NUM_CLASSES,
        hidden_dims=CONFIG["hidden_dims"],
        dropout=CONFIG["dropout"],
    )

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  参数量: {trainable_params:,} (总计 {total_params:,})")

    # ── 损失 & 优化器 & 调度器 ──
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=CONFIG["lr"],
        weight_decay=CONFIG["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=CONFIG["lr_factor"],
        patience=CONFIG["lr_patience"],
    )

    # ── 训练循环 ──
    print("\n" + "=" * 55)
    print("  开始训练")
    print("=" * 55)
    print(f"  {'Epoch':>6s}  {'Train Loss':>11s}  {'Train Acc':>10s}"
          f"  {'Val Loss':>9s}  {'Val Acc':>8s}  {'LR':>10s}  {'Time':>7s}")
    print("  " + "─" * 70)

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val_acc = 0.0
    best_epoch = 0
    best_model_path = os.path.join(OUTPUT_DIR, "best_model.pth")
    patience_counter = 0
    t_start = time.time()

    for epoch in range(1, CONFIG["epochs"] + 1):
        t_ep = time.time()

        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, criterion, adj, device,
        )
        val_loss, val_acc, val_cm = validate(
            model, val_loader, criterion, adj, device,
        )

        # 记录
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        # 学习率调度
        current_lr = optimizer.param_groups[0]["lr"]
        scheduler.step(val_loss)

        elapsed = time.time() - t_ep

        # 打印
        print(f"  {epoch:>5d}   {train_loss:>10.4f}  {train_acc:>9.2%}"
              f"  {val_loss:>8.4f}  {val_acc:>7.2%}"
              f"  {current_lr:>9.2e}  {elapsed:>5.1f}s")

        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            patience_counter = 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_acc": val_acc,
                "val_loss": val_loss,
                "config": CONFIG,
                "adj": adj.cpu(),
            }, best_model_path)
            print(f"         ↑ 最佳模型已保存 (val_acc={val_acc:.2%})")
        else:
            patience_counter += 1

        # Early stop
        if patience_counter >= CONFIG["patience"]:
            print(f"\n  ⏹ Early stop @ epoch {epoch}（{CONFIG['patience']} epochs 无提升）")
            break

    total_time = time.time() - t_start
    print("\n" + "=" * 55)
    print(f"  训练完成！总耗时: {total_time:.0f}s ({total_time/60:.1f}min)")
    print(f"  最佳验证准确率: {best_val_acc:.2%} (epoch {best_epoch})")
    print(f"  模型已保存: {best_model_path}")
    print("=" * 55)

    # ── 绘制曲线 ──
    curve_path = os.path.join(OUTPUT_DIR, "training_curves.png")
    plot_curves(history, curve_path)

    # ── 最终评估 ──
    print("\n  加载最佳模型进行最终评估...")
    checkpoint = torch.load(best_model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    _, final_acc, final_cm = validate(model, val_loader, criterion, adj, device)

    gesture_names = [g[2] for g in GESTURES]

    print(f"\n  最终验证准确率: {final_acc:.2%}")

    # 基于验证集重新收集预测
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for landmarks, hand_vec, labels in val_loader:
            landmarks = landmarks.to(device)
            hand_vec = hand_vec.to(device)
            logits = model(landmarks, adj, hand_vec)
            all_preds.extend(logits.argmax(dim=1).cpu().tolist())
            all_labels.extend(labels.tolist())

    if len(set(all_labels)) > 1:
        print("\n  分类报告 (验证集实际预测):")
        print(classification_report(
            all_labels, all_preds,
            target_names=gesture_names,
            zero_division=0,
        ))

    # 混淆矩阵
    print("\n  混淆矩阵:")
    cm = confusion_matrix(all_labels, all_preds, labels=list(range(NUM_CLASSES)))
    header = "        " + "".join(f"{g[2][:4]:>6s}" for g in GESTURES)
    print(header)
    for i, g in enumerate(GESTURES):
        row = "".join(f"{cm[i, j]:>6d}" if i != j else f"\033[92m{cm[i, j]:>6d}\033[0m"
                     for j in range(NUM_CLASSES))
        print(f"  {g[2]:<6s}{row}")

    print(f"\n  ✅ 模型就绪: {best_model_path}")


if __name__ == "__main__":
    main()
