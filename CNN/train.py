"""
CNN digit classifier — training script v2.

Fixes:
  - Data augmentation simulating air-writing (rotation <=10 deg, NO flip)
  - Deeper model (3 conv layers instead of 2)
  - Train/val transform separated (augmentation only for training)

Dataset: QMNIST (60K train, 10K test)
"""

import os, sys, time, math, json
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.metrics import confusion_matrix, classification_report
from tqdm import tqdm

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = _THIS_DIR

CONFIG = {
    "batch_size": 4096,
    "epochs": 40,
    "lr": 4e-3,
    "weight_decay": 1e-4,
    "dropout": 0.4,
    "patience": 10,
    "num_workers": 0,
}


class DigitCNN(nn.Module):
    """Conv1 -> Conv2 -> Conv3 -> FC1 -> FC2"""

    def __init__(self, num_classes=10, dropout=0.4):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)       # 32x32 -> 32x32
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)      # 32x32 -> 32x32
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)     # 32x32 -> 32x32
        self.pool = nn.MaxPool2d(2, 2)                     # -> 16x16 -> 8x8 -> 4x4
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(128 * 4 * 4, 256)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))    # (B,32,16,16)
        x = self.pool(F.relu(self.bn2(self.conv2(x))))    # (B,64,8,8)
        x = self.pool(F.relu(self.bn3(self.conv3(x))))    # (B,128,4,4)
        x = x.view(x.size(0), -1)                          # (B,2048)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)


def get_device():
    if torch.cuda.is_available():
        print(f"  CUDA: {torch.cuda.get_device_name(0)}")
        return torch.device("cuda")
    print("  CPU")
    return torch.device("cpu")


def get_dataloaders(data_root, batch_size, num_workers):
    # Training: augment to simulate air-writing variation
    train_transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.Grayscale(num_output_channels=3),
        transforms.RandomAffine(degrees=10, translate=(0.12, 0.12), scale=(0.85, 1.15)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    # Validation: clean, no augmentation
    val_transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    train_set = datasets.QMNIST(
        root=data_root, what="train", download=True, transform=train_transform)
    val_set = datasets.QMNIST(
        root=data_root, what="test", download=True, transform=val_transform)

    train_loader = DataLoader(train_set, batch_size=batch_size,
                              shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=batch_size,
                            shuffle=False, num_workers=num_workers, pin_memory=True)
    return train_loader, val_loader


def train_epoch(model, loader, optimizer, criterion, device, epoch, total_epochs):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    pbar = tqdm(loader, desc=f"Train {epoch}/{total_epochs}", unit="b", leave=False)
    for x, y in pbar:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        bc = (logits.argmax(1) == y).sum().item()
        total_loss += loss.item() * x.size(0)
        correct += bc; total += y.size(0)
        pbar.set_postfix(loss=f"{loss.item():.3f}", acc=f"{bc/x.size(0):.1%}")
    return total_loss / total, correct / total


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []
    pbar = tqdm(loader, desc="Validate", unit="b", leave=False)
    for x, y in pbar:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        total_loss += criterion(logits, y).item() * x.size(0)
        preds = logits.argmax(1)
        correct += (preds == y).sum().item()
        total += y.size(0)
        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(y.cpu().tolist())
    return total_loss / total, correct / total, all_preds, all_labels


def plot_curves(history, path):
    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        axes[0].plot(history["train_loss"], label="Train")
        axes[0].plot(history["val_loss"], label="Val")
        axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss")
        axes[0].legend(); axes[0].grid(True, alpha=0.3)
        axes[1].plot(history["train_acc"], label="Train")
        axes[1].plot(history["val_acc"], label="Val")
        axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Accuracy")
        axes[1].legend(); axes[1].grid(True, alpha=0.3)
        best = int(np.argmin(history["val_loss"]))
        bl = history["val_loss"][best]
        ba = history["val_acc"][best]
        axes[0].annotate(f"Best loss: {bl:.4f} @ epoch {best+1}",
                         xy=(best, bl), xytext=(best + 3, bl + 0.05),
                         arrowprops=dict(arrowstyle="->", color="green"), fontsize=9, color="green")
        axes[1].annotate(f"Acc at best loss: {ba:.1%}",
                         xy=(best, ba), xytext=(best + 3, ba - 0.08),
                         arrowprops=dict(arrowstyle="->", color="green"), fontsize=9, color="green")
        plt.tight_layout(); plt.savefig(path, dpi=150); plt.close()
    except ImportError: pass


def plot_confusion(cm, path, class_names):
    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(8, 7))
        im = ax.imshow(cm, cmap="Blues")
        ax.set_xticks(range(len(class_names))); ax.set_yticks(range(len(class_names)))
        ax.set_xticklabels(class_names); ax.set_yticklabels(class_names)
        ax.set_xlabel("Predicted"); ax.set_ylabel("True")
        for i in range(len(class_names)):
            for j in range(len(class_names)):
                ax.text(j, i, cm[i, j], ha="center", va="center",
                        fontsize=8, color="white" if cm[i, j] > cm.max()/2 else "black")
        plt.colorbar(im, ax=ax)
        plt.tight_layout(); plt.savefig(path, dpi=150); plt.close()
    except ImportError: pass


def main():
    print("=" * 55)
    print("  CNN Digit Classifier Training v2")
    print("=" * 55)
    device = get_device()
    if device.type == "cuda":
        print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory/1024**3:.0f} GB")
    print()

    data_root = os.path.join(os.path.dirname(_THIS_DIR), "data")
    train_loader, val_loader = get_dataloaders(
        data_root, CONFIG["batch_size"], CONFIG["num_workers"])
    print(f"  Train: {len(train_loader.dataset):,}  Test: {len(val_loader.dataset):,}")

    model = DigitCNN(dropout=CONFIG["dropout"]).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {n_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG["lr"],
                                  weight_decay=CONFIG["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CONFIG["epochs"])
    criterion = nn.CrossEntropyLoss()

    hdr = (f"  {'Ep':>4s}  {'Train Loss':>10s}  {'Train Acc':>9s}"
           f"  {'Val Loss':>8s}  {'Val Acc':>7s}  {'LR':>8s}  {'Time':>5s}")
    print("\n" + hdr + "\n  " + "-" * 60)

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_loss = float('inf'); best_epoch = 0; patience_ctr = 0
    best_path = os.path.join(OUTPUT_DIR, "qmnist_digit_model.pth")
    t_start = time.time()

    for ep in range(1, CONFIG["epochs"] + 1):
        t_ep = time.time()
        tl, ta = train_epoch(model, train_loader, optimizer, criterion, device, ep, CONFIG["epochs"])
        vl, va, preds, labels = validate(model, val_loader, criterion, device)
        scheduler.step()

        history["train_loss"].append(tl); history["train_acc"].append(ta)
        history["val_loss"].append(vl); history["val_acc"].append(va)

        print(f"  {ep:>4d}  {tl:>10.4f}  {ta:>8.2%}  {vl:>8.4f}"
              f"  {va:>7.2%}  {optimizer.param_groups[0]['lr']:>7.1e}"
              f"  {time.time()-t_ep:>4.1f}s")

        if vl < best_loss:
            best_loss = vl; best_epoch = ep; patience_ctr = 0
            torch.save(model.state_dict(), best_path)
            print(f"         best loss ({vl:.4f}, acc={va:.2%})")
        else:
            patience_ctr += 1
        if patience_ctr >= CONFIG["patience"]:
            print(f"\n  Early stop at epoch {ep}"); break

    total_t = time.time() - t_start
    print(f"\n  Done: {total_t:.0f}s  |  Best loss: {best_loss:.4f} at epoch {best_epoch}")

    model.load_state_dict(torch.load(best_path, map_location=device))
    _, final_acc, preds, labels = validate(model, val_loader, criterion, device)
    cm = confusion_matrix(labels, preds)

    plot_curves(history, os.path.join(OUTPUT_DIR, "training_curves.png"))
    plot_confusion(cm, os.path.join(OUTPUT_DIR, "confusion_matrix.png"),
                   [str(i) for i in range(10)])

    report = classification_report(labels, preds, target_names=[str(i) for i in range(10)])
    with open(os.path.join(OUTPUT_DIR, "classification_report.txt"), "w") as f:
        f.write(report)

    info = {
        "model": "DigitCNN-v2",
        "parameters": n_params,
        "architecture": "Conv1(3->32)->Pool->Conv2(32->64)->Pool->Conv3(64->128)->Pool->FC(2048->256)->FC(256->10)",
        "augmentation": "RandomAffine(deg=10, translate=12%, scale=85-115%), NO flip",
        "dataset": "QMNIST (60K train, 10K test)",
        "training_time_s": round(total_t, 1),
        "best_val_loss": round(best_loss, 4),
        "best_val_acc": round(final_acc, 4),
        "best_epoch": best_epoch,
        "epochs_trained": ep,
        "classification_report": report,
        "history": history,
    }
    with open(os.path.join(OUTPUT_DIR, "training_log.json"), "w") as f:
        json.dump(info, f, indent=2, ensure_ascii=False)

    print(f"\n  Final accuracy: {final_acc:.2%}")
    print(f"  Outputs saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
