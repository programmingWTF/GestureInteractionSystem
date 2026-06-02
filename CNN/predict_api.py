"""
CNN 预测接口 - 内置于 GestureInteractionSystem 项目

用法：python cnn/predict_api.py <image_path>
输出：JSON {"label","display","confidence","probs":[...]}

模型：EnhancedCNN (10类数字 0-9)
依赖：torch, pillow, numpy (在项目 conda 环境中安装)
"""

import os, sys, json
import torch, torch.nn as nn
import numpy as np
from PIL import Image, ImageOps


class EnhancedCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1); self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 32, 3, padding=1); self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1); self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 64, 3, padding=1); self.bn4 = nn.BatchNorm2d(64)
        self.conv5 = nn.Conv2d(64, 128, 3, padding=1); self.bn5 = nn.BatchNorm2d(128)
        self.pool = nn.MaxPool2d(2, 2)
        self.global_pool = nn.AdaptiveAvgPool2d((4, 4))
        self.fc1 = nn.Linear(128 * 4 * 4, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 10)
        self.dropout = nn.Dropout(0.5)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x))); x = self.pool(x)
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.relu(self.bn4(self.conv4(x))); x = self.pool(x)
        x = self.relu(self.bn5(self.conv5(x)))
        x = self.global_pool(x); x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x)); x = self.dropout(x)
        x = self.relu(self.fc2(x)); x = self.dropout(x)
        x = self.fc3(x)
        return x


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(SCRIPT_DIR, "qmnist_digit_model.pth")


def get_device():
    try:
        import torch_directml as dml
        return dml.device()
    except Exception:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model(device):
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"模型不存在: {MODEL_PATH}")
    model = EnhancedCNN()
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    model.to(device)
    model.eval()
    return model


def preprocess(img_path):
    img = Image.open(img_path).convert("L")
    img = ImageOps.invert(img)
    img = img.resize((28, 28), Image.LANCZOS)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    arr = (arr - 0.1307) / 0.3081
    return torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)


def predict(image_path, device=None):
    if device is None:
        device = get_device()
    model = load_model(device)
    x = preprocess(image_path).to(device)
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()
    best = int(np.argmax(probs))
    return {
        "label": str(best), "display": str(best),
        "confidence": round(float(probs[best]), 4),
        "probs": [round(float(p), 4) for p in probs],
    }


if __name__ == "__main__":
    if len(sys.argv) < 2 or not os.path.exists(sys.argv[1]):
        print(json.dumps({"error": "Usage: predict_api.py <image>"}))
        sys.exit(1)
    try:
        print(json.dumps(predict(sys.argv[1])))
    except Exception as e:
        print(json.dumps({"error": str(e)}))
        sys.exit(1)
