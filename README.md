# GestureInteractionSystem 🖐️

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?logo=pytorch&logoColor=white" alt="PyTorch">
  <img src="https://img.shields.io/badge/OpenCV-4.8+-5C3EE8?logo=opencv&logoColor=white" alt="OpenCV">
  <img src="https://img.shields.io/badge/MediaPipe-0.10+-00C853" alt="MediaPipe">
  <img src="https://img.shields.io/badge/license-Educational-blue" alt="License">
</p>

<p align="center"><b>基于 MediaPipe + GCN 的实时手势识别系统</b></p>
<p align="center">Apple Vision Pro 风格捏合书写交互 · 10 类静态手势分类 · 空中手写 CNN 数字识别</p>

---

## 项目结构

```
GestureInteractionSystem/
│
├── main.py                      # 主程序（摄像头 + GUI + 交互逻辑）
├── gesture_recognizer.py        # 曲率比手势分类器（已弃用）
├── trajectory_recognizer.py     # $1 Unistroke 轨迹识别器（已弃用）
├── hand_landmarker.task         # MediaPipe 手部关键点检测模型 (~7.5 MB)
│
├── CNN/                         # CNN 数字分类器
│   ├── train.py                 # 训练脚本（QMNIST，三层卷积 + 数据增强）
│   ├── predict_api.py           # 推理接口（已弃用，模型由 main.py 直接加载）
│   ├── qmnist_digit_model.pth   # 训练权重
│   ├── training_curves.png      # 训练曲线图
│   ├── confusion_matrix.png     # 混淆矩阵热力图
│   ├── classification_report.txt
│   └── training_log.json        # 完整训练指标
│
├── GCN/                         # GCN 手势分类器
│   ├── collect_data.py          # 数据采集工具（摄像头 → CSV）
│   ├── train.py                 # 训练脚本（时序划分 + 余弦退火）
│   ├── model.py                 # GCN 模型（三层 GCNConv + 骨骼特征）
│   ├── predictor.py             # 推理封装（EMA 平滑 + 1€ 滤波）
│   ├── dataset.py               # PyTorch Dataset（CSV → Tensor）
│   ├── dataset_metadata.py      # 元数据（手势标签、关键点名称、骨骼边）
│   ├── best_model.pth           # 训练权重
│   ├── training_curves.png      # 训练曲线图
│   ├── confusion_matrix.png     # 混淆矩阵热力图
│   ├── classification_report.txt
│   ├── training_log.json        # 完整训练指标
│   ├── DataSet/                 # 采集的训练数据（每类一个 CSV）
│   └── README.md
│
├── data/QMNIST/                 # QMNIST 数据集（自动下载）
├── requirements.txt
└── README.md
```

## 交互逻辑

| 手   | 手势               | 动作                         |
| ---- | ------------------ | ---------------------------- |
| 左手 | OK                 | 进入书写模式（同时清空画布） |
| 左手 | 点赞               | 提交笔画给 CNN 识别数字      |
| 左手 | 手掌张开           | 清空所有笔画（保持书写模式） |
| 右手 | 捏合（拇食指并拢） | 落笔，笔尖跟随两指中点       |
| 右手 | 手掌张开           | 抬笔，结束当前笔画           |
| —    | ESC                | 退出                         |

手势分类由 GCN 模型完成；捏合检测使用拇指尖与食指尖的欧氏距离（阈值 < 画面宽度的 4%）；左手手势带 8 帧防抖锁定。

## 快速开始

```bash
# 安装依赖
pip install opencv-python mediapipe torch numpy Pillow scikit-learn matplotlib tqdm torchvision

# 1. 采集训练数据（可选，项目已附带预采集数据）
python GCN/collect_data.py

# 2. 训练 GCN
python GCN/train.py

# 3. 训练 CNN
python CNN/train.py

# 4. 运行主程序
python main.py
```

强制 CPU 推理：将 `main.py` 顶部的 `_GCN_USE_CUDA = True` 改为 `False`。

## GCN 模型

| 组件     | 说明                                                                      |
| -------- | ------------------------------------------------------------------------- |
| 输入     | 每节点 8 维：归一化 (x,y,z) + 左右手标签 + 4 维骨骼特征 (dx,dy,dz,length) |
| 架构     | GCNConv(8→160) ×3，每层带 LayerNorm + ReLU + Dropout                      |
| 读出     | 全局平均池化 → Linear(160→80) → ReLU → Linear(80→10)                      |
| 参数量   | ~68,000                                                                   |
| 推理速度 | < 0.2 ms/帧 (CPU)                                                         |

**骨骼特征** 对 21 条手部骨骼边计算方向向量和长度，按目标节点聚合。**每轴独立归一化** 以手腕为原点，各轴除以自身跨度。**1 Euro Filter**（min_cutoff=6.5 Hz, beta=0.002）对关键点坐标做自适应平滑，左右手独立滤波。**EMA**（α=0.75）对预测概率跨帧平滑。

## CNN 模型

| 组件     | 说明                                                                          |
| -------- | ----------------------------------------------------------------------------- |
| 输入     | 3×32×32（灰度图重复为三通道）                                                 |
| 架构     | Conv1(3→32)→Pool→Conv2(32→64)→Pool→Conv3(64→128)→Pool→FC(2048→256)→FC(256→10) |
| 参数量   | ~621,000                                                                      |
| 数据集   | QMNIST（60K 训练，60K 测试）                                                  |
| 数据增强 | RandomAffine(±10°,±12%,85-115%)                                           |

训练时使用轻量仿射增强，验证集不做增强。

## 手势类别

| ID  | 名称          | 标签  | 说明               |
| --- | ------------- | ----- | ------------------ |
| 0   | open_palm     | Open  | 五指全伸           |
| 1   | fist          | Fist  | 五指全屈           |
| 2   | index_point   | Idx   | 仅食指伸出         |
| 3   | victory       | V     | 食指 + 中指伸出    |
| 4   | ok            | OK    | 拇指食指成圈       |
| 5   | thumbs_up     | Like  | 拇指向上           |
| 6   | three_fingers | Three | 食、中、无名指伸出 |
| 7   | pinch         | Pinch | 拇食指指尖捏合     |
| 8   | four_fingers  | Four  | 除拇指外全伸       |
| 9   | thumb_down    | Down  | 拇指向下           |

## 配置项

| 文件                  | 变量               | 默认值 | 说明                           |
| --------------------- | ------------------ | ------ | ------------------------------ |
| `main.py`             | `_GCN_USE_CUDA`    | `True` | GCN 是否使用 GPU 推理          |
| `main.py`             | 捏合阈值           | `0.04` | 捏合判定距离（画面宽度的比例） |
| `main.py`             | 防抖帧数           | `8`    | 左手手势触发锁定帧数           |
| `GCN/collect_data.py` | `COLLECT_INTERVAL` | `0.2`  | 采集间隔（秒），5 FPS          |
| `GCN/train.py`        | `CONFIG` 字典      | —      | 训练超参数                     |
| `CNN/train.py`        | `CONFIG` 字典      | —      | 训练超参数                     |

## 参考文献

> Zhang, F., et al. _MediaPipe Hands: On-device Real-time Hand Tracking._ arXiv:2006.10214, 2020.
>
> Kipf, T. N. & Welling, M. _Semi-Supervised Classification with Graph Convolutional Networks._ ICLR 2017.
>
> Casiez, G., Roussel, N., & Vogel, D. _1€ Filter: A Simple Speed-based Low-pass Filter for Noisy Input in Interactive Systems._ CHI 2012.
>
> Wobbrock, J. O., Wilson, A. D., & Li, Y. _Gestures without Libraries, Toolkits or Training: A $1 Recognizer for User Interface Prototypes._ UIST 2007.

## 项目信息

南京航空航天大学「天目启航」专项 — 基于 MediaPipe 的实时手势识别系统研发
