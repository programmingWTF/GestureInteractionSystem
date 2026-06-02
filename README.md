# GestureInteractionSystem 🖐️

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?logo=pytorch&logoColor=white" alt="PyTorch">
  <img src="https://img.shields.io/badge/OpenCV-4.8+-5C3EE8?logo=opencv&logoColor=white" alt="OpenCV">
  <img src="https://img.shields.io/badge/MediaPipe-0.10+-00C853" alt="MediaPipe">
  <img src="https://img.shields.io/badge/license-Educational-blue" alt="License">
</p>

<p align="center"><b>基于 MediaPipe + GCN + CNN 的实时手势识别系统</b></p>
<p align="center">Apple Vision Pro 风格捏合书写交互 · 10 类静态手势分类 · CNN 手写数字识别</p>

---

## 项目结构

```
GestureInteractionSystem/
│
├── main.py                      # 主程序入口（摄像头 + GUI + 交互逻辑）
├── gesture_recognizer.py        # 曲率比手势分类器（规则方法，已弃用）
├── trajectory_recognizer.py     # $1 Unistroke 轨迹识别器（已弃用）
├── hand_landmarker.task         # MediaPipe 手部关键点检测模型
│
├── CNN/                         # CNN 手写数字识别
│   ├── predict_api.py           # 推理接口（基于 QMNIST）
│   └── qmnist_digit_model.pth   # 预训练权重
│
├── GCN/                         # GCN 手势识别子系统
│   ├── collect_data.py          # 数据采集工具
│   ├── train.py                 # 模型训练脚本
│   ├── model.py                 # GCN 模型定义
│   ├── predictor.py             # 推理封装（加载 / 预处理 / 平滑）
│   ├── dataset.py               # PyTorch Dataset
│   ├── dataset_metadata.py      # 元数据（手势标签、关键点名称、骨骼边）
│   ├── best_model.pth           # 训练好的模型权重
│   ├── DateSet/                 # 采集的训练数据
│   └── README.md
│
├── requirements.txt
└── README.md
```

## 交互模型

参考 Apple Vision Pro 的空间交互设计：

| 手      | 手势               | 动作                         |
| ------- | ------------------ | ---------------------------- |
| 🟦 左手 | OK                 | 进入书写模式（同时清空画布） |
| 🟦 左手 | 点赞               | 提交笔画给 CNN 识别数字      |
| 🟦 左手 | 手掌张开           | 清空所有笔画（保持书写模式） |
| 🟩 右手 | 捏合（拇食指并拢） | 落笔，笔尖跟随两指中点       |
| 🟩 右手 | 手掌张开           | 抬笔，结束当前笔画           |

- 手势分类由 GCN 模型完成，捏合检测使用欧氏距离阈值
- 左手手势带 8 帧防抖锁定，防止误触发
- 所有关键点经过 1 Euro Filter 平滑，消除骨架抖动

## 环境

| 依赖         | 版本   | 说明                 |
| ------------ | ------ | -------------------- |
| Python       | ≥ 3.10 | —                    |
| OpenCV       | ≥ 4.8  | 摄像头采集与画面渲染 |
| MediaPipe    | ≥ 0.10 | 手部 21 点关键点检测 |
| PyTorch      | ≥ 2.0  | GCN 模型训练与推理   |
| NumPy        | ≥ 1.24 | 数组计算             |
| Pillow       | ≥ 9.0  | CNN 输入图渲染       |
| scikit-learn | ≥ 1.0  | 分类报告与混淆矩阵   |
| matplotlib   | ≥ 3.5  | 训练曲线绘制         |

```bash
pip install opencv-python mediapipe torch numpy Pillow scikit-learn matplotlib
```

## 快速开始

### 1. 采集数据

```bash
python GCN/collect_data.py
```

在 OpenCV 窗口中操作：

| 按键      | 功能                    |
| --------- | ----------------------- |
| `N` / `P` | 上一个 / 下一个手势类别 |
| `H`       | 切换左手 / 右手         |
| `SPACE`   | 暂停 / 继续录制         |
| `R`       | 重置当前手势的 CSV      |
| `ESC`     | 退出                    |

数据以 5 FPS 保存至 `GCN/DateSet/<类别名>/<类别名>.csv`。每行 66 列：时间戳 + 手标签 + 手势标签 + 21 个关键点 × (x, y, z)。

> **建议**：每种手势录制 ≥ 400 帧，在不同光照和手距下分多次采集，提升模型泛化能力。

### 2. 训练模型

```bash
python GCN/train.py
```

自动检测 CUDA 并优先使用 GPU。输出 `best_model.pth` 和 `training_curves.png`。

### 3. 运行

```bash
python main.py
```

如需强制 CPU 推理，将 `main.py` 顶部的 `_GCN_USE_CUDA = True` 改为 `False`。

## GCN 模型

### 架构

```
输入 (B, 21, 8)
  │  ← xyz_norm(3) + handedness(1) + bone_dx,dy,dz,len(4)
  ▼
GCNConv(8 → 160)  + LayerNorm + ReLU + Dropout(0.3)
  ▼
GCNConv(160 → 160) + LayerNorm + ReLU + Dropout(0.3)
  ▼
GCNConv(160 → 160) + LayerNorm + ReLU
  ▼
Global Mean Pool (21 节点 → 1 向量)
  ▼
Linear(160 → 80) + ReLU + Dropout(0.3)
  ▼
Linear(80 → 10)
```

| 指标           | 数值         |
| -------------- | ------------ |
| 参数量         | ~68,000      |
| 单帧推理 (CPU) | < 0.2 ms     |
| 单帧推理 (GPU) | < 0.05 ms    |
| 输入维度       | 8 维 / 节点  |
| 输出           | 10 类 logits |

### 关键设计

<details>
<summary><b>骨骼特征 (Bone Features)</b></summary>

对 21 条手部骨骼边计算方向向量 `(dx, dy, dz)` 和长度 `len`，按目标节点做 scatter-mean 聚合。每个节点获得 4 维骨骼特征，使模型能感知拇指朝向、手指弯曲程度等几何信息，显著改善"拳头 vs 点赞"等依赖拇指方向的分类。

```
bone(u→v) = [ (xv-xu)/len, (yv-yu)/len, (zv-zu)/len, len ]
node(v)   = mean( bone(u→v) for all incoming edges u→v )
```

</details>

<details>
<summary><b>每轴独立归一化</b></summary>

以手腕为原点平移后，x、y、z 轴各自除以其跨度（max − min），而非用统一的 3D 手部尺度。避免 z 轴被 x,y 主导的范数压至接近零，确保三个维度在输入中量级相当。

```
h_centered = h - wrist
span       = h_centered.max(axis=1) - h_centered.min(axis=1)  # per-axis
h_norm     = h_centered / span
```

</details>

<details>
<summary><b>1 Euro Filter</b></summary>

对 21 个关键点的 63 个坐标分量各维护一个自适应低通滤波器。参数 `min_cutoff=6.5 Hz, beta=0.002`：

- 手静止时强平滑，滤除 MediaPipe 像素级抖动
- 手快速移动时自动放宽截止频率，保持低延迟跟踪

左右手各维护独立的滤波器组，避免双手同时出现时状态串扰。

</details>

<details>
<summary><b>时序平滑 (EMA)</b></summary>

推理时对概率向量做指数移动平均（α=0.75），左手和右手分别维护独立的 EMA 状态。相邻帧之间手势不会因瞬时噪声跳变。

```
p_smooth(t) = 0.75 × p_raw(t) + 0.25 × p_smooth(t-1)
```

</details>

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

## 训练配置

| 参数            | 值               | 说明                     |
| --------------- | ---------------- | ------------------------ |
| batch_size      | 64               | 每批样本数               |
| epochs          | 400              | 最大训练轮数（含早停）   |
| lr              | 2e-3             | 初始学习率               |
| optimizer       | AdamW            | weight_decay=1e-5        |
| scheduler       | Cosine annealing | 5 epoch 线性 warmup      |
| label_smoothing | 0.05             | 防止过拟合               |
| dropout         | 0.3              | GCN 层与分类器的丢弃率   |
| 数据划分        | 时序划分         | 前 80% 训练，后 20% 验证 |
| 早停            | 50 epochs        | 验证准确率不再提升时停止 |

<details>
<summary><b>为什么用时序划分？</b></summary>

数据以 0.2 秒间隔连续录制，相邻帧高度相似。随机划分会导致几乎相同的帧同时出现在训练集和验证集（数据泄漏），验证准确率虚高但真实场景表现差。时序划分将每类数据按采集顺序切分——前 80% 训练，后 20% 验证——验证集来自采集会话末期，手的姿态已有漂移，更接近真实泛化场景。

</details>

## CNN 轨迹识别

提交笔画时（左手点赞），系统执行以下流程：

1. 计算所有笔画像素点的包围盒
2. 取正方形裁剪区域（15% 扩边，不拉伸，不丢失信息）
3. 用 PIL 渲染为 280×280 白底黑字图像
4. 通过 subprocess 调用 `CNN/predict_api.py`
5. 在左下角显示识别结果 + 真实比例预览图 + Top-3 置信度

## 配置项

| 位置                  | 变量                | 默认值    | 说明                  |
| --------------------- | ------------------- | --------- | --------------------- |
| `main.py`             | `_GCN_USE_CUDA`     | `True`    | GCN 是否使用 GPU 推理 |
| `main.py`             | `pinch_dist < 0.04` | 4% 画面宽 | 捏合距离阈值          |
| `main.py`             | `left_lock = 8`     | 8 帧      | 左手手势防抖锁定时长  |
| `GCN/collect_data.py` | `COLLECT_INTERVAL`  | `0.2`     | 采集间隔（秒）        |

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
