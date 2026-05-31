# GestureInteractionSystem 🖐️

> 基于 MediaPipe Task API 的实时手势识别系统 — 支持静态手势识别 + 空中数字书写

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![OpenCV](https://img.shields.io/badge/OpenCV-4.8+-green)
![MediaPipe](https://img.shields.io/badge/MediaPipe-0.10+-orange)

---

## ✨ 功能

### 静态手势识别（6 种）
| 手势 | 操作映射 | 识别原理 |
|------|---------|---------|
| ✋ 手掌张开 | 进入/退出书写模式 | 5 指全伸 |
| ✊ 拳头 | 暂停/继续绘画 | 5 指全屈 |
| ☝️ 食指 | 空中书写 | 仅食指伸直 |
| ✌️ 胜利 | 截屏 | 食指 + 中指伸直 |
| 👌 OK | 确认轨迹 / 切换模式 | 拇指食指成圈 |
| 👍 点赞 | 好评 / 回车 | 拇指竖起 |

### 空中书写 ✍️
- 食指指尖追踪轨迹，支持**多笔画**
- **双引擎轨迹识别**：
  - **$1 Unistroke Recognizer** — 模板匹配，识别数字 0-9 + 6 种符号（✓✗←→↑↓）
  - **CNN 数字分类器** — 基于 QMNIST 训练的轻量卷积网络，仅识别数字 0-9
- 书写模式下手掌张开清空画布，OK 手势提交识别

### 交互体验
- 实时防抖（滑动窗口投票 + 最短驻留）
- 完整中文 UI 面板（手势名、置信度进度条、手指状态、操作提示）
- 轨迹预览缩略图
- FPS 实时显示
- 截屏保存（`S` 键）/ 全屏切换（`F` 键）

---

## 🏗️ 架构

```
GestureInteractionSystem/
├── main.py                   # 主程序入口
├── gesture_recognizer.py     # 手势识别器（曲率比算法）
│   ├── GestureRecognizer     # 曲率比手势分类
│   └── GestureStabilizer     # 滑动窗口防抖
├── trajectory_recognizer.py  # $1 Unistroke 轨迹识别器
├── cnn/
│   ├── predict_api.py        # CNN 数字预测接口
│   └── qmnist_digit_model.pth  # 预训练模型权重
├── hand_landmarker.task      # MediaPipe 手部检测模型
├── requirements.txt
├── .gitignore
└── README.md
```

### 技术栈
- **视觉推理** — MediaPipe Task API（Hand Landmarker）
- **图像处理** — OpenCV + Pillow
- **轨迹识别** — $1 Unistroke Recognizer（纯算法，零训练）
- **数字分类** — PyTorch 轻量 CNN（EnhancedCNN, ~2.8M 参数）
- **UI 渲染** — OpenCV + PIL 混合渲染（支持中文）

---

## 🚀 快速开始

### 环境要求
- Python 3.10+
- 普通 USB 摄像头

### 安装

```bash
# 1. 克隆仓库
git clone https://github.com/programmingWTF/GestureInteractionSystem.git
cd GestureInteractionSystem

# 2. 安装依赖（核心运行）
pip install -r requirements.txt

# 3. 下载 MediaPipe 手部模型
# 从 https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task
# 下载到项目根目录，或使用项目自带版本

# 4. 运行
python main.py
```

> **CNN 数字识别为可选功能**，需要额外安装 PyTorch（建议 2.x）。若未安装，系统自动回退 $1 识别器。

### 操作键位
| 按键 | 功能 |
|------|------|
| `ESC` | 退出程序 |
| `S` | 截屏（保存至 `screenshots/`） |
| `F` | 切换全屏 |

---

## 🎯 使用说明

### 书写模式
1. 摄像头前做 **OK 手势** → 进入书写模式
2. **伸出食指** → 空中写字（轨迹实时显示）
3. **握拳** → 暂停当前笔画，再握拳清空全部
4. 再次做 **OK 手势** → 提交识别并退出书写模式

### 识别结果
书写模式下做 OK 手势提交后，系统会：
1. 优先调用 CNN 模型预测数字
2. 若 CNN 不可用，自动回退 $1 识别器
3. 结果显示在右侧面板 + 左下角预览图

---

## 📊 性能目标
- 实时帧率 ≥ 30 FPS（普通笔记本 CPU）
- 手势识别准确率 ≥ 90%（标准光照条件）
- 端到端延迟 ≤ 100ms

---

## 🔭 未来方向
- [ ] 动态手势识别（挥手、画圆、推拉）
- [ ] 双手模式（`num_hands=2`）
- [ ] 民航驾驶舱场景演示（手势切屏、参数输入）
- [ ] 自定义手势→动作映射配置
- [ ] 语音反馈播报
- [ ] Web 部署版

---

## 📄 项目信息

- **项目**：南京航空航天大学「天目启航」专项 — 基于 MediaPipe 的实时手势识别系统研发
- **组长**：李桂聿
- **组员**：黄周卓琳、纪雅宁、张艺丹、范恺莹
- **指导教师**：孙有朝教授、吴红兰高工
- **时间**：2026.01 – 2026.06

---

## 📚 参考文献

- [MediaPipe Hands: On-device Real-time Hand Tracking](https://arxiv.org/abs/2006.10214)
- [$1 Unistroke Recognizer](https://dl.acm.org/doi/10.1145/1294211.1294238) — Wobbrock et al., 2007
- [QMNIST Dataset](https://github.com/facebookresearch/qmnist)

## 📜 许可

本项目仅用于南京航空航天大学创新实践教学用途。
