# GCN 手势数据集采集 & 训练

## 目录结构

```
GCN/
├── collect_data.py       # 数据采集主程序
├── dataset_metadata.py   # 手势元数据（类别名、关键点、骨架边）
├── DataSet/              # 采集的 CSV 数据
│   ├── open_palm/
│   │   └── open_palm.csv
│   ├── fist/
│   │   └── fist.csv
│   └── ...
└── README.md
```

## 1. 采集数据

```bash
python GCN/collect_data.py
```

### 操作流程

1. **启动时**在终端选择左手/右手 和 手势类型
2. 摄像头打开后，**做对应手势**面对摄像头
3. 系统每 0.5 秒自动保存一帧关键点数据
4. 在终端输入命令或按键盘快捷键切换

### 控制方式

| 终端命令 | 键盘快捷键 | 功能 |
|---------|-----------|------|
| 输入 `0`-`9` | `N` | 切换手势（循环） |
| `L` / `R` | `H` | 切换左手/右手 |
| `P` | `SPACE` | 暂停/继续采集 |
| `S` | `S` | 查看采集统计 |
| `Q` | `ESC` / `Q` | 退出 |

### CSV 格式

每行一帧，66 列：
```
timestamp, handedness, gesture, lm0_wrist_x, lm0_wrist_y, lm0_wrist_z, ..., lm20_pinky_tip_x, lm20_pinky_tip_y, lm20_pinky_tip_z
```

## 2. 手势列表（10 种）

| ID | 名称 | 中文 |
|----|------|------|
| 0 | open_palm | 手掌张开 |
| 1 | fist | 拳头 |
| 2 | index_point | 食指指出 |
| 3 | victory | 胜利 V |
| 4 | ok | OK |
| 5 | thumbs_up | 点赞 |
| 6 | three_fingers | 三指 |
| 7 | pinch | 捏合 |
| 8 | four_fingers | 四指 |
| 9 | thumb_down | 拇指向下 |

## 3. 训练 GCN 模型

```bash
python GCN/train.py
```

训练时自动检测 CUDA，有 GPU 则用 GPU，否则用 CPU。
