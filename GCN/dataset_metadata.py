"""
GCN 手势数据集元数据

此文件记录所有手势类别信息，供训练脚本读取。
"""

# 手势列表：(id, name, display_name_cn)
GESTURES = [
    (0,   "open_palm",      "手掌张开"),
    (1,   "fist",           "拳头"),
    (2,   "index_point",    "食指指出"),
    (3,   "victory",        "胜利V"),
    (4,   "ok",             "OK"),
    (5,   "thumbs_up",      "点赞"),
    (6,   "three_fingers",  "三指"),
    (7,   "pinch",          "捏合"),
    (8,   "four_fingers",   "四指"),
    (9,   "thumb_down",     "拇指向下"),
]

NUM_CLASSES = len(GESTURES)  # 10

GESTURE_BY_ID = {g[0]: g for g in GESTURES}
GESTURE_BY_NAME = {g[1]: g for g in GESTURES}

# 手部关键点名称（MediaPipe 标准，共 21 个）
LANDMARK_NAMES = [
    "wrist",
    "thumb_cmc", "thumb_mcp", "thumb_ip", "thumb_tip",
    "index_mcp", "index_pip", "index_dip", "index_tip",
    "middle_mcp", "middle_pip", "middle_dip", "middle_tip",
    "ring_mcp", "ring_pip", "ring_dip", "ring_tip",
    "pinky_mcp", "pinky_pip", "pinky_dip", "pinky_tip",
]

NUM_LANDMARKS = 21
NUM_FEATURES = NUM_LANDMARKS * 3  # 63 (x, y, z per landmark)

# 手部骨架边（用于 GCN 邻接矩阵）
HAND_EDGES = [
    (0, 1), (1, 2), (2, 3), (3, 4),        # 拇指
    (0, 5), (5, 6), (6, 7), (7, 8),        # 食指
    (5, 9), (9, 10), (10, 11), (11, 12),    # 中指
    (9, 13), (13, 14), (14, 15), (15, 16),  # 无名指
    (13, 17), (0, 17),                       # 小指 + 手腕到小指根部
    (17, 18), (18, 19), (19, 20),            # 小指
]

# DataFrame 列名（CSV 输出顺序）
def get_csv_columns():
    cols = ["timestamp", "handedness", "gesture"]
    for i, name in enumerate(LANDMARK_NAMES):
        cols.append(f"lm{i}_{name}_x")
        cols.append(f"lm{i}_{name}_y")
        cols.append(f"lm{i}_{name}_z")
    return cols
