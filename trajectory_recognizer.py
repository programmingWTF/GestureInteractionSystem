"""
轨迹识别器 - $1 Unistroke Recognizer 实现

基于 Wobbrock et al. (2007) 的 $1 识别器算法。
无需训练，通过路径距离匹配预定义模板。

支持识别：
  - 数字 0-9
  - 符号: ✓ 确认 / ✗ 取消 / ← 左 / → 右 / ↑ 上 / ↓ 下
"""

import math
import numpy as np

# ══════════════════════════════════════════════════════════
# 点集工具函数
# ══════════════════════════════════════════════════════════


def _path_length(points):
    """计算路径总长度"""
    return sum(math.dist(points[i], points[i + 1]) for i in range(len(points) - 1))


def _resample(points, n=64):
    """将路径重采样为 n 个等距点"""
    if len(points) < 2:
        return points
    total = _path_length(points)
    if total == 0:
        return points
    interval = total / (n - 1)
    result = [points[0]]
    dist = 0.0
    i = 0
    while len(result) < n:
        if i >= len(points) - 1:
            result.append(points[-1])
        else:
            d = math.dist(points[i], points[i + 1])
            if dist + d >= interval:
                t = (interval - dist) / d
                qx = points[i][0] + t * (points[i + 1][0] - points[i][0])
                qy = points[i][1] + t * (points[i + 1][1] - points[i][1])
                result.append((qx, qy))
                points.insert(i + 1, (qx, qy))
                dist = 0.0
            else:
                dist += d
            i += 1
    # 确保正好 n 个点
    while len(result) < n:
        result.append(points[-1])
    return result[:n]


def _centroid(points):
    x = sum(p[0] for p in points) / len(points)
    y = sum(p[1] for p in points) / len(points)
    return (x, y)


def _indicative_angle(points):
    """从质心到第一个点的角度"""
    c = _centroid(points)
    return math.atan2(points[0][1] - c[1], points[0][0] - c[0])


def _rotate_by(points, angle):
    """按角度旋转点集"""
    c = _centroid(points)
    cos_a = math.cos(angle)
    sin_a = math.sin(angle)
    result = []
    for x, y in points:
        dx = x - c[0]
        dy = y - c[1]
        rx = dx * cos_a - dy * sin_a
        ry = dx * sin_a + dy * cos_a
        result.append((rx + c[0], ry + c[1]))
    return result


def _scale_to(points, size=250.0):
    """将点集缩放到标准大小"""
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    w = max_x - min_x
    h = max_y - min_y
    scale = size / max(w, h, 1)
    result = []
    for x, y in points:
        result.append((x * scale, y * scale))
    return result


def _translate_to(points, pt=(0, 0)):
    """平移点集使质心位于 pt"""
    c = _centroid(points)
    dx = pt[0] - c[0]
    dy = pt[1] - c[1]
    return [(x + dx, y + dy) for x, y in points]


def _path_distance(a, b):
    """计算两个等长路径的距离（取最小旋转偏移）"""
    n = len(a)
    best = float("inf")
    for offset in range(n):
        d = 0.0
        for i in range(n):
            j = (i + offset) % n
            d += math.dist(a[i], b[j])
        d /= n
        best = min(best, d)
    return best


def _normalize(points, n=64, size=250.0):
    """
    完整的 $1 归一化流程：
    resample → rotate to 0° → scale → translate to origin
    """
    pts = _resample(points, n)
    angle = _indicative_angle(pts)
    pts = _rotate_by(pts, -angle)
    pts = _scale_to(pts, size)
    pts = _translate_to(pts, (0, 0))
    return pts


# ══════════════════════════════════════════════════════════
# 模板定义（数字 0-9 + 符号）
# 坐标范围为 [0, 100]
# ══════════════════════════════════════════════════════════

def _make_digit_templates():
    """生成数字 0-9 的模板路径（逆时针/自上而下）"""
    R = 45.0
    cx, cy = 50.0, 50.0
    t = {}

    # 0: 椭圆（逆时针）
    pts0 = []
    for i in range(32):
        angle = math.pi * 2 * i / 32
        pts0.append((cx + 40 * math.cos(angle), cy + 45 * math.sin(angle)))
    t[0] = _normalize(pts0)

    # 1: 纵向直线
    t[1] = _normalize([(50, 5), (50, 95)])

    # 2: 右上→右下→左下→右下
    pts2 = [(20, 20), (80, 20), (80, 55), (20, 55), (20, 55), (20, 90), (80, 90)]
    t[2] = _normalize(pts2)

    # 3: 类似两个右半圆
    pts3 = [(20, 15), (80, 15), (80, 45), (50, 45),
            (50, 45), (80, 45), (80, 75), (80, 75), (20, 85)]
    t[3] = _normalize(pts3)

    # 4: 竖折横
    t[4] = _normalize([(60, 5), (60, 50), (5, 50), (5, 50), (95, 50), (95, 50), (60, 50), (60, 95)])

    # 5: 左上→右上→左下→右下
    t[5] = _normalize([(80, 10), (20, 10), (20, 10), (20, 55), (80, 55), (80, 55), (80, 90), (20, 90)])

    # 6: 逆时针卷曲
    pts6 = [(80, 10), (20, 10), (20, 50), (20, 90), (80, 90), (80, 55), (20, 55)]
    t[6] = _normalize(pts6)

    # 7: 横折斜
    t[7] = _normalize([(10, 10), (90, 10), (90, 10), (90, 10), (30, 90)])

    # 8: 两个圈/沙漏形
    pts8 = []
    for i in range(24):
        a = math.pi * 2 * i / 24
        pts8.append((cx + 35 * math.cos(a), 30 + 28 * math.sin(a)))
    for i in range(24):
        a = math.pi * 2 * i / 24
        pts8.append((cx + 35 * math.cos(a), 70 + 28 * math.sin(a)))
    t[8] = _normalize(pts8)

    # 9: 顺时针卷曲
    pts9 = [(20, 90), (80, 90), (80, 50), (80, 10), (20, 10), (20, 45), (80, 45)]
    t[9] = _normalize(pts9)

    return t


def _make_symbol_templates():
    """生成符号模板"""
    t = {}

    # ✓ 勾号
    t["check"] = _normalize([(10, 50), (40, 80), (90, 10)])

    # ✗ 叉号
    t["cross"] = _normalize([(10, 10), (90, 90), (50, 50), (10, 90), (90, 10)])

    # ← 左箭头
    t["left"] = _normalize([(90, 50), (10, 50), (30, 25), (10, 50), (30, 75)])

    # → 右箭头
    t["right"] = _normalize([(10, 50), (90, 50), (70, 25), (90, 50), (70, 75)])

    # ↑ 上箭头
    t["up"] = _normalize([(50, 90), (50, 10), (25, 30), (50, 10), (75, 30)])

    # ↓ 下箭头
    t["down"] = _normalize([(50, 10), (50, 90), (25, 70), (50, 90), (75, 70)])

    # ○ 圆圈
    pts_circle = []
    for i in range(32):
        a = math.pi * 2 * i / 32
        pts_circle.append((50 + 45 * math.cos(a), 50 + 45 * math.sin(a)))
    t["circle"] = _normalize(pts_circle)

    return t


# ══════════════════════════════════════════════════════════
# 识别器
# ══════════════════════════════════════════════════════════


class TrajectoryRecognizer:
    """$1 轨迹识别器"""

    def __init__(self, resample_n=64):
        self.resample_n = resample_n

        # 预归一化所有模板
        self.digit_templates = _make_digit_templates()
        self.symbol_templates = _make_symbol_templates()

        # 合并所有模板
        self._templates = {}
        for k, v in self.digit_templates.items():
            self._templates[str(k)] = v
        for k, v in self.symbol_templates.items():
            self._templates[k] = v

        # 模板名称列表（用于报告）
        self._label_map = {
            "0": "0", "1": "1", "2": "2", "3": "3", "4": "4",
            "5": "5", "6": "6", "7": "7", "8": "8", "9": "9",
            "check": "对勾", "cross": "叉号",
            "left": "左箭头", "right": "右箭头",
            "up": "上箭头", "down": "下箭头",
            "circle": "圆圈",
        }

    def recognize(self, points, min_confidence=0.55) -> dict:
        """
        识别轨迹。

        Args:
            points: [(x, y), ...] 原始轨迹点列表
            min_confidence: 最低置信度阈值

        Returns:
            {
                "label": str,        # 识别结果标签
                "display": str,      # 用于展示的名称
                "confidence": float, # 置信度 0-1
                "score": float,      # 原始距离分数
            }
        """
        if len(points) < 8:
            return {"label": None, "display": "轨迹太短",
                    "confidence": 0, "score": float("inf")}

        # 归一化输入
        try:
            normalized = _normalize(points, self.resample_n)
        except (ValueError, ZeroDivisionError):
            return {"label": None, "display": "无法识别",
                    "confidence": 0, "score": float("inf")}

        # 匹配所有模板
        best_label = None
        best_score = float("inf")
        half_diagonal = 0.5 * math.sqrt(250**2 + 250**2)  # ~177

        for label, template in self._templates.items():
            d = _path_distance(normalized, template)
            score = 1.0 - d / half_diagonal
            if d < best_score:
                best_score = d
                best_label = label

        confidence = max(0.0, min(1.0, 1.0 - best_score / half_diagonal))

        if confidence < min_confidence:
            return {"label": None, "display": "未匹配",
                    "confidence": confidence, "score": best_score}

        display = self._label_map.get(best_label, best_label)
        return {"label": best_label, "display": display,
                "confidence": confidence, "score": best_score}

    def recognize_top_n(self, points, n=3) -> list:
        """返回 top-n 识别结果"""
        if len(points) < 8:
            return []

        try:
            normalized = _normalize(points, self.resample_n)
        except (ValueError, ZeroDivisionError):
            return []

        results = []
        half_diagonal = 0.5 * math.sqrt(250**2 + 250**2)

        for label, template in self._templates.items():
            d = _path_distance(normalized, template)
            confidence = max(0.0, min(1.0, 1.0 - d / half_diagonal))
            results.append({
                "label": label,
                "display": self._label_map.get(label, label),
                "confidence": confidence,
                "score": d,
            })

        results.sort(key=lambda r: r["score"])
        return results[:n]
