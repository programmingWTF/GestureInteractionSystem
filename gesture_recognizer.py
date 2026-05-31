"""
手势识别器 - 曲率比算法 + 防抖稳定器

核心改进：
  用 手指曲率比 替代 y 坐标判断伸直/弯曲：
    曲率比 = dist(tip, mcp) / (dist(tip, pip) + dist(pip, mcp))
    直指 ≈ 1.0  |  弯指 < 0.85

优势：
  - 天然左右手通用（不依赖 x 坐标方向）
  - 手旋转不变（不依赖 y 轴方向）
  - 距离比抵消手型/摄像头距离影响
"""

import math
from collections import deque, Counter
import numpy as np

# MediaPipe 手部关键点索引
WRIST = 0
(THUMB_CMC, THUMB_MCP, THUMB_IP, THUMB_TIP) = (1, 2, 3, 4)
(INDEX_MCP, INDEX_PIP, INDEX_DIP, INDEX_TIP) = (5, 6, 7, 8)
(MIDDLE_MCP, MIDDLE_PIP, MIDDLE_DIP, MIDDLE_TIP) = (9, 10, 11, 12)
(RING_MCP, RING_PIP, RING_DIP, RING_TIP) = (13, 14, 15, 16)
(PINKY_MCP, PINKY_PIP, PINKY_DIP, PINKY_TIP) = (17, 18, 19, 20)

GESTURE_ACTIONS = {
    "手掌张开": "进入/退出书写模式",
    "拳头": "暂停/继续绘画",
    "食指": "空中书写",
    "胜利": "截屏",
    "OK": "确认轨迹/识别",
    "点赞": "好评/回车",
}


# ══════════════════════════════════════════════════════════
# 防抖稳定器
# ══════════════════════════════════════════════════════════

class GestureStabilizer:
    """滑动窗口投票 + 最短驻留防抖"""

    def __init__(self, window_size=15, min_ratio=0.55, lock_frames=6):
        self.window = deque(maxlen=window_size)
        self.min_ratio = min_ratio
        self.lock_frames = lock_frames
        self.locked = "无手势"
        self.candidate = None
        self.candidate_streak = 0

    def update(self, raw_gesture: str) -> str:
        self.window.append(raw_gesture)
        if len(self.window) < 5:
            return self.locked
        counts = Counter(self.window)
        top_gesture, top_count = counts.most_common(1)[0]
        ratio = top_count / len(self.window)
        if ratio >= self.min_ratio and top_gesture != "未知":
            if top_gesture == self.candidate:
                self.candidate_streak += 1
                if self.candidate_streak >= self.lock_frames:
                    self.locked = self.candidate
            else:
                self.candidate = top_gesture
                self.candidate_streak = 1
        else:
            self.candidate = None
            self.candidate_streak = 0
        return self.locked

    def reset(self):
        self.window.clear()
        self.locked = "无手势"
        self.candidate = None
        self.candidate_streak = 0


# ══════════════════════════════════════════════════════════
# 手势识别器（曲率比算法）
# ══════════════════════════════════════════════════════════

class GestureRecognizer:
    """基于手指曲率比的手势识别器"""

    @staticmethod
    def _dist(lm, a: int, b: int) -> float:
        return math.hypot(lm[a].x - lm[b].x, lm[a].y - lm[b].y)

    def _curvature(self, lm, tip: int, pip: int, mcp: int) -> float:
        """
        手指曲率比。
        = dist(tip, mcp) / (dist(tip, pip) + dist(pip, mcp))

        直指 ≈ 1.0（三点近共线）
        弯指 < 0.85（明显弯曲）
        """
        d1 = self._dist(lm, tip, pip)
        d2 = self._dist(lm, pip, mcp)
        d3 = self._dist(lm, tip, mcp)
        denom = d1 + d2
        return d3 / denom if denom > 1e-8 else 0.0

    def _hand_size(self, lm) -> float:
        return self._dist(lm, WRIST, INDEX_MCP)

    def recognize(self, hand_landmarks, handedness: str) -> dict:
        is_right = (handedness == "Right")
        lm = hand_landmarks
        sz = self._hand_size(lm)

        # 曲率比检测每根手指
        thumb = self._thumb_extended(lm, is_right, sz)
        index = self._finger_straight(lm, INDEX_TIP, INDEX_PIP, INDEX_MCP, strict=True)
        middle = self._finger_straight(lm, MIDDLE_TIP, MIDDLE_PIP, MIDDLE_MCP, strict=True)
        ring = self._finger_straight(lm, RING_TIP, RING_PIP, RING_MCP, strict=False)
        pinky = self._finger_straight(lm, PINKY_TIP, PINKY_PIP, PINKY_MCP, strict=False)

        finger_states = [thumb, index, middle, ring, pinky]
        gesture, confidence = self._classify(lm, finger_states, is_right, sz)

        return {
            "gesture": gesture,
            "action": GESTURE_ACTIONS.get(gesture, "无"),
            "confidence": confidence,
            "finger_states": finger_states,
        }

    # ── 手指检测 ──────────────────────────────────────

    def _finger_straight(self, lm, tip, pip, mcp, strict: bool) -> bool:
        """
        曲率比判断手指伸直。

        strict=True (食指/中指): 曲率 > 0.88
        strict=False (无名/小指): 曲率 > 0.80（解剖限制，不能完全独立伸直）
        """
        c = self._curvature(lm, tip, pip, mcp)
        return c > (0.88 if strict else 0.80)

    def _thumb_extended(self, lm, is_right: bool, sz: float) -> bool:
        """
        拇指伸出检测（收紧阈值，减少误判）。

        ① 竖起：拇指尖明显高于 IP 关节（点赞）
        ② 张开：拇指尖显著远离手掌（手掌张开）

        握拳时拇指自然微翘不应算「伸出」。
        """
        tip_y = lm[THUMB_TIP].y
        ip_y = lm[THUMB_IP].y
        tip_x = lm[THUMB_TIP].x
        idx_x = lm[INDEX_MCP].x

        # 竖起：拇指尖明显高于 IP
        thumb_up = (ip_y - tip_y) > sz * 0.07

        # 水平张开：拇指尖显著远离食指根部
        gap = sz * 0.08
        if is_right:
            thumb_out = tip_x < (idx_x - gap)
        else:
            thumb_out = tip_x > (idx_x + gap)

        return thumb_up or thumb_out

    def _thumb_pointing_up(self, lm, sz: float) -> bool:
        """
        拇指是否真正朝上（区分点赞 vs 握拳拇指外翘）。

        要求：
          1. 拇指尖明显高于 IP 关节
          2. 拇指尖也高于 MCP 关节（确保整体竖起，不是关节微弯）
          3. 拇指尖不能离手掌太远（点赞时拇指贴近手掌侧面）
        """
        tip = lm[THUMB_TIP]
        ip = lm[THUMB_IP]
        mcp = lm[THUMB_MCP]

        # 必须同时高于 IP 和 MCP
        above_ip = (ip.y - tip.y) > sz * 0.06
        above_mcp = (mcp.y - tip.y) > sz * 0.03

        if not (above_ip and above_mcp):
            return False

        # 水平偏移不能太大（点赞时拇指在手掌边缘，不会飘太远）
        horizontal_offset = abs(tip.x - mcp.x)
        if horizontal_offset > sz * 0.25:
            return False

        return True

    def _circle_dist(self, lm) -> float:
        """拇指尖与食指尖距离"""
        return self._dist(lm, THUMB_TIP, INDEX_TIP)

    # ── 分类 ──────────────────────────────────────────

    def _classify(self, lm, fs, is_right, sz) -> tuple:
        thumb, index, middle, ring, pinky = fs
        ext = sum(fs)

        # ── OK 👌 ──（两个条件任满足其一）
        d = self._circle_dist(lm)
        # 条件 A：圆圈足够小 + 中指伸直（侧面/正面均可）
        ok_by_circle = d < sz * 0.12 and middle
        # 条件 B：圆圈很紧（正对摄像头时拇指食指紧贴）+ 至少中指不全弯
        ok_by_tight = d < sz * 0.07 and not ring and not pinky

        if ok_by_circle:
            # 侧面时食指应弯曲，正面时不强求（2D 投影看不清弯曲）
            idx_curve = self._curvature(lm, INDEX_TIP, INDEX_PIP, INDEX_MCP)
            if idx_curve < 0.80:
                return "OK", 0.93  # 侧面 OK，食指明显弯曲
            return "OK", 0.85      # 正面 OK，食指看起来直
        if ok_by_tight:
            return "OK", 0.82      # 圆圈极紧，可能是正面 OK

        # ── 胜利 ✌️ ──
        if index and middle and not thumb and not ring and not pinky:
            return "胜利", 0.93
        # 食指中指 + 小指微弯也算
        if index and middle and not thumb and not ring and pinky:
            return "胜利", 0.85

        # ── 食指 ☝️ ──
        if index and not middle and not ring and not pinky:
            return "食指", 0.90

        # ── 点赞 👍 ──（仅拇指竖起，且必须明显朝上）
        if thumb and not index and not middle and not ring and not pinky:
            if self._thumb_pointing_up(lm, sz):
                return "点赞", 0.91
            # 拇指微伸但不竖起 → 拳头（握拳时拇指自然外翘）
            return "拳头", 0.85

        # ── 手掌张开 ✋ ──
        if ext >= 5:
            return "手掌张开", 0.94
        if ext == 4 and (index and middle and (thumb or ring or pinky)):
            return "手掌张开", 0.87

        # ── 拳头 ✊ ──
        if ext == 0:
            return "拳头", 0.95
        if ext == 1 and thumb and not self._thumb_pointing_up(lm, sz):
            return "拳头", 0.90
        if ext <= 2 and not index and not middle:
            return "拳头", 0.85

        # ── 宽松食指：食指直 + 中指弯 ──
        if index and not middle:
            return "食指", 0.80

        # ── 宽松胜利：食指中指直 + 无名指可能弯 ──
        if index and middle and not thumb:
            return "胜利", 0.78

        return "未知", 0.30
