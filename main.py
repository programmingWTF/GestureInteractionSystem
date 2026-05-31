"""
GestureInteractionSystem - 主程序入口

基于 MediaPipe Task API + OpenCV 的实时手势识别系统
支持 6 种静态手势识别，并映射到对应的操作。

双手模式（v2）：
  - 左手（蓝色骨架）→ 控制：OK切换、拳头暂停、手掌清空、点赞提交
  - 右手（绿色骨架）→ 书写：食指指尖在空中写字
  - 仅单手可见 → 自动回退原版单手握拳+书写混合模式

按 ESC 退出，按 S 截图，按 F 全屏。
"""

import cv2
import time
import os
import sys
import math
import json
import subprocess
import tempfile
import numpy as np
from PIL import Image, ImageDraw, ImageFont

import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

from gesture_recognizer import GestureRecognizer, GestureStabilizer, GESTURE_ACTIONS
from trajectory_recognizer import TrajectoryRecognizer


# ══════════════════════════════════════════════════════════
# 配置
# ══════════════════════════════════════════════════════════
CAMERA_ID = 0

_MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(_MODULE_DIR, "hand_landmarker.task")

CNN_PYTHON = sys.executable
CNN_SCRIPT = os.path.join(_MODULE_DIR, "cnn", "predict_api.py")

# ══════════════════════════════════════════════════════════
# 中文字体加载（全局缓存，只加载一次）
# ══════════════════════════════════════════════════════════
_FONT_PATHS = [
    "C:/Windows/Fonts/msyh.ttc",
    "C:/Windows/Fonts/simhei.ttf",
    "C:/Windows/Fonts/simsun.ttc",
]


def _load_font(size: int) -> ImageFont.FreeTypeFont:
    for fp in _FONT_PATHS:
        try:
            return ImageFont.truetype(fp, size)
        except (OSError, IOError):
            continue
    return ImageFont.load_default()


_FONT_LARGE = None
_FONT_MID = None
_FONT_SMALL = None
_FONT_XS = None
_FONT_XXS = None
_FONT_MINI = None


def _init_fonts():
    global _FONT_LARGE, _FONT_MID, _FONT_SMALL
    global _FONT_XS, _FONT_XXS, _FONT_MINI
    _FONT_LARGE = _load_font(34)
    _FONT_MID = _load_font(24)
    _FONT_SMALL = _load_font(22)
    _FONT_XS = _load_font(17)
    _FONT_XXS = _load_font(15)
    _FONT_MINI = _load_font(14)


# ══════════════════════════════════════════════════════
# 轨迹渲染 & CNN 预测
# ══════════════════════════════════════════════════════


def render_trajectory(strokes, size=280, line_width=22, margin=15,
                      crop_to_content=True, frame_w=1920, frame_h=1080):
    from PIL import ImageDraw as PILD
    from PIL import ImageOps as PILOps

    aspect = frame_w / frame_h
    draw_size = size * 2 if crop_to_content else size
    ds = draw_size
    m = margin * (draw_size / size)

    canvas_w = ds
    canvas_h = int(ds / aspect)
    img = Image.new("L", (canvas_w, canvas_h), color=255)
    draw = PILD.Draw(img)

    for stroke in strokes:
        if len(stroke) < 2:
            for x, y in stroke:
                px = int(x * (canvas_w - 2 * m) + m)
                py = int(y * (canvas_h - 2 * m) + m)
                r = line_width // 2
                draw.ellipse([px - r, py - r, px + r, py + r], fill=0)
            continue
        pts = [(int(x * (canvas_w - 2 * m) + m),
                int(y * (canvas_h - 2 * m) + m)) for x, y in stroke]
        draw.line(pts, fill=0, width=line_width)
        for px, py in (pts[0], pts[-1]):
            r = line_width // 2
            draw.ellipse([px - r, py - r, px + r, py + r], fill=0)

    if crop_to_content:
        bbox = PILOps.invert(img).getbbox()
        if bbox:
            pad = 8
            x1, y1, x2, y2 = bbox
            x1 = max(0, x1 - pad)
            y1 = max(0, y1 - pad)
            x2 = min(canvas_w, x2 + pad)
            y2 = min(canvas_h, y2 + pad)
            img = img.crop((x1, y1, x2, y2))

        pw, ph = img.size
        if pw > 0 and ph > 0:
            scale = size / max(pw, ph)
            new_w = max(1, int(pw * scale))
            new_h = max(1, int(ph * scale))
            img = img.resize((new_w, new_h), Image.LANCZOS)
            canvas = Image.new("L", (size, size), color=255)
            ox = (size - new_w) // 2
            oy = (size - new_h) // 2
            canvas.paste(img, (ox, oy))
            img = canvas

    return img


def predict_with_cnn(strokes, frame_w=1920, frame_h=1080):
    if not os.path.exists(CNN_PYTHON):
        return None
    if not os.path.exists(CNN_SCRIPT):
        return None

    img = render_trajectory(strokes, frame_w=frame_w, frame_h=frame_h)
    tmp_path = os.path.join(tempfile.gettempdir(), "gesture_traj.png")
    img.save(tmp_path)

    try:
        cmd = [CNN_PYTHON, CNN_SCRIPT, tmp_path]
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=15,
            cwd=os.path.dirname(CNN_SCRIPT),
        )
        if proc.returncode != 0:
            return None
        stdout = proc.stdout.strip()
        if not stdout:
            return None
        result = json.loads(stdout)
        if "error" in result:
            return None
        return result
    except (subprocess.TimeoutExpired, json.JSONDecodeError, Exception):
        return None


# ══════════════════════════════════════════════════════════
# 手部骨架绘制（双色模式）
# ══════════════════════════════════════════════════════════

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (0, 17),
    (17, 18), (18, 19), (19, 20),
]


def draw_hand_landmarks(img, landmarks, img_w, img_h, color=None):
    """
    纯 OpenCV 绘制 21 个关键点 + 骨架连线。
    color: (B,G,R) 元组，None 则使用默认配色
    """
    if color is None:
        joint_color = (0, 220, 100)
        line_color = (80, 180, 255)
    else:
        joint_color = color
        line_color = tuple(min(255, c + 60) for c in color)

    pts = {}
    for i, lm in enumerate(landmarks):
        x, y = int(lm.x * img_w), int(lm.y * img_h)
        pts[i] = (x, y)
        cv2.circle(img, (x, y), 4, joint_color, -1)
        cv2.circle(img, (x, y), 6, (255, 255, 255), 1)
    for a, b in HAND_CONNECTIONS:
        if a in pts and b in pts:
            cv2.line(img, pts[a], pts[b], line_color, 2)
    return pts.get(0)


class FPS:
    def __init__(self, alpha=0.9):
        self.prev = time.time()
        self.val = 0.0
        self.alpha = alpha

    def tick(self):
        now = time.time()
        dt = now - self.prev
        self.prev = now
        if dt > 0:
            self.val = self.alpha * self.val + (1 - self.alpha) * (1.0 / dt)
        return self.val


# ══════════════════════════════════════════════════════════
# 右侧信息面板（一次 PIL 批量渲染）
# ══════════════════════════════════════════════════════════

def draw_side_panel(img, hands_info, fps, extra=None):
    if extra is None:
        extra = {}
    drawing_mode = extra.get("drawing_mode", False)
    drawing_paused = extra.get("drawing_paused", False)
    stroke_count = extra.get("stroke_count", 0)
    traj_result = extra.get("traj_result", None)
    h, w = img.shape[:2]
    px = w - 310

    pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    d = ImageDraw.Draw(pil_img)

    d.rectangle([px, 0, w, h], fill=(28, 28, 30))

    X0 = px + 15
    XR = w - 15
    LINE_COLOR = (72, 72, 74)
    BG_COLOR = (52, 52, 54)

    def hline(y):
        d.line([(X0, y), (XR, y)], fill=LINE_COLOR, width=1)

    def rect(x1, y1, x2, y2, color):
        d.rectangle([x1, y1, x2, y2], fill=color)

    y = 14

    # 标题
    d.text((X0, y), "手势识别系统", font=_FONT_SMALL, fill=(100, 220, 255))
    y += 32

    if drawing_mode:
        if drawing_paused:
            d.text((X0 + 130, y - 28), f"[已暂停 {stroke_count}笔]",
                   font=_load_font(14), fill=(100, 200, 255))
        else:
            d.text((X0 + 130, y - 28), f"[书写中 {stroke_count}笔]",
                   font=_load_font(14), fill=(80, 255, 255))
        d.rectangle([px + 2, 2, w - 2, h - 2], outline=(255, 200, 0), width=2)
    elif traj_result:
        src_tag = traj_result.get("source", "")
        d.text((X0 + 110, y - 28), f"[{src_tag}] {traj_result['display']}",
               font=_load_font(16), fill=(255, 200, 100))

    fps_color = (0, 220, 0) if fps >= 25 else (0, 180, 255) if fps >= 15 else (0, 100, 255)
    d.text((X0, y), f"FPS: {fps:.0f}", font=_FONT_XS, fill=fps_color)
    y += 26
    hline(y)
    y += 12

    # ── 左手信息 ──
    left = hands_info.get("left")
    if left and left["gesture"] != "无手势":
        d.text((X0, y), "🟦 左手 (控制)", font=_load_font(16), fill=(100, 180, 255))
        y += 22
        d.text((X0 + 5, y), left["gesture"], font=_FONT_MID, fill=(120, 200, 255))
        y += 28
        d.text((X0 + 5, y), left["action"], font=_FONT_XS, fill=(200, 200, 200))
        y += 20
    else:
        d.text((X0, y), "🟦 左手 —", font=_load_font(16), fill=(100, 100, 100))
        y += 22
        d.text((X0 + 5, y), "未检测到", font=_FONT_MID, fill=(100, 100, 100))
        y += 28
    hline(y)
    y += 10

    # ── 右手信息 ──
    right = hands_info.get("right")
    if right and right["gesture"] != "无手势":
        d.text((X0, y), "🟩 右手 (书写)", font=_load_font(16), fill=(100, 255, 150))
        y += 22
        d.text((X0 + 5, y), right["gesture"], font=_FONT_MID, fill=(120, 255, 160))
        y += 28
        d.text((X0 + 5, y), right["action"], font=_FONT_XS, fill=(200, 200, 200))
        y += 20
    else:
        d.text((X0, y), "🟩 右手 —", font=_load_font(16), fill=(100, 100, 100))
        y += 22
        d.text((X0 + 5, y), "未检测到", font=_FONT_MID, fill=(100, 100, 100))
        y += 28
    hline(y)
    y += 12

    # 操作提示（非书写模式下）
    if not drawing_mode and not traj_result:
        d.text((X0, y), "操作提示", font=_load_font(16), fill=(170, 170, 170))
        y += 24
        tips = [
            "左手 👌 OK → 书写模式",
            "左手 ✊ 拳头 → 暂停笔画",
            "左手 ✋ 张开 → 清空画布",
            "右手 ☝️ 食指 → 空中写字",
            "左手 👍 点赞 → 提交识别",
        ]
        for tip in tips:
            d.text((X0 + 5, y), tip, font=_FONT_MINI, fill=(190, 190, 190))
            y += 18

    # 轨迹概率分布
    probs = traj_result.get("probs", []) if traj_result else []
    if probs and len(probs) == 10:
        d.text((X0, y), "预测概率分布", font=_load_font(16), fill=(100, 220, 255))
        y += 24
        bar_w = 260
        ranked = sorted(enumerate(probs), key=lambda x: -x[1])
        for rank, (dgt, p) in enumerate(ranked):
            bh = 8
            by = y + rank * (bh + 2)
            bw = int(bar_w * p)
            if rank == 0:
                bc = (255, 200, 50)
            elif rank < 3:
                bc = (100, 200, 255)
            else:
                bc = (70, 70, 80)
            d.rectangle([X0, by, X0 + bw, by + bh], fill=bc)
            d.text((X0 + bw + 4, by - 3), f"{dgt}:{p*100:.1f}%",
                   font=_load_font(11), fill=(190, 190, 190))
        y += 10 * (8 + 2) + 6

    hline(y)
    y += 12

    # ── 手势清单 ──
    all_g = ["手掌张开", "拳头", "食指", "胜利", "OK", "点赞"]
    gcol = {
        "手掌张开": (200, 255, 200), "拳头": (255, 200, 200),
        "食指": (180, 255, 180), "胜利": (180, 200, 255),
        "OK": (255, 180, 255), "点赞": (255, 255, 160),
    }
    short_map = {"手掌张开": "[开]", "拳头": "[拳]", "食指": "[指]",
                 "胜利": "[V]", "OK": "[OK]", "点赞": "[赞]"}

    active_left = hands_info.get("left", {}).get("gesture", "")
    active_right = hands_info.get("right", {}).get("gesture", "")

    for gn in all_g:
        is_active = (gn == active_left or gn == active_right)
        box_color = gcol[gn] if is_active else BG_COLOR
        text_color = (0, 0, 0) if is_active else gcol[gn]
        rect(X0, y, X0 + 25, y + 22, box_color)
        d.text((X0 + 33, y + 2), f"{short_map[gn]} {gn}", font=_FONT_XXS, fill=text_color)
        y += 24

    # ── 底部状态栏 ──
    bar_h = 26
    bar_y = h - bar_h
    bar_color = (40, 40, 42)
    if drawing_mode:
        bar_color = (30, 30, 35)
    d.rectangle([0, bar_y, w, h], fill=bar_color)

    if drawing_mode:
        if drawing_paused:
            hint = "左手:✊继续新笔  ·  ✋清空  ·  👍识别退出"
        else:
            hint = "右手:✍️写字  ·  左手:✊暂停  ·  ✋清空  ·  👍识别退出"
        d.text((10, bar_y + 3), hint, font=_FONT_XS, fill=(0, 255, 255))
        if stroke_count > 0:
            d.text((w - 150, bar_y + 3), f"共 {stroke_count} 笔",
                   font=_FONT_XS, fill=(200, 200, 200))
    else:
        d.text((10, bar_y + 3), "左手OK:书写模式 | ESC:退出 | S:截图 | F:全屏",
               font=_FONT_XS, fill=(160, 160, 160))

    # 暂停覆盖文字
    if drawing_paused and drawing_mode:
        pause_text = f"已暂停 (共 {stroke_count} 笔)"
        bbox = d.textbbox((0, 0), pause_text, font=_FONT_MID)
        tw = bbox[2] - bbox[0]
        tx = (w - px) // 2 + px - tw // 2
        d.rectangle([tx - 10, 40, tx + tw + 10, 72], fill=(0, 0, 0))
        d.text((tx, 42), pause_text, font=_FONT_MID, fill=(0, 220, 255))

    img[:] = cv2.cvtColor(np.array(pil_img.convert("RGB")), cv2.COLOR_RGB2BGR)
    return img


def _make_empty_hand_info():
    return {"gesture": "无手势", "action": "—", "confidence": 0.0, "finger_states": [False] * 5}


# ══════════════════════════════════════════════════════════
# 主循环
# ══════════════════════════════════════════════════════════

def main():
    print("=" * 58)
    print("  GestureInteractionSystem")
    print("  基于 MediaPipe Task API 的实时手势识别系统")
    print("  🖐️  双手模式：左手控制 | 右手书写")
    print("=" * 58)

    _init_fonts()

    # ── 1. 摄像头 ──
    print("\n[1/3] 初始化摄像头...")
    cap = cv2.VideoCapture(CAMERA_ID)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    if not cap.isOpened():
        print("[ERROR] 无法打开摄像头！")
        return
    real_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    real_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"      分辨率: {real_w}x{real_h}")

    # ── 2. MediaPipe ──
    print("[2/3] 加载 MediaPipe 手部检测模型...")
    if not os.path.exists(MODEL_PATH):
        print(f"[ERROR] 模型文件不存在: {MODEL_PATH}")
        cap.release()
        return

    base_opts = mp_python.BaseOptions(model_asset_path=MODEL_PATH)
    opts = vision.HandLandmarkerOptions(
        base_options=base_opts,
        num_hands=2,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
        running_mode=vision.RunningMode.VIDEO,
    )
    landmarker = vision.HandLandmarker.create_from_options(opts)
    print("      ✓")

    # ── 3. 识别器 ──
    print("[3/3] 初始化手势识别引擎...")
    recognizer = GestureRecognizer()
    left_stabilizer = GestureStabilizer(window_size=12, min_ratio=0.55, lock_frames=5)
    right_stabilizer = GestureStabilizer(window_size=12, min_ratio=0.55, lock_frames=5)
    fps_counter = FPS()

    print("\n  系统就绪！ 🦞")
    print("  ─────────────────────────────")
    print("  🖐️  双手模式：")
    print("     左手(蓝) — 控制: OK切换/拳头暂停/手掌清空/点赞提交")
    print("     右手(绿) — 书写: 食指写字")
    print("  ─────────────────────────────")
    print("  ESC  退出    S  截图    F  全屏")
    if os.path.exists(CNN_PYTHON) and os.path.exists(CNN_SCRIPT):
        print(f"  CNN 模型: 已检测到 ✅")
    else:
        print(f"  CNN 模型: 未找到 ❌ (回退 $1 识别器)")
    print("  ─────────────────────────────\n")

    window = "GestureInteractionSystem"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window, 1280, 720)
    fullscreen = False
    frame_ts = 0

    # ── 手势跟踪（防止重复触发） ──
    prev_left_gesture = None
    prev_right_gesture = None

    # ── 轨迹识别状态 ──
    drawing_mode = False
    traj_strokes = []
    traj_current = []
    drawing_paused = False
    traj_result = None
    traj_timer = 0
    traj_recognizer = TrajectoryRecognizer()

    # ── 辅助函数（闭包，引用 main 的局部变量） ──

    def _toggle_pause():
        nonlocal drawing_paused, traj_current, traj_strokes, traj_result, right_hand_down, right_cur
        if not drawing_paused:
            if traj_current:
                traj_strokes.append(traj_current)
                traj_current = []
            drawing_paused = True
            right_hand_down = (right_cur != "食指")  # 记录暂停时刻右手是否已收回
            print(f"  ⏸️  笔画 {len(traj_strokes)} 完成，暂停中...")
        else:
            traj_strokes = []
            traj_current = []
            drawing_paused = False
            traj_result = None
            right_hand_down = False
            print("  🗑️  全部笔画已清空")

    def _do_submit():
        nonlocal traj_strokes, traj_current, traj_result, traj_timer
        if traj_current:
            traj_strokes.append(traj_current)
            traj_current = []
        all_pts = [p for stroke in traj_strokes for p in stroke]
        if len(all_pts) > 10:
            cnn_result = predict_with_cnn(traj_strokes, frame_w=real_w, frame_h=real_h)
            if cnn_result:
                preview_img = render_trajectory(
                    traj_strokes, size=280, line_width=18, frame_w=real_w, frame_h=real_h)
                traj_result = {
                    "display": cnn_result["display"],
                    "confidence": cnn_result["confidence"],
                    "probs": cnn_result.get("probs", []),
                    "source": "CNN",
                    "preview": preview_img,
                }
                top3 = sorted(enumerate(cnn_result.get("probs", [])), key=lambda x: -x[1])[:3]
                tops = ", ".join(f"{d}:{p*100:.1f}%" for d, p in top3 if p > 0.05)
                print(f"  ✅ CNN: {cnn_result['display']} ({cnn_result['confidence']*100:.1f}%)  |  {tops}")
            else:
                preview_img = render_trajectory(
                    traj_strokes, size=280, line_width=18, frame_w=real_w, frame_h=real_h)
                r = traj_recognizer.recognize(all_pts)
                traj_result = {
                    "display": r["display"],
                    "confidence": r["confidence"],
                    "probs": [],
                    "source": "$1",
                    "preview": preview_img,
                }
                print(f"  ✅ $1: {r['display']}  (置信度 {r['confidence']*100:.0f}%)")
            traj_timer = 90
        else:
            print("  ❌ 轨迹太短，已忽略")
        traj_strokes = []
        traj_current = []

    def _track_index_finger(hand_lms):
        nonlocal traj_current
        if hand_lms is None:
            return
        ix = hand_lms[8].x
        iy = hand_lms[8].y
        if len(traj_current) == 0 or math.dist((ix, iy), traj_current[-1]) > 0.002:
            traj_current.append((ix, iy))

    # ── 暂停恢复冷却（防重复触发 + 防拉直线） ──
    resume_guard = 0      # > 0 时禁止再次触发 resume

    # ── 主循环 ──
    while True:
        ok, frame = cap.read()
        if not ok:
            time.sleep(0.05)
            continue

        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]

        # ── MediaPipe 推理 ──
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
        )
        frame_ts += 33
        result = landmarker.detect_for_video(mp_image, frame_ts)

        # ── 分离左右手 ──
        left_info = _make_empty_hand_info()
        right_info = _make_empty_hand_info()
        left_lms = None
        right_lms = None
        num_hands = 0

        if result.hand_landmarks and len(result.hand_landmarks) > 0:
            num_hands = len(result.hand_landmarks)
            for i, hand_lms in enumerate(result.hand_landmarks):
                handedness = "Right"
                if result.handedness and len(result.handedness) > i and len(result.handedness[i]) > 0:
                    handedness = result.handedness[i][0].category_name
                # 画面镜像反转了左右手，MediaPipe 检测到的 handedness 需取反
                handedness = "Left" if handedness == "Right" else "Right"

                if handedness == "Left":
                    left_lms = hand_lms
                    draw_hand_landmarks(frame, hand_lms, w, h, color=(220, 100, 50))
                    left_info = recognizer.recognize(hand_lms, handedness)
                else:
                    right_lms = hand_lms
                    draw_hand_landmarks(frame, hand_lms, w, h, color=(80, 220, 60))
                    right_info = recognizer.recognize(hand_lms, handedness)

        # ── 防抖 ──
        left_raw = left_info["gesture"]
        left_stable = left_stabilizer.update(left_raw)
        if left_stable != left_raw:
            left_info["gesture"] = left_stable
            left_info["action"] = GESTURE_ACTIONS.get(left_stable, "无")

        right_raw = right_info["gesture"]
        right_stable = right_stabilizer.update(right_raw)
        if right_stable != right_raw:
            right_info["gesture"] = right_stable
            right_info["action"] = GESTURE_ACTIONS.get(right_stable, "无")

        left_cur = left_info["gesture"]
        right_cur = right_info["gesture"]

        # ── 控制逻辑 ──
        single_hand = (num_hands == 1)

        if single_hand:
            # ── 单手回退（原版混合逻辑） ──
            if left_lms:
                active_cur = left_cur
                active_info = left_info
                active_lms = left_lms
            else:
                active_cur = right_cur
                active_info = right_info
                active_lms = right_lms

            prev = prev_left_gesture if left_lms else prev_right_gesture

            # OK 切换
            if active_cur == "OK" and active_cur != prev and active_info["confidence"] > 0.7:
                if left_lms:
                    prev_left_gesture = active_cur
                else:
                    prev_right_gesture = active_cur
                drawing_mode = not drawing_mode
                if drawing_mode:
                    traj_strokes = []
                    traj_current = []
                    drawing_paused = False
                    traj_result = None
                    print("  ✏️  单手 · 进入书写")
                else:
                    _do_submit()
                    print("  👋  单手 · 退出书写")

            # 手掌清空
            if drawing_mode and active_cur == "手掌张开" and active_cur != prev:
                if left_lms:
                    prev_left_gesture = active_cur
                else:
                    prev_right_gesture = active_cur
                traj_strokes = []
                traj_current = []
                drawing_paused = False
                traj_result = None
                print("  🧹 画布已清空")

            # 拳头暂停/继续
            if drawing_mode and active_cur == "拳头" and active_cur != prev:
                if left_lms:
                    prev_left_gesture = active_cur
                else:
                    prev_right_gesture = active_cur
                _toggle_pause()

            # 食指追踪
            if drawing_mode and not drawing_paused and active_cur == "食指":
                _track_index_finger(active_lms)

            # 暂停下继续：只用右手食指 + 冷却 + 右手必须收回再伸出
            if drawing_mode and drawing_paused and active_cur == "食指" and active_info["confidence"] > 0.75 and resume_guard <= 0:
                if not left_lms:
                    # 只有右手时，单手上也是 index 续画
                    pass
                traj_current = []
                drawing_paused = False
                resume_guard = 10
                print(f"  ✍️  继续第 {len(traj_strokes) + 1} 笔...")

            # 日志
            if active_cur != prev and active_cur not in ("OK", "拳头", "手掌张开", "无手势"):
                if left_lms:
                    prev_left_gesture = active_cur
                else:
                    prev_right_gesture = active_cur
                if not drawing_mode:
                    print(f"  🎯 {active_cur}  →  {active_info['action']}")

            # 清空 prev 以便下次触发
            if active_cur == "无手势":
                if left_lms:
                    prev_left_gesture = None
                else:
                    prev_right_gesture = None

        else:
            # ── 双手模式 ──

            # 左手 OK → 切换书写
            if left_cur == "OK" and left_cur != prev_left_gesture and left_info["confidence"] > 0.7:
                prev_left_gesture = left_cur
                drawing_mode = not drawing_mode
                if drawing_mode:
                    traj_strokes = []
                    traj_current = []
                    drawing_paused = False
                    traj_result = None
                    print("  ✏️  双手 · 进入书写 — 右手写字，左手控制")
                else:
                    _do_submit()
                    print("  👋  双手 · 退出书写")

            # 左手 手掌 → 清空
            if drawing_mode and left_cur == "手掌张开" and left_cur != prev_left_gesture:
                prev_left_gesture = left_cur
                traj_strokes = []
                traj_current = []
                drawing_paused = False
                traj_result = None
                print("  🧹 画布已清空")

            # 左手 拳头 → 暂停/继续
            if drawing_mode and left_cur == "拳头" and left_cur != prev_left_gesture:
                prev_left_gesture = left_cur
                _toggle_pause()

            # 左手 点赞 → 提交并退出
            if drawing_mode and left_cur == "点赞" and left_cur != prev_left_gesture and left_info["confidence"] > 0.75:
                prev_left_gesture = left_cur
                _do_submit()
                drawing_mode = False
                drawing_paused = False
                print("  👋  点赞提交 · 退出书写")

            # 右手 食指 → 追踪轨迹
            if drawing_mode and not drawing_paused and right_cur == "食指":
                _track_index_finger(right_lms)

            # 暂停下右手食指 → 继续（必须右手收回再伸出，防止左手松开时误续画）
            if drawing_mode and drawing_paused and right_cur == "食指" and right_info["confidence"] > 0.75 and right_hand_down and resume_guard <= 0:
                prev_right_gesture = right_cur
                traj_current = []
                drawing_paused = False
                right_hand_down = False
                resume_guard = 10
                print(f"  ✍️  右手续画第 {len(traj_strokes) + 1} 笔...")

            # 日志
            if left_cur != prev_left_gesture and left_cur not in ("OK", "拳头", "手掌张开", "无手势"):
                prev_left_gesture = left_cur
                if not drawing_mode:
                    print(f"  🎯 [左手] {left_cur}  →  {left_info['action']}")

            if right_cur != prev_right_gesture and right_cur not in ("无手势",):
                prev_right_gesture = right_cur
                if not drawing_mode:
                    print(f"  🎯 [右手] {right_cur}  →  {right_info['action']}")

            # 重置 prev 允许下次触发
            if left_cur == "无手势":
                prev_left_gesture = None
            if right_cur == "无手势":
                prev_right_gesture = None

        # ── 暂停中：右手收回即标记可恢复 ──
        if drawing_paused and right_cur != "食指":
            right_hand_down = True

        # ── resume 冷却自减 ──
        if resume_guard > 0:
            resume_guard -= 1

        # ── 绘制轨迹 ──
        for stroke in traj_strokes:
            if len(stroke) >= 2:
                pts = [(int(x * w), int(y * h)) for x, y in stroke]
                for i in range(1, len(pts)):
                    cv2.line(frame, pts[i - 1], pts[i], (0, 220, 220), 2)
                cv2.circle(frame, pts[0], 4, (0, 180, 200), -1)
                cv2.circle(frame, pts[-1], 4, (0, 160, 180), -1)

        if traj_current and len(traj_current) >= 2:
            pts = [(int(x * w), int(y * h)) for x, y in traj_current]
            for i in range(1, len(pts)):
                cv2.line(frame, pts[i - 1], pts[i], (0, 255, 255), 3)
            cv2.circle(frame, pts[0], 5, (0, 200, 255), -1)
            if not drawing_paused:
                cv2.circle(frame, pts[-1], 7, (0, 255, 255), -1)

        if traj_timer > 0:
            traj_timer -= 1
            if traj_timer == 0:
                traj_result = None

        # ── 面板 ──
        hands_info = {"left": left_info, "right": right_info}
        panel_extra = {
            "drawing_mode": drawing_mode,
            "drawing_paused": drawing_paused,
            "stroke_count": len(traj_strokes) + (1 if traj_current else 0),
            "traj_result": traj_result,
        }

        fps = fps_counter.tick()
        frame = draw_side_panel(frame, hands_info, fps, panel_extra)

        # ── 轨迹预览（带醒目识别结果标签） ──
        preview = traj_result.get("preview") if traj_result else None
        if preview:
            pv = preview.copy()
            pv_w, pv_h = pv.size
            max_display = 220
            scale = max_display / max(pv_w, pv_h)
            new_w, new_h = int(pv_w * scale), int(pv_h * scale)
            pv = pv.resize((new_w, new_h), Image.LANCZOS)
            bx, by = 10, h - new_h - 50
            bw, bh = new_w + 6, new_h + 6

            # 识别结果标签（大号，黄底黑字）
            src = traj_result.get('source', '?')
            display = traj_result['display']
            label = f"{src}: {display}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_DUPLEX, 1.2, 3)
            label_bx = bx
            label_by = by - th - 14
            label_bw = tw + 20
            label_bh = th + 12
            # 黄色背景条
            cv2.rectangle(frame,
                          (label_bx, label_by),
                          (label_bx + label_bw, label_by + label_bh),
                          (50, 210, 255), -1)
            cv2.rectangle(frame,
                          (label_bx, label_by),
                          (label_bx + label_bw, label_by + label_bh),
                          (0, 0, 0), 2)
            # 黑色大字
            cv2.putText(frame, label,
                        (label_bx + 10, label_by + th + 6),
                        cv2.FONT_HERSHEY_DUPLEX, 1.2, (0, 0, 0), 3)

            # 预览图外框
            cv2.rectangle(frame, (bx, by), (bx + bw, by + bh), (80, 80, 80), -1)
            cv2.rectangle(frame, (bx, by), (bx + bw, by + bh), (160, 160, 160), 1)
            cv2.rectangle(frame, (bx + 3, by + 3),
                          (bx + 3 + new_w, by + 3 + new_h), (255, 255, 255), -1)
            pv_np = np.array(pv)
            pv_bgr = cv2.cvtColor(pv_np, cv2.COLOR_GRAY2BGR)
            frame[by + 3:by + 3 + new_h, bx + 3:bx + 3 + new_w] = pv_bgr

        cv2.imshow(window, frame)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            break
        elif key == ord('s'):
            os.makedirs("screenshots", exist_ok=True)
            fn = f"screenshots/gesture_{time.strftime('%Y%m%d_%H%M%S')}.png"
            cv2.imwrite(fn, frame)
            print(f"  📸 {fn}")
        elif key == ord('f'):
            fullscreen = not fullscreen
            cv2.setWindowProperty(window, cv2.WND_PROP_FULLSCREEN,
                                  cv2.WINDOW_FULLSCREEN if fullscreen else cv2.WINDOW_NORMAL)

    cap.release()
    cv2.destroyAllWindows()
    landmarker.close()
    print("  再见！🦞")


if __name__ == "__main__":
    main()
