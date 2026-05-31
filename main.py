"""
GestureInteractionSystem - 主程序入口

基于 MediaPipe Task API + OpenCV 的实时手势识别系统
支持 6 种静态手势识别，并映射到对应的操作。

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

# 模型文件路径（与 main.py 同目录）
_MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(_MODULE_DIR, "hand_landmarker.task")

# CNN 预测模型路径（项目内置，需在当前环境安装 torch）
CNN_PYTHON = sys.executable  # 当前 Python 解释器
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


_FONT_LARGE = None   # 34px
_FONT_MID = None     # 24px
_FONT_SMALL = None   # 22px
_FONT_XS = None      # 17px
_FONT_XXS = None     # 15px
_FONT_MINI = None    # 14px


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


def render_trajectory(strokes, size=280, line_width=22, margin=15, crop_to_content=True, frame_w=1920, frame_h=1080):
    """
    将多笔画轨迹渲染为白底黑字图片。

    Args:
        strokes: [[(x,y),...], ...] 归一化坐标 (0-1)
        size: 输出尺寸
        line_width: 线宽
        margin: 边距
        crop_to_content: 是否裁切到内容区域
        frame_w, frame_h: 摄像头原始分辨率（用于修正横纵比）

    关键：摄像头画面是 frame_w:frame_h（如 16:9），
    归一化坐标 x∈[0,1] y∈[0,1] 对应不同物理尺度。
    y 方向需要乘以 (frame_w/frame_h) 来纠正比例。
    """
    from PIL import ImageDraw as PILD
    from PIL import ImageOps as PILOps

    # 横纵比修正系数
    aspect = frame_w / frame_h  # e.g. 1920/1080 = 1.778

    # 先用较大分辨率绘制
    draw_size = size * 2 if crop_to_content else size
    ds = draw_size
    m = margin * (draw_size / size)

    # 创建画布（宽度按 size，高度按比例缩小，使实际比例匹配摄像头）
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
        pts = [(int(x * (canvas_w - 2 * m) + m), int(y * (canvas_h - 2 * m) + m)) for x, y in stroke]
        draw.line(pts, fill=0, width=line_width)
        for px, py in (pts[0], pts[-1]):
            r = line_width // 2
            draw.ellipse([px - r, py - r, px + r, py + r], fill=0)

    # 裁切到内容区域
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

        # 伸缩到目标正方形（保持宽高比，空白处填白）
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
    """调用自训练 CNN 模型预测轨迹（详细日志版）。"""
    if not os.path.exists(CNN_PYTHON):
        print(f"  [WARN] Conda Python 不存在: {CNN_PYTHON}")
        return None
    if not os.path.exists(CNN_SCRIPT):
        print(f"  [WARN] predict_api.py 不存在: {CNN_SCRIPT}")
        return None

    img = render_trajectory(strokes, frame_w=frame_w, frame_h=frame_h)
    tmp_path = os.path.join(tempfile.gettempdir(), "gesture_traj.png")
    img.save(tmp_path)
    print(f"  [CNN] 图片: {tmp_path}")

    try:
        cmd = [CNN_PYTHON, CNN_SCRIPT, tmp_path]
        print(f"  [CNN] 执行: {' '.join(cmd)}")
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=15,
            cwd=os.path.dirname(CNN_SCRIPT),
        )
        print(f"  [CNN] 退出码: {proc.returncode}")
        if proc.stderr and proc.stderr.strip():
            print(f"  [CNN] stderr: {proc.stderr.strip()[:300]}")
        if proc.returncode != 0:
            return None
        stdout = proc.stdout.strip()
        if not stdout:
            print("  [WARN] CNN 无输出")
            return None
        result = json.loads(stdout)
        if "error" in result:
            print(f"  [WARN] CNN 错误: {result['error']}")
            return None
        return result
    except subprocess.TimeoutExpired:
        print("  [WARN] CNN 超时 (15s)")
        return None
    except json.JSONDecodeError as e:
        print(f"  [WARN] CNN 输出非 JSON: {e}")
        return None
    except Exception as e:
        print(f"  [WARN] CNN 异常: {e}")
        return None


# ══════════════════════════════════════════════════════════
# 手部骨架绘制（纯 OpenCV，无 PIL）
# ══════════════════════════════════════════════════════════

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (0, 17),
    (17, 18), (18, 19), (19, 20),
]


def draw_hand_landmarks(img, landmarks, img_w, img_h):
    """纯 OpenCV 绘制 21 个关键点 + 骨架连线"""
    pts = {}
    for i, lm in enumerate(landmarks):
        x, y = int(lm.x * img_w), int(lm.y * img_h)
        pts[i] = (x, y)
        cv2.circle(img, (x, y), 4, (0, 220, 100), -1)
        cv2.circle(img, (x, y), 6, (255, 255, 255), 1)
    for a, b in HAND_CONNECTIONS:
        if a in pts and b in pts:
            cv2.line(img, pts[a], pts[b], (80, 180, 255), 2)
    return img


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
# 右侧信息面板（一次 PIL 批量渲染所有中文）
# ══════════════════════════════════════════════════════════

def draw_side_panel(img, gesture_info, fps, extra=None):
    """
    纯 PIL 渲染右侧信息面板。所有文字、线条、色块均在 PIL 画布上绘制，
    全程只做一次 BGR→PIL→BGR 转换，彻底避免坐标混乱。
    """
    if extra is None:
        extra = {}
    drawing_mode = extra.get("drawing_mode", False)
    drawing_paused = extra.get("drawing_paused", False)
    stroke_count = extra.get("stroke_count", 0)
    traj_result = extra.get("traj_result", None)
    h, w = img.shape[:2]
    px = w - 310  # 面板左边界

    name = gesture_info.get("gesture", "无手势")
    conf = gesture_info.get("confidence", 0)
    action = gesture_info.get("action", "—")

    # ── 一次转 PIL ──
    pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    d = ImageDraw.Draw(pil_img)

    # 面板背景（纯色，简单可靠）
    d.rectangle([px, 0, w, h], fill=(28, 28, 30))

    # ── 布局参数（所有 y 坐标均以 PIL 为准） ──
    X0 = px + 15       # 文字左边界
    XR = w - 15        # 右边界
    LINE_COLOR = (72, 72, 74)
    BG_COLOR = (52, 52, 54)

    def hline(y):
        d.line([(X0, y), (XR, y)], fill=LINE_COLOR, width=1)

    def rect(x1, y1, x2, y2, color):
        d.rectangle([x1, y1, x2, y2], fill=color)

    y = 14

    # ── 标题行 ──
    d.text((X0, y), "手势识别系统", font=_FONT_SMALL, fill=(100, 220, 255))
    y += 32

    # 书写模式指示器
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

    # FPS（用 PIL 画，避免与标题重叠）
    fps_color = (0, 220, 0) if fps >= 25 else (0, 180, 255) if fps >= 15 else (0, 100, 255)
    d.text((X0, y), f"FPS: {fps:.0f}", font=_FONT_XS, fill=fps_color)
    y += 26
    hline(y)
    y += 12

    # ── 当前手势（大字） ──
    gc = (0, 255, 120) if name not in ("无手势",) else (140, 140, 140)
    d.text((X0, y), name, font=_FONT_LARGE, fill=gc)
    y += 46

    # 置信度进度条
    if conf > 0:
        bar_w = 270
        bar_h = 6
        rect(X0, y, X0 + bar_w, y + bar_h, BG_COLOR)
        fw = int(bar_w * min(conf, 1.0))
        bc = (0, 230, 80) if conf > 0.8 else (0, 180, 240) if conf > 0.5 else (0, 100, 240)
        rect(X0, y, X0 + fw, y + bar_h, bc)
        y += 8
        d.text((X0 + 5, y), f"{conf*100:.0f}%", font=_load_font(14), fill=(190, 190, 190))
        y += 16
    y += 8

    # ── 操作 ──
    d.text((X0, y), "操作:", font=_FONT_XS, fill=(180, 180, 180))
    y += 22
    d.text((X0, y), action, font=_FONT_MID, fill=(255, 210, 100))
    y += 34
    hline(y)
    y += 12

    # ── 手指状态 / 概率分布 ──
    probs = traj_result.get("probs", []) if traj_result else []
    if probs and len(probs) == 10:
        # 有 CNN 概率分布 → 显示 Top-10 条状图
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
    else:
        # 正常显示手指状态
        d.text((X0, y), "手指状态", font=_load_font(16), fill=(170, 170, 170))
        y += 26
        fingers = ["拇指", "食指", "中指", "无名指", "小指"]
        states = gesture_info.get("finger_states", [False] * 5)
        for i, (fn, st) in enumerate(zip(fingers, states)):
            cx = X0 + (i & 1) * 145
            cy = y + (i // 2) * 26
            sc = (100, 240, 100) if st else (240, 100, 100)
            stxt = "v 伸直" if st else "x 弯曲"
            d.text((cx, cy), f"{fn}: {stxt}", font=_FONT_MINI, fill=sc)
        y += 80
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

    for gn in all_g:
        is_active = (name == gn)
        box_color = gcol[gn] if is_active else BG_COLOR
        text_color = (0, 0, 0) if is_active else gcol[gn]
        rect(X0, y, X0 + 25, y + 22, box_color)
        d.text((X0 + 33, y + 2), f"{short_map[gn]} {gn}", font=_FONT_XXS, fill=text_color)
        y += 24

    # ── 底部状态栏（PIL 中文，替换 cv2.putText 乱码） ──
    bar_h = 26
    bar_y = h - bar_h
    bar_color = (40, 40, 42)
    if drawing_mode:
        bar_color = (30, 30, 35)
    d.rectangle([0, bar_y, w, h], fill=bar_color)

    if drawing_mode:
        if drawing_paused:
            hint = "食指:新笔画  |  手掌:清空  |  OK:识别"
        else:
            hint = "握拳:暂停  |  手掌:清空  |  OK:识别 & 退出"
        d.text((10, bar_y + 3), hint, font=_FONT_XS, fill=(0, 255, 255))
        # 笔画计数
        if stroke_count > 0:
            d.text((w - 150, bar_y + 3), f"共 {stroke_count} 笔",
                   font=_FONT_XS, fill=(200, 200, 200))
    else:
        d.text((10, bar_y + 3), "OK:书写模式 | ESC:退出 | S:截图 | F:全屏",
               font=_FONT_XS, fill=(160, 160, 160))

    # ── 暂停覆盖文字（画面中央） ──
    if drawing_paused and drawing_mode:
        pause_text = f"已暂停 (共 {stroke_count} 笔)"
        bbox = d.textbbox((0, 0), pause_text, font=_FONT_MID)
        tw = bbox[2] - bbox[0]
        tx = (w - px) // 2 + px - tw // 2
        d.rectangle([tx - 10, 40, tx + tw + 10, 72], fill=(0, 0, 0))
        d.text((tx, 42), pause_text, font=_FONT_MID, fill=(0, 220, 255))

    # ── 一次转回 OpenCV ──
    img[:] = cv2.cvtColor(np.array(pil_img.convert("RGB")), cv2.COLOR_RGB2BGR)
    return img


# ══════════════════════════════════════════════════════════
# 主循环
# ══════════════════════════════════════════════════════════

def main():
    print("=" * 58)
    print("  GestureInteractionSystem")
    print("  基于 MediaPipe Task API 的实时手势识别系统")
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
        num_hands=1,
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
    stabilizer = GestureStabilizer(window_size=15, min_ratio=0.55, lock_frames=6)
    fps_counter = FPS()

    print("\n  系统就绪！ 🦞")
    print("  ─────────────────────────────")
    print("  ESC  退出    S  截图    F  全屏")
    # CNN 可用性检查
    if os.path.exists(CNN_PYTHON) and os.path.exists(CNN_SCRIPT):
        print(f"  CNN 模型: 已检测到 ✅")
    else:
        print(f"  CNN 模型: 未找到 ❌ (回退 $1 识别器)")
        if not os.path.exists(CNN_PYTHON):
            print(f"            路径不存在: {CNN_PYTHON}")
        if not os.path.exists(CNN_SCRIPT):
            print(f"            路径不存在: {CNN_SCRIPT}")
    print("  ─────────────────────────────\n")

    window = "GestureInteractionSystem"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window, 1280, 720)
    fullscreen = False
    last_gesture = None
    frame_ts = 0

    # ── 轨迹识别状态 ──
    drawing_mode = False
    traj_strokes = []          # 已完成笔画 [[(x,y),...], ...]
    traj_current = []           # 当前笔画 [(x,y),...]
    drawing_paused = False      # 笔抬起（暂停）
    traj_result = None
    traj_timer = 0
    traj_recognizer = TrajectoryRecognizer()

    while True:
        ok, frame = cap.read()
        if not ok:
            time.sleep(0.05)
            continue

        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]

        # MediaPipe 推理
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB,
                            data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        frame_ts += 33  # 约 30fps 的时间戳步进
        result = landmarker.detect_for_video(mp_image, frame_ts)

        gesture_info = {
            "gesture": "无手势", "action": "—",
            "confidence": 0.0, "finger_states": [False] * 5,
        }

        if result.hand_landmarks and len(result.hand_landmarks) > 0:
            hand_lms = result.hand_landmarks[0]
            handedness = "Right"
            if result.handedness and len(result.handedness) > 0:
                handedness = result.handedness[0][0].category_name

            frame = draw_hand_landmarks(frame, hand_lms, w, h)
            gesture_info = recognizer.recognize(hand_lms, handedness)

        # 防抖稳定：原始手势 → 稳定手势
        raw_gesture = gesture_info["gesture"]
        stable_gesture = stabilizer.update(raw_gesture)
        if stable_gesture != raw_gesture:
            gesture_info["gesture"] = stable_gesture
            gesture_info["action"] = GESTURE_ACTIONS.get(stable_gesture, "无")

        # ── 书写模式逻辑 ──
        cur = gesture_info["gesture"]

        # OK 手势切换书写模式（防重复触发，加置信度门槛）
        if cur == "OK" and cur != last_gesture and gesture_info["confidence"] > 0.7:
            last_gesture = cur
            drawing_mode = not drawing_mode
            if drawing_mode:
                traj_strokes = []
                traj_current = []
                drawing_paused = False
                traj_result = None
                print("  ✏️  进入书写模式 — 伸出食指写字，握拳暂停/起笔")
            else:
                # 退出时：如果有未完成笔画，先保存
                if traj_current:
                    traj_strokes.append(traj_current)
                    traj_current = []
                all_pts = [p for stroke in traj_strokes for p in stroke]
                if len(all_pts) > 10:
                    # 优先用 CNN 模型
                    cnn_result = predict_with_cnn(traj_strokes, frame_w=w, frame_h=h)
                    if cnn_result:
                        preview_img = render_trajectory(traj_strokes, size=280, line_width=18, frame_w=w, frame_h=h)
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
                        # CNN 不可用时回退 $1 识别器
                        preview_img = render_trajectory(traj_strokes, size=280, line_width=18, frame_w=w, frame_h=h)
                        result = traj_recognizer.recognize(all_pts)
                        traj_result = {
                            "display": result["display"],
                            "confidence": result["confidence"],
                            "probs": [],
                            "source": "$1",
                            "preview": preview_img,
                        }
                        print(f"  ✅ $1: {result['display']}  (置信度 {result['confidence']*100:.0f}%)")
                    traj_timer = 90
                else:
                    print("  ❌ 轨迹太短，已忽略")
                traj_strokes = []
                traj_current = []
                drawing_paused = False
                print("  👋 退出书写模式")

        # 书写模式下：手掌张开 = 清空画布
        if drawing_mode and cur == "手掌张开" and cur != last_gesture:
            last_gesture = cur
            traj_strokes = []
            traj_current = []
            drawing_paused = False
            traj_result = None
            print("  🧹 画布已清空")

        # 书写模式下：拳头 = 暂停/继续（笔起/笔落）
        if drawing_mode and cur == "拳头" and cur != last_gesture:
            last_gesture = cur
            if not drawing_paused:
                # 正在画 → 暂停，保存当前笔画（裁去末尾过渡噪声）
                if traj_current:
                    # 去掉最后 ~10 帧的过渡点（握拳过程中手指移动的拖尾）
                    trim = min(10, len(traj_current) // 4)
                    if trim > 0 and len(traj_current) > trim + 3:
                        traj_current = traj_current[:-trim]
                    traj_strokes.append(traj_current)
                    traj_current = []
                drawing_paused = True
                print(f"  ⏸️  笔画 {len(traj_strokes)} 完成，握拳暂停中...")
            else:
                # 已暂停 → 再次握拳清空全部笔画
                traj_strokes = []
                traj_current = []
                drawing_paused = False
                traj_result = None
                print("  🗑️  全部笔画已清空，可重新书写")

        # 书写模式下：记录食指指尖轨迹（仅在当前手势为食指时）
        if drawing_mode and result.hand_landmarks and not drawing_paused:
            # 只有稳定手势为「食指」时才记录轨迹，避免切换手势时的拖尾噪声
            if cur == "食指":
                hand_lms = result.hand_landmarks[0]
                ix = hand_lms[8].x
                iy = hand_lms[8].y
                if len(traj_current) == 0 or \
                   math.dist((ix, iy), traj_current[-1]) > 0.002:
                    traj_current.append((ix, iy))

        # 暂停状态下，食指重新伸出 → 自动开始新笔画
        if drawing_mode and drawing_paused and cur == "食指" and cur != last_gesture and gesture_info["confidence"] > 0.75:
            last_gesture = cur
            drawing_paused = False
            print(f"  ✍️  开始第 {len(traj_strokes) + 1} 笔...")

        # 手势切换日志（OK/拳头/手掌张开已在上面处理）
        if cur != last_gesture and cur not in ("OK", "拳头", "手掌张开"):
            last_gesture = cur
            if cur not in ("无手势",) and not drawing_mode:
                print(f"  🎯 {cur}  →  {gesture_info['action']}  (置信度 {gesture_info['confidence']*100:.0f}%)")

        # ── 绘制轨迹 ──
        h, w = frame.shape[:2]

        # 已完成笔画（实线）
        for stroke in traj_strokes:
            if len(stroke) >= 2:
                pts = [(int(x * w), int(y * h)) for x, y in stroke]
                for i in range(1, len(pts)):
                    cv2.line(frame, pts[i - 1], pts[i], (0, 220, 220), 2)
                cv2.circle(frame, pts[0], 4, (0, 180, 200), -1)     # 起点
                cv2.circle(frame, pts[-1], 4, (0, 160, 180), -1)    # 终点

        # 当前笔画（亮黄，闪烁端点）
        if traj_current and len(traj_current) >= 2:
            pts = [(int(x * w), int(y * h)) for x, y in traj_current]
            for i in range(1, len(pts)):
                cv2.line(frame, pts[i - 1], pts[i], (0, 255, 255), 3)
            cv2.circle(frame, pts[0], 5, (0, 200, 255), -1)
            if not drawing_paused:
                cv2.circle(frame, pts[-1], 7, (0, 255, 255), -1)    # 闪烁笔尖

        # 暂停指示器（已移入 draw_side_panel PIL 渲染）

        # 倒计时清除识别结果
        if traj_timer > 0:
            traj_timer -= 1
            if traj_timer == 0:
                traj_result = None

        # 传递给面板的额外信息
        panel_extra = {
            "drawing_mode": drawing_mode,
            "drawing_paused": drawing_paused,
            "stroke_count": len(traj_strokes) + (1 if traj_current else 0),
            "traj_result": traj_result,
            "preview": traj_result.get("preview") if traj_result else None,
        }

        # UI 面板（含底部状态栏，PIL 渲染全部中文）
        fps = fps_counter.tick()
        frame = draw_side_panel(frame, gesture_info, fps, panel_extra)

        # 轨迹预览图（左下角大图，自适应缩放不拉伸）
        preview = traj_result.get("preview") if traj_result else None
        if preview:
            h, w = frame.shape[:2]
            pv = preview.copy()
            pv_w, pv_h = pv.size
            # 自适应：取 min(width, height) 方向撑满，另一方向保持比例
            max_display = 280
            scale = max_display / max(pv_w, pv_h)
            new_w, new_h = int(pv_w * scale), int(pv_h * scale)
            pv = pv.resize((new_w, new_h), Image.LANCZOS)
            # 灰色背景板
            bx, by = 10, h - new_h - 50
            bw, bh = new_w + 6, new_h + 6
            cv2.rectangle(frame, (bx, by), (bx + bw, by + bh), (80, 80, 80), -1)
            cv2.rectangle(frame, (bx, by), (bx + bw, by + bh), (160, 160, 160), 1)
            # 白底
            cv2.rectangle(frame, (bx + 3, by + 3), (bx + 3 + new_w, by + 3 + new_h), (255, 255, 255), -1)
            pv_np = np.array(pv)
            pv_bgr = cv2.cvtColor(pv_np, cv2.COLOR_GRAY2BGR)
            frame[by + 3:by + 3 + new_h, bx + 3:bx + 3 + new_w] = pv_bgr
            # 标签
            label = f"{traj_result.get('source','')}: {traj_result['display']}"
            cv2.putText(frame, label, (bx, by - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

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
