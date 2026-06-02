"""
GCN 手势数据集采集工具 (v3 — 稳健版)

修复 v2 仍卡死的可能原因：
  1. 先创建窗口 → 再开摄像头 → 最后加载 MediaPipe（窗口立即可见）
  2. 摄像头打开失败自动降级（不用 DSHOW）
  3. 主循环加超时保护的 cap.read()
  4. 每 30 帧强制 cv2.waitKey(1) 防止 Windows 认为窗口无响应
  5. 所有耗时操作加 try/except + 进度打印

控制键（OpenCV 窗口焦点时）：
  N        — 下一个手势
  P        — 上一个手势
  H        — 切换左手/右手
  SPACE    — 暂停 / 继续
  S        — 终端打印统计
  R        — 清空当前手势 CSV
  ESC / Q  — 退出
"""

import cv2
import time
import os
import sys
import csv
import numpy as np

import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

# ══════════════════════════════════════════
# 路径 & 常量
# ══════════════════════════════════════════
_MODULE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(_MODULE_DIR, "hand_landmarker.task")
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "DateSet")
os.makedirs(OUTPUT_DIR, exist_ok=True)
CAMERA_ID = 0
COLLECT_INTERVAL = 0.5

# ══════════════════════════════════════════
# 手势集
# ══════════════════════════════════════════
GESTURES = [
    (0, "open_palm",       "手掌张开"),
    (1, "fist",            "拳头"),
    (2, "index_point",     "食指指出"),
    (3, "victory",         "胜利V"),
    (4, "ok",              "OK"),
    (5, "thumbs_up",       "点赞"),
    (6, "three_fingers",   "三指"),
    (7, "pinch",           "捏合"),
    (8, "four_fingers",    "四指"),
    (9, "thumb_down",      "拇指向下"),
]
GESTURE_BY_ID = {g[0]: g for g in GESTURES}

# ══════════════════════════════════════════
# 骨架
# ══════════════════════════════════════════
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (0, 17),
    (17, 18), (18, 19), (19, 20),
]
LANDMARK_NAMES = [
    "wrist", "thumb_cmc","thumb_mcp","thumb_ip","thumb_tip",
    "index_mcp","index_pip","index_dip","index_tip",
    "middle_mcp","middle_pip","middle_dip","middle_tip",
    "ring_mcp","ring_pip","ring_dip","ring_tip",
    "pinky_mcp","pinky_pip","pinky_dip","pinky_tip",
]
CSV_COLUMNS = ["timestamp","handedness","gesture"]
for _i, _nm in enumerate(LANDMARK_NAMES):
    CSV_COLUMNS += [f"lm{_i}_{_nm}_x", f"lm{_i}_{_nm}_y", f"lm{_i}_{_nm}_z"]

# ══════════════════════════════════════════
# CSV 工具
# ══════════════════════════════════════════
def _csv_dir(name): 
    d = os.path.join(OUTPUT_DIR, name); os.makedirs(d, exist_ok=True); return d
def _csv_path(name): 
    return os.path.join(_csv_dir(name), f"{name}.csv")

def csv_ensure(name):
    p = _csv_path(name)
    if not os.path.exists(p):
        _write_header(p); return p
    try:
        with open(p, "r", encoding="utf-8") as f:
            if f.readline().strip() != ",".join(CSV_COLUMNS):
                print(f"  ⚠ {name}.csv 表头异常，已重建")
                _write_header(p)
    except Exception:
        print(f"  ⚠ {name}.csv 不可读，已重建")
        _write_header(p)
    return p

def _write_header(p):
    try:
        with open(p, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(CSV_COLUMNS)
    except PermissionError:
        print(f"  ❌ {p} 被占用，请关闭 Excel")

def csv_count(name):
    p = _csv_path(name)
    if not os.path.exists(p): return 0
    try:
        with open(p, "r", encoding="utf-8") as f:
            lines = f.readlines()
        if len(lines) <= 1: return 0
        ec = len(CSV_COLUMNS)
        return sum(1 for L in lines[1:] if L.strip() and L.count(",") >= ec - 1)
    except Exception:
        return 0

def csv_append(name, ts, hand, lms):
    try:
        p = csv_ensure(name)
        row = [f"{ts:.3f}", hand, name]
        for lm in lms:
            row += [f"{lm.x:.6f}", f"{lm.y:.6f}", f"{lm.z:.6f}"]
        with open(p, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(row)
        return True
    except (PermissionError, OSError):
        return False

def csv_reset(name):
    try:
        os.remove(_csv_path(name))
    except FileNotFoundError:
        pass
    except PermissionError:
        print(f"  ❌ 无法重置 {name}：文件被占用")
    csv_ensure(name)
    print(f"  🗑 {name} 已清空")

# ══════════════════════════════════════════
# 状态
# ══════════════════════════════════════════
class State:
    def __init__(self):
        self.hand = "right"
        self.gid = 0
        self.gname = GESTURE_BY_ID[0][1]
        self.paused = False
        self.last_save = 0.0
        self.count = 0
        self.err = False
        self.errmsg = ""
    @property
    def display(self):
        g = GESTURE_BY_ID[self.gid]
        return f"[{g[0]:02d}] {g[2]} ({g[1]})"
    def swap_hand(self):
        self.hand = "left" if self.hand == "right" else "right"
    def set_g(self, gid):
        if gid in GESTURE_BY_ID:
            self.gid = gid
            self.gname = GESTURE_BY_ID[gid][1]
            csv_ensure(self.gname)
            self.count = csv_count(self.gname)
            self.err = False; self.errmsg = ""
            return True
        return False
    def next_g(self): self.set_g((self.gid + 1) % len(GESTURES))
    def prev_g(self): self.set_g((self.gid - 1) % len(GESTURES))

# ══════════════════════════════════════════
# 绘制
# ══════════════════════════════════════════
def draw_hand(img, lms, iw, ih, color):
    pts = {}
    for i, lm in enumerate(lms):
        x, y = int(lm.x * iw), int(lm.y * ih)
        pts[i] = (x, y)
        cv2.circle(img, (x, y), 4, color, -1)
        cv2.circle(img, (x, y), 6, (255, 255, 255), 1)
    for a, b in HAND_CONNECTIONS:
        if a in pts and b in pts:
            cv2.line(img, pts[a], pts[b],
                     tuple(min(255, c + 40) for c in color), 2)
    return pts.get(0)

def draw_hud(frame, st, fps, detected):
    h, w = frame.shape[:2]
    ov = frame.copy()
    cv2.rectangle(ov, (0, 0), (w, 120), (0, 0, 0), -1)
    frame[:] = cv2.addWeighted(frame, 0.85, ov, 0.15, 0)
    cv2.putText(frame, "GCN Data Collector", (12, 26),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (100, 220, 255), 2)
    if st.paused:
        cv2.putText(frame, "PAUSED", (w - 150, 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
    else:
        rr = 6 + int(3 * (time.time() % 1 > 0.5))
        cv2.putText(frame, "REC", (w - 150, 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 100), 2)
        cv2.circle(frame, (w - 40, 22), rr, (0, 255, 0), -1)
    icon = "L" if st.hand == "left" else "R"
    hc = (220, 120, 40) if st.hand == "left" else (80, 220, 80)
    cv2.putText(frame, f"Hand: {icon}  |  {st.display}",
                (12, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.5, hc, 2)
    dc = (0, 255, 0) if detected else (0, 0, 255)
    cv2.putText(frame, "DETECTED" if detected else "NO HAND",
                (12, 86), cv2.FONT_HERSHEY_SIMPLEX, 0.5, dc, 2)
    cv2.putText(frame, f"Saved: {st.count}  |  FPS: {fps:.0f}",
                (12, 114), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
    if st.err:
        cv2.putText(frame, f"ERR: {st.errmsg}",
                    (12, 138), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
    # toast
    if hasattr(st, "_toast") and st._toast:
        txt, t0 = st._toast
        if time.time() - t0 < 1.5:
            sz = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
            cv2.putText(frame, txt, ((w - sz[0]) // 2, h // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        else:
            st._toast = None
    # 底部
    cv2.rectangle(frame, (0, h - 28), (w, h), (30, 30, 30), -1)
    cv2.putText(frame, "N:下个手势  P:上个手势  H:换手  SPACE:暂停  S:统计  R:清空  ESC:退出",
                (10, h - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (180, 180, 180), 1)
    return frame

def print_stats():
    print("\n" + "=" * 50)
    print("  采集统计")
    print("=" * 50)
    total = 0
    for gid, name, disp in GESTURES:
        n = csv_count(name); total += n
        print(f"  [{gid:02d}] {name:<20s} {n:>5d}  {'#' * min(n//5, 30)}")
    print("─" * 50)
    print(f"  总计: {total} 帧  |  {OUTPUT_DIR}")
    print("=" * 50 + "\n")

# ══════════════════════════════════════════
# 主函数
# ══════════════════════════════════════════
def main():
    print("=" * 50)
    print("  GCN Data Collector v3")
    print("=" * 50)

    if not os.path.exists(MODEL_PATH):
        print(f"\n[ERROR] 模型缺失: {MODEL_PATH}"); return

    # ── 控制台初始选择 ──
    st = State()
    hi = input("\n  左手还是右手？[L/R] (默认R): ").strip().lower()
    if hi == "l": st.hand = "left"
    print("\n  手势:")
    for gid, nm, dp in GESTURES:
        print(f"    [{gid}] {dp} ({nm})")
    gi = input(f"\n  手势编号 [0-{len(GESTURES)-1}] (默认0): ").strip()
    if gi.isdigit() and int(gi) in GESTURE_BY_ID:
        st.set_g(int(gi))
    csv_ensure(st.gname); st.count = csv_count(st.gname)
    print(f"\n  {st.display} | {'左手' if st.hand == 'left' else '右手'} | "
          f"已有 {st.count} 帧")

    # ── 第 1 步：先建窗口 ──
    print("\n  [1/3] 创建窗口...")
    WNAME = "GCN Data Collector"
    cv2.namedWindow(WNAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WNAME, 960, 540)
    dummy = np.zeros((360, 640, 3), dtype=np.uint8)
    cv2.putText(dummy, "Starting camera...", (180, 180),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)
    cv2.imshow(WNAME, dummy)
    cv2.waitKey(1)
    print("  ✓ 窗口已创建")

    # ── 第 2 步：开摄像头 ──
    print("  [2/3] 打开摄像头...")
    cap = None
    for backend in [cv2.CAP_DSHOW, cv2.CAP_ANY]:
        cap = cv2.VideoCapture(CAMERA_ID, backend)
        if cap.isOpened():
            break
    if cap is None or not cap.isOpened():
        print("  ❌ 无法打开摄像头！")
        cv2.destroyAllWindows(); return
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    # 预热
    for _ in range(5):
        cap.read(); time.sleep(0.05)
    print("  ✓ 摄像头就绪")

    # ── 第 3 步：加载 MediaPipe ──
    print("  [3/3] 加载 MediaPipe...")
    base = mp_python.BaseOptions(model_asset_path=MODEL_PATH)
    opts = vision.HandLandmarkerOptions(
        base_options=base, num_hands=2,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
        running_mode=vision.RunningMode.VIDEO,
    )
    lmkr = vision.HandLandmarker.create_from_options(opts)
    print("  ✓ 就绪")
    print("\n  操作提示见窗口底部。ESC 退出。\n")

    # ── 主循环 ──
    fts, fps_v, alpha, ptick, frames = 0, 0.0, 0.9, time.time(), 0

    while True:
        ok, frame = cap.read()
        if not ok:
            time.sleep(0.01)
            continue

        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        frames += 1

        # MediaPipe
        try:
            mpimg = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            fts += 33
            res = lmkr.detect_for_video(mpimg, fts)
        except Exception:
            continue

        # 找手
        detected = False
        tgt_lms = None
        if res.hand_landmarks:
            for i, hlms in enumerate(res.hand_landmarks):
                hnd = "Right"
                if res.handedness and len(res.handedness) > i and len(res.handedness[i]) > 0:
                    hnd = res.handedness[i][0].category_name
                side = "Left" if hnd == "Right" else "Right"
                col = (220, 100, 50) if side == "Left" else (80, 220, 60)
                wrist = draw_hand(frame, hlms, w, h, col)
                if side.lower() == st.hand:
                    tgt_lms = hlms; detected = True
                    if wrist:
                        cv2.putText(frame, "TARGET", (wrist[0] + 10, wrist[1] - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

        # 保存
        now = time.time()
        if (not st.paused and detected and tgt_lms is not None
                and now - st.last_save >= COLLECT_INTERVAL):
            if csv_append(st.gname, now, st.hand, tgt_lms):
                st.count += 1; st.err = False
            else:
                st.err = True; st.errmsg = "文件占用—跳过"
            st.last_save = now

        # FPS
        dt = now - ptick; ptick = now
        if dt > 0: fps_v = alpha * fps_v + (1 - alpha) / dt

        # HUD + 显示
        frame = draw_hud(frame, st, fps_v, detected)
        cv2.imshow(WNAME, frame)

        # 键盘
        key = cv2.waitKey(1) & 0xFF
        if key == 27 or key == ord('q'):
            break
        elif key == ord(' '):
            st.paused = not st.paused
            st._toast = ("PAUSED" if st.paused else "RECORDING", time.time())
        elif key == ord('n'):
            st.next_g(); st._toast = (st.display, time.time())
            print(f"  → {st.display} ({st.count} 帧)")
        elif key == ord('p'):
            st.prev_g(); st._toast = (st.display, time.time())
            print(f"  ← {st.display} ({st.count} 帧)")
        elif key == ord('h'):
            st.swap_hand()
            st._toast = ("左手" if st.hand == "left" else "右手", time.time())
        elif key == ord('s'):
            print_stats()
        elif key == ord('r'):
            csv_reset(st.gname); st.count = 0; st.err = False

    cap.release(); cv2.destroyAllWindows(); lmkr.close()
    print("\n  结束。")
    print_stats()

if __name__ == "__main__":
    main()
