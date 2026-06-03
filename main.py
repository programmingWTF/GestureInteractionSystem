"""
GestureInteractionSystem v3

Interaction:
  L OK        → start writing
  L ThumbsUp  → submit to CNN
  L OpenPalm  → clear canvas (keep writing)
  R Pinch     → pen down
  R OpenPalm  → pen up
  ESC         → quit
"""

import cv2
import time
import os
import sys
import math
import tempfile
import numpy as np
import torch

import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

_MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(_MODULE_DIR, "hand_landmarker.task")

_GCN_MODEL_PATH = os.path.join(_MODULE_DIR, "GCN", "best_model.pth")
_GCN_AVAILABLE = os.path.exists(_GCN_MODEL_PATH)
_GCN_USE_CUDA = True   # ← 改成 False 即强制 CPU 推理
if _GCN_AVAILABLE:
    sys.path.insert(0, os.path.join(_MODULE_DIR, "GCN"))
    from predictor import GCNPredictor

# CNN model (loaded once at startup)
sys.path.insert(0, os.path.join(_MODULE_DIR, "CNN"))
from train import DigitCNN
CNN_MODEL_PATH = os.path.join(_MODULE_DIR, "CNN", "qmnist_digit_model.pth")
cnn_model = None   # set in main()

# gesture name mapping (CN→EN for display)
G_MAP = {
    "手掌张开": "Open", "拳头": "Fist", "食指指出": "Idx",
    "胜利V": "V", "OK": "OK", "点赞": "Like",
    "三指": "Three", "捏合": "Pinch", "四指": "Four", "拇指向下": "Down",
    None: "--",
}

HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),
    (5,9),(9,10),(10,11),(11,12),(9,13),(13,14),(14,15),(15,16),
    (13,17),(0,17),(17,18),(18,19),(19,20),
]


def draw_hand(img, lms, iw, ih, color):
    pts = {}
    for i, lm in enumerate(lms):
        x, y = int(lm.x * iw), int(lm.y * ih)
        pts[i] = (x, y)
        cv2.circle(img, (x, y), 4, color, -1)
        cv2.circle(img, (x, y), 6, (255,255,255), 1)
    for a, b in HAND_CONNECTIONS:
        if a in pts and b in pts:
            cv2.line(img, pts[a], pts[b],
                     tuple(min(255,c+40) for c in color), 2)
    return pts


def predict_cnn(strokes, frame_w, frame_h):
    """Returns (label, preview, top3_list) or (None, None, [])"""
    all_pts = [p for s in strokes for p in s if s]
    if len(all_pts) < 5:
        return None, None, []

    xs = [p[0] for p in all_pts]; ys = [p[1] for p in all_pts]
    x1, x2 = min(xs), max(xs); y1, y2 = min(ys), max(ys)
    bw, bh = x2 - x1, y2 - y1
    if bw < 5 or bh < 5:
        return None, None

    # square crop with 15% padding
    pad = 0.15
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    side = max(bw, bh) * (1 + pad)
    x1 = max(0, int(cx - side / 2))
    y1 = max(0, int(cy - side / 2))
    x2 = min(frame_w, int(cx + side / 2))
    y2 = min(frame_h, int(cy + side / 2))
    s = min(x2 - x1, y2 - y1)
    x2, y2 = x1 + s, y1 + s

    size = 280
    from PIL import Image, ImageDraw, ImageOps
    img = Image.new("L", (size, size), 255)
    draw = ImageDraw.Draw(img)
    for stroke in strokes:
        if len(stroke) < 2:
            continue
        pts = [(int((p[0] - x1) / s * size),
                int((p[1] - y1) / s * size)) for p in stroke]
        draw.line(pts, fill=0, width=18)

    tmp = os.path.join(tempfile.gettempdir(), "_cnn_traj.png")
    img.save(tmp)

    # preview as BGR numpy for display
    preview = cv2.cvtColor(np.array(img.convert("RGB")), cv2.COLOR_RGB2BGR)

    # CNN inference — preprocessing matches training val_transform
    if cnn_model is None:
        return None, preview, []

    try:
        pil_img = Image.open(tmp).convert("L")
        pil_img = ImageOps.invert(pil_img)
        pil_img = pil_img.resize((32, 32), Image.LANCZOS)

        arr = np.array(pil_img, dtype=np.float32) / 255.0
        arr = np.stack([arr, arr, arr], axis=0)
        arr = (arr - 0.5) / 0.5
        x = torch.from_numpy(arr).unsqueeze(0).to(next(cnn_model.parameters()).device)

        with torch.no_grad():
            logits = cnn_model(x)
            probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()
        best = int(probs.argmax())
        top3 = sorted(enumerate(probs), key=lambda x: -x[1])[:3]
        return str(best), preview, [(d, float(p)) for d, p in top3]
    except Exception:
        return None, preview, []


# ---------------------------------------------------------------------------
# 1 Euro Filter: adaptive low-pass for landmark smoothing
# ---------------------------------------------------------------------------

class OneEuroFilter:
    """1€ Filter: 低延迟自适应平滑。

    min_cutoff: 最低截止频率 (Hz)，越小越平滑。默认 1.2
    beta:       速度系数，控制自适应强度。默认 0.007
    """
    def __init__(self, min_cutoff=6.5, beta=0.002):
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.x_prev = None
        self.dx_prev = None
        self.t_prev = None

    def reset(self):
        self.x_prev = None; self.dx_prev = None; self.t_prev = None

    def __call__(self, x, t):
        """x: float, t: seconds"""
        if self.x_prev is None:
            self.x_prev = x; self.dx_prev = 0.0; self.t_prev = t
            return x
        dt = t - self.t_prev
        if dt <= 0: return self.x_prev
        # derivative
        dx = (x - self.x_prev) / dt
        dx_smooth = 0.4 * self.dx_prev + 0.6 * dx  if self.dx_prev is not None else dx
        # adaptive cutoff
        cutoff = self.min_cutoff + self.beta * abs(dx_smooth)
        tau = 1.0 / (2.0 * 3.1415926535 * cutoff)
        alpha = 1.0 / (1.0 + tau / dt)
        x_smooth = self.x_prev + alpha * (x - self.x_prev)
        self.x_prev = x_smooth; self.dx_prev = dx_smooth; self.t_prev = t
        return x_smooth


def smooth_landmarks(filters, lms, t):
    """对 21 个关键点的 (x,y,z) 分别做 1€ 滤波"""
    for i, lm in enumerate(lms):
        lm.x = filters[i * 3 + 0](lm.x, t)
        lm.y = filters[i * 3 + 1](lm.y, t)
        lm.z = filters[i * 3 + 2](lm.z, t)


def pinch_dist(lms):
    return math.hypot(lms[4].x - lms[8].x, lms[4].y - lms[8].y)

def pinch_mid(lms):
    return (lms[4].x + lms[8].x) / 2, (lms[4].y + lms[8].y) / 2

def gcn_pred(predictor, lms, handedness, hand_id):
    if predictor is None:
        return None, 0
    try:
        name, conf, _ = predictor.predict(lms, handedness, hand_id=hand_id)
        return name, conf
    except:
        return None, 0


# ---------------------------------------------------------------------------
# Heads-up display
# ---------------------------------------------------------------------------

def draw_hud(frame, mode, left_g, left_gc, right_g, right_gc, pen, strokes,
             fps, result_label, preview_img, cnn_top3, show_guide):
    h, w = frame.shape[:2]

    # bottom bar
    cv2.rectangle(frame, (0, h - 32), (w, h), (18, 18, 20), -1)

    if mode == "write":
        md, mc = "WRITING", (0, 255, 200)
    elif result_label:
        md, mc = f"CNN: {result_label}", (255, 220, 0)
    else:
        md, mc = "IDLE", (120, 120, 120)
    cv2.putText(frame, md, (12, h - 9), cv2.FONT_HERSHEY_SIMPLEX,
                0.55, mc, 2)

    pn = "PEN DOWN" if pen else "PEN UP"
    pc = (0, 255, 100) if pen else (80, 80, 80)
    cv2.putText(frame, pn, (w - 320, h - 9), cv2.FONT_HERSHEY_SIMPLEX,
                0.45, pc, 1)

    cv2.putText(frame, f"S:{len(strokes)} FPS:{fps:.0f}",
                (w - 160, h - 9), cv2.FONT_HERSHEY_SIMPLEX,
                0.4, (180, 180, 180), 1)

    # hand labels
    # hand label colors: L=blue, R=green
    # confidence colors: green>0.6, yellow>0.3, orange<0.3
    def conf_color(c):
        return (0,220,0) if c>0.6 else (0,220,255) if c>0.3 else (0,140,255)
    lg = G_MAP.get(left_g, '--'); lc = conf_color(left_gc)
    rg = G_MAP.get(right_g, '--'); rc = conf_color(right_gc)
    cv2.putText(frame, f"L: {lg}", (12, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 140, 50), 1)
    cv2.putText(frame, f"{left_gc:.0%}", (110, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, lc, 1)
    cv2.putText(frame, f"R: {rg}", (12, 48),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (80, 220, 80), 1)
    cv2.putText(frame, f"{right_gc:.0%}", (110, 48),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, rc, 1)

    # interaction guide
    if show_guide:
        ov = frame.copy()
        cv2.rectangle(ov, (w - 260, 0), (w, 140), (0, 0, 0), -1)
        frame[:] = cv2.addWeighted(frame, 0.82, ov, 0.18, 0)
        lines = [
            "L OK        -> start",
            "L ThumbsUp  -> submit",
            "L OpenPalm  -> clear",
            "R Pinch     -> pen down",
            "R OpenPalm  -> pen up",
        ]
        for i, ln in enumerate(lines):
            cv2.putText(frame, ln, (w - 252, 16 + i * 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

    # CNN preview + top-3 (bottom-left)
    if preview_img is not None:
        pv = cv2.resize(preview_img, (140, 140))
        ph, pw = pv.shape[:2]
        px, py = 10, h - 180
        frame[py:py+ph, px:px+pw] = pv
        cv2.rectangle(frame, (px, py), (px+pw, py+ph), (255, 220, 0), 1)
    # CNN result overlay (centered, prominent)
    if result_label:
        ov = frame.copy()
        # semi-transparent dark bar across center
        bar_h = 80
        cy = h // 2
        cv2.rectangle(ov, (0, cy - bar_h // 2), (w, cy + bar_h // 2), (0, 0, 0), -1)
        frame[:] = cv2.addWeighted(frame, 0.55, ov, 0.45, 0)
        # recognized digit — large
        txt = f"CNN: {result_label}"
        tsz = cv2.getTextSize(txt, cv2.FONT_HERSHEY_DUPLEX, 1.8, 3)[0]
        cv2.putText(frame, txt, ((w - tsz[0]) // 2, cy + 10),
                    cv2.FONT_HERSHEY_DUPLEX, 1.8, (255, 220, 0), 3)
        # top-3 below
        if cnn_top3:
            top_txt = "  |  ".join(f"{d}:{p:.0%}" for d, p in cnn_top3[:3])
            tsz2 = cv2.getTextSize(top_txt, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)[0]
            cv2.putText(frame, top_txt, ((w - tsz2[0]) // 2, cy + 32),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)

    # CNN preview (bottom-left)
    if preview_img is not None:
        pv = cv2.resize(preview_img, (140, 140))
        ph, pw = pv.shape[:2]
        px, py = 10, h - 180
        frame[py:py+ph, px:px+pw] = pv
        cv2.rectangle(frame, (px, py), (px+pw, py+ph), (255, 220, 0), 1)


def main():
    print("=" * 50)
    print("  GestureInteractionSystem v3")
    print("=" * 50)

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    if not cap.isOpened():
        print("ERROR: camera"); return
    fw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    base = mp_python.BaseOptions(model_asset_path=MODEL_PATH)
    opts = vision.HandLandmarkerOptions(
        base_options=base, num_hands=2,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
        running_mode=vision.RunningMode.VIDEO,
    )
    lmkr = vision.HandLandmarker.create_from_options(opts)

    # CNN model (load once, reuse)
    global cnn_model
    if os.path.exists(CNN_MODEL_PATH):
        try:
            cnn_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            cnn_model = DigitCNN()
            cnn_model.load_state_dict(torch.load(CNN_MODEL_PATH, map_location="cpu"))
            cnn_model.to(cnn_device)
            cnn_model.eval()
            print("  CNN model loaded")
        except Exception as e:
            print(f"  CNN failed: {e}")

    # GCN model
    gcn = None
    if _GCN_AVAILABLE:
        try:
            device = "cuda" if (_GCN_USE_CUDA and torch.cuda.is_available()) else "cpu"
            gcn = GCNPredictor(_GCN_MODEL_PATH, device=device)
            print("  GCN loaded")
        except Exception as e:
            print(f"  GCN failed: {e}")

    print("\n  L OK->write | ThumbsUp->submit | OpenPalm->clear")
    print("  R Pinch->pen | OpenPalm->lift | ESC->quit\n")

    mode = "idle"; strokes = []; cur = []; pen = False
    result_label = None; preview_img = None; cnn_top3 = []
    left_last = None; left_lock = 0
    fts = 0; fps_v = 0.0; ptick = time.time()

    # 1 Euro filters: separate state per hand
    filter_L = [OneEuroFilter() for _ in range(63)]
    filter_R = [OneEuroFilter() for _ in range(63)]

    cv2.namedWindow("Gesture v3", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Gesture v3", 1280, 720)

    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            break

        ok, frame = cap.read()
        if not ok:
            time.sleep(0.01); continue

        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]

        mpimg = mp.Image(image_format=mp.ImageFormat.SRGB,
                         data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        fts += 33
        res = lmkr.detect_for_video(mpimg, fts)

        left_lms = None; right_lms = None
        left_g = None; right_g = None
        left_gc = 0; right_gc = 0

        if res.hand_landmarks:
            for i, hlms in enumerate(res.hand_landmarks):
                hnd = "Right"
                if res.handedness and len(res.handedness) > i and len(res.handedness[i]) > 0:
                    hnd = res.handedness[i][0].category_name
                actual = "Left" if hnd == "Right" else "Right"

                # 1 Euro filter (per-hand)
                flt = filter_L if actual == "Left" else filter_R
                smooth_landmarks(flt, hlms, time.time())

                col = (220, 100, 50) if actual == "Left" else (80, 220, 60)
                draw_hand(frame, hlms, w, h, col)

                g, gc = gcn_pred(gcn, hlms, actual, actual.lower())
                if actual == "Left":
                    left_lms = hlms; left_g = g; left_gc = gc
                else:
                    right_lms = hlms; right_g = g; right_gc = gc

        # left hand control (debounced)
        if left_lock > 0:
            left_lock -= 1
        if left_g != left_last:
            left_last = left_g; left_lock = 8
        if left_lock == 1:
            if left_g == "手掌张开":
                strokes = []; cur = []; pen = False
                result_label = None; preview_img = None; cnn_top3 = []
                print("  clear")
            elif left_g == "OK" and mode == "idle":
                mode = "write"; strokes = []; cur = []
                pen = False; result_label = None; preview_img = None; cnn_top3 = []
                print("  start")
            elif left_g == "点赞" and mode == "write":
                if cur: strokes.append(cur); cur = []
                label, pv, top3 = predict_cnn(strokes, fw, fh)
                result_label = label or f"{len(strokes)} strokes"
                preview_img = pv; cnn_top3 = top3
                print(f"  submit: {result_label}")
                mode = "idle"; pen = False

        # right hand writing
        if mode == "write" and right_lms:
            d = pinch_dist(right_lms)
            pinching = d < 0.04
            if pinching and not pen:
                pen = True; cur = []
            elif not pinching and pen:
                pen = False
                if len(cur) > 2: strokes.append(cur)
                cur = []
            if pen:
                mx, my = pinch_mid(right_lms)
                px, py = int(mx * w), int(my * h)
                if not cur or math.hypot(px - cur[-1][0], py - cur[-1][1]) > 2:
                    cur.append((px, py))
                cv2.circle(frame, (px, py), 10, (0, 255, 255), 2)

        # stroke canvas
        cv2_canvas = np.zeros_like(frame)
        for st in strokes:
            if len(st) >= 2:
                for i in range(len(st) - 1):
                    cv2.line(cv2_canvas, st[i], st[i+1], (0, 255, 255), 3, cv2.LINE_AA)
        if pen and cur and len(cur) >= 2:
            for i in range(len(cur) - 1):
                cv2.line(cv2_canvas, cur[i], cur[i+1], (0, 255, 255), 3, cv2.LINE_AA)
        frame = cv2.addWeighted(frame, 0.7, cv2_canvas, 0.4, 0)

        # FPS
        now = time.time()
        dt = now - ptick; ptick = now
        if dt > 0: fps_v = 0.85 * fps_v + 0.15 / dt

        # fade out CNN result
        if result_label and mode != "idle":
            result_label = None; preview_img = None; cnn_top3 = []

        draw_hud(frame, mode, left_g, left_gc, right_g, right_gc,
                 pen, strokes, fps_v, result_label, preview_img,
                 cnn_top3, mode == "idle" and not result_label)

        cv2.imshow("Gesture v3", frame)

    cap.release(); cv2.destroyAllWindows(); lmkr.close()
    print("Done.")


if __name__ == "__main__":
    main()
