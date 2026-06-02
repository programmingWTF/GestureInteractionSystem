"""
GCN 手势数据集采集工具 (v4)

纯 GUI 驱动 — 不使用终端 input()，所有操作在摄像头窗口完成。

启动后在窗口内：
  点击窗口使其获得焦点，然后用键盘操作。

控制键：
  N   — 下一个手势
  P   — 上一个手势
  H   — 切换左手/右手
  SPACE — 暂停/继续采集
  S   — 终端打印统计
  R   — 清空当前手势 CSV
  ESC/Q — 退出
"""

import os
import sys

# 关闭 MediaPipe 遥测日志（clearcut uploader 报错不影响功能）
os.environ["GLOG_minloglevel"] = "2"
os.environ["GLOG_stderrthreshold"] = "2"

import cv2
import time
import csv
import numpy as np

import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

# ════════════════════════════
_MODULE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(_MODULE_DIR, "hand_landmarker.task")
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "DateSet")
os.makedirs(OUTPUT_DIR, exist_ok=True)
COLLECT_INTERVAL = 0.2   # 每秒 5 帧（之前 0.5s=2fps）

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

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (0, 17),
    (17, 18), (18, 19), (19, 20),
]

LANDMARK_NAMES = [
    "wrist","thumb_cmc","thumb_mcp","thumb_ip","thumb_tip",
    "index_mcp","index_pip","index_dip","index_tip",
    "middle_mcp","middle_pip","middle_dip","middle_tip",
    "ring_mcp","ring_pip","ring_dip","ring_tip",
    "pinky_mcp","pinky_pip","pinky_dip","pinky_tip",
]

CSV_COLUMNS = ["timestamp","handedness","gesture"]
for _i, _nm in enumerate(LANDMARK_NAMES):
    CSV_COLUMNS += [f"lm{_i}_{_nm}_x", f"lm{_i}_{_nm}_y", f"lm{_i}_{_nm}_z"]


# ════════════════════════════
# CSV
# ════════════════════════════
def _csv_dir(name): d=os.path.join(OUTPUT_DIR,name); os.makedirs(d,exist_ok=True); return d
def _csv_path(name): return os.path.join(_csv_dir(name), f"{name}.csv")

def csv_ensure(name):
    p=_csv_path(name)
    if not os.path.exists(p): _write_header(p); return p
    try:
        with open(p,"r",encoding="utf-8") as f:
            if f.readline().strip()!=",".join(CSV_COLUMNS):
                _write_header(p)
    except: _write_header(p)
    return p

def _write_header(p):
    try:
        with open(p,"w",newline="",encoding="utf-8") as f:
            csv.writer(f).writerow(CSV_COLUMNS)
    except PermissionError: pass

def csv_count(name):
    p=_csv_path(name)
    if not os.path.exists(p): return 0
    try:
        with open(p,"r",encoding="utf-8") as f:
            lines=f.readlines()
        if len(lines)<=1: return 0
        ec=len(CSV_COLUMNS)
        return sum(1 for L in lines[1:] if L.strip() and L.count(",")>=ec-1)
    except: return 0

def csv_append(name,ts,hand,lms):
    try:
        p=csv_ensure(name)
        row=[f"{ts:.3f}",hand,name]
        for lm in lms: row+=[f"{lm.x:.6f}",f"{lm.y:.6f}",f"{lm.z:.6f}"]
        with open(p,"a",newline="",encoding="utf-8") as f:
            csv.writer(f).writerow(row)
        return True
    except (PermissionError,OSError): return False

def csv_reset(name):
    try: os.remove(_csv_path(name))
    except FileNotFoundError: pass
    except PermissionError: print(f"  CANNOT reset {name}: file locked")
    csv_ensure(name)


# ════════════════════════════
# 状态
# ════════════════════════════
class State:
    def __init__(self):
        self.hand="right"; self.gid=0; self.gname="open_palm"
        self.paused=False; self.last_save=0.0; self.count=0
        self.err=False; self.errmsg=""
    @property
    def display(self):
        g=GESTURE_BY_ID[self.gid]; return f"[{g[0]:02d}] {g[2]}"
    def swap(self): self.hand="left" if self.hand=="right" else "right"
    def set_g(self,gid):
        if gid in GESTURE_BY_ID:
            self.gid=gid; self.gname=GESTURE_BY_ID[gid][1]
            csv_ensure(self.gname); self.count=csv_count(self.gname)
            self.err=False; self.errmsg=""
    def next_g(self): self.set_g((self.gid+1)%len(GESTURES))
    def prev_g(self): self.set_g((self.gid-1)%len(GESTURES))


# ════════════════════════════
# 绘制
# ════════════════════════════
def draw_hand(img,lms,iw,ih,col):
    pts={}
    for i,lm in enumerate(lms):
        x,y=int(lm.x*iw),int(lm.y*ih); pts[i]=(x,y)
        cv2.circle(img,(x,y),4,col,-1)
        cv2.circle(img,(x,y),6,(255,255,255),1)
    for a,b in HAND_CONNECTIONS:
        if a in pts and b in pts:
            cv2.line(img,pts[a],pts[b],tuple(min(255,c+40) for c in col),2)
    return pts.get(0)

def draw_hud(fr,st,fps,detected):
    h,w=fr.shape[:2]
    ov=fr.copy(); cv2.rectangle(ov,(0,0),(w,120),(0,0,0),-1)
    fr[:]=cv2.addWeighted(fr,0.85,ov,0.15,0)
    cv2.putText(fr,"GCN Data Collector (v4)",(12,26),
                cv2.FONT_HERSHEY_SIMPLEX,0.6,(100,220,255),2)
    if st.paused:
        cv2.putText(fr,"PAUSED",(w-140,26),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,165,255),2)
        cv2.circle(fr,(w-40,22),6,(0,165,255),-1)
    else:
        cv2.putText(fr,"REC",(w-140,26),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,255,100),2)
        cv2.circle(fr,(w-40,22),6+int(3*(time.time()%1>0.5)),(0,255,0),-1)
    ic="L" if st.hand=="left" else "R"
    hc=(220,120,40) if st.hand=="left" else (80,220,80)
    cv2.putText(fr,f"Hand: {ic}  |  {st.display}",(12,58),
                cv2.FONT_HERSHEY_SIMPLEX,0.5,hc,2)
    cv2.putText(fr,"DETECTED" if detected else "NO HAND",(12,86),
                cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,0) if detected else (0,0,255),2)
    cv2.putText(fr,f"Saved: {st.count}  |  FPS: {fps:.0f}",(12,114),
                cv2.FONT_HERSHEY_SIMPLEX,0.45,(200,200,200),1)
    if st.err:
        cv2.putText(fr,f"ERR: {st.errmsg}",(12,136),
                    cv2.FONT_HERSHEY_SIMPLEX,0.4,(0,0,255),2)
    # toast
    if hasattr(st,"_toast") and st._toast:
        txt,t0=st._toast
        if time.time()-t0<1.5:
            sz=cv2.getTextSize(txt,cv2.FONT_HERSHEY_SIMPLEX,0.7,2)[0]
            cv2.putText(fr,txt,((w-sz[0])//2,h//2),
                        cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,255),2)
        else: st._toast=None
    # bottom bar
    cv2.rectangle(fr,(0,h-28),(w,h),(30,30,30),-1)
    cv2.putText(fr,"N:下个  P:上个  H:换手  SPACE:暂停  S:统计  R:清空  ESC:退出",
                (10,h-8),cv2.FONT_HERSHEY_SIMPLEX,0.38,(180,180,180),1)
    return fr


# ════════════════════════════
# 主函数
# ════════════════════════════
def main():
    print("="*50)
    print("  GCN Data Collector v4")
    print("  All controls via OpenCV window!")
    print("="*50)

    if not os.path.exists(MODEL_PATH):
        print(f"\nERROR: model not found: {MODEL_PATH}"); return

    st=State()
    WNAME="GCN Data Collector"

    # ── Step 1: 创建窗口（第一步，确保用户立刻看到） ──
    print("\n[1/4] Creating window...")
    cv2.namedWindow(WNAME, cv2.WINDOW_NORMAL|cv2.WINDOW_GUI_NORMAL)
    cv2.resizeWindow(WNAME, 960, 540)
    # 立即显示启动画面
    splash = np.zeros((540, 960, 3), dtype=np.uint8)
    cv2.putText(splash, "GCN Data Collector v4", (220, 200),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (100, 220, 255), 2)
    cv2.putText(splash, "Loading camera...", (340, 250),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
    cv2.putText(splash, "Default: Right hand, Open Palm", (300, 300),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
    cv2.putText(splash, "Use N/P/H/SPACE to control after start", (270, 330),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150, 150, 150), 1)
    cv2.imshow(WNAME, splash)
    cv2.waitKey(1)
    print("  Window created")

    # ── Step 2: 摄像头 ──
    print("[2/4] Opening camera...")
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(0)  # fallback
    if not cap.isOpened():
        cv2.putText(splash, "ERROR: Cannot open camera!", (280, 400),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        cv2.imshow(WNAME, splash); cv2.waitKey(3000)
        cv2.destroyAllWindows(); return
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    for _ in range(5): cap.read(); time.sleep(0.05)
    print("  Camera ready")

    # 更新启动画面
    cv2.putText(splash, "Loading MediaPipe model...", (300, 400),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
    cv2.imshow(WNAME, splash); cv2.waitKey(1)

    # ── Step 3: MediaPipe ──
    print("[3/4] Loading MediaPipe...")
    base = mp_python.BaseOptions(model_asset_path=MODEL_PATH)
    opts = vision.HandLandmarkerOptions(
        base_options=base, num_hands=2,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
        running_mode=vision.RunningMode.VIDEO,
    )
    lmkr = vision.HandLandmarker.create_from_options(opts)

    csv_ensure(st.gname); st.count = csv_count(st.gname)
    print(f"  Ready!  Starting gesture: {st.display}  Hand: Right")
    print(f"  Existing frames: {st.count}")
    print(f"\n  [4/4] Running... Click the window and use keys!\n")

    # ── 主循环 ──
    fts, fps_v, alpha, ptick = 0, 0.0, 0.9, time.time()

    while True:
        # ── 必须：每帧都 waitKey 驱动 GUI ──
        key = cv2.waitKey(1) & 0xFF
        if key == 27 or key == ord('q'):
            break

        ok, frame = cap.read()
        if not ok:
            time.sleep(0.01)
            continue

        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]

        # MediaPipe
        try:
            mpimg = mp.Image(image_format=mp.ImageFormat.SRGB,
                             data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            fts += 33
            res = lmkr.detect_for_video(mpimg, fts)
        except Exception:
            continue

        # 找手
        detected = False; tgt = None
        if res.hand_landmarks:
            for i, hlms in enumerate(res.hand_landmarks):
                hnd = "Right"
                if res.handedness and len(res.handedness)>i and len(res.handedness[i])>0:
                    hnd = res.handedness[i][0].category_name
                side = "Left" if hnd == "Right" else "Right"
                col = (220, 100, 50) if side == "Left" else (80, 220, 60)
                wrist = draw_hand(frame, hlms, w, h, col)
                if side.lower() == st.hand:
                    tgt = hlms; detected = True
                    if wrist:
                        cv2.putText(frame, "TARGET", (wrist[0]+10, wrist[1]-10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

        # 保存
        now = time.time()
        if (not st.paused and detected and tgt is not None
                and now - st.last_save >= COLLECT_INTERVAL):
            if csv_append(st.gname, now, st.hand, tgt):
                st.count += 1; st.err = False
            else:
                st.err = True; st.errmsg = "file locked"
            st.last_save = now

        # FPS
        dt = now - ptick; ptick = now
        if dt > 0: fps_v = alpha * fps_v + (1 - alpha) / dt

        # 绘制 & 显示
        frame = draw_hud(frame, st, fps_v, detected)
        cv2.imshow(WNAME, frame)

        # 处理键盘
        if key == ord(' '):
            st.paused = not st.paused
            st._toast = ("PAUSED" if st.paused else "REC", time.time())
        elif key == ord('n'):
            st.next_g(); st._toast = (st.display, time.time())
            print(f"  -> {st.display}  ({st.count} saved)")
        elif key == ord('p'):
            st.prev_g(); st._toast = (st.display, time.time())
            print(f"  <- {st.display}  ({st.count} saved)")
        elif key == ord('h'):
            st.swap(); st._toast = ("L Hand" if st.hand=="left" else "R Hand", time.time())
            lr = "左手" if st.hand=="left" else "右手"
            print(f"  Hand: {lr}")
        elif key == ord('s'):
            print("\n  === Stats ===")
            for gid,nm,dp in GESTURES:
                n=csv_count(nm); print(f"  [{gid}] {dp:<10s} {n:>5d}")
            print()
        elif key == ord('r'):
            csv_reset(st.gname); st.count = 0; st.err = False
            print(f"  Cleared: {st.gname}")

    cap.release(); cv2.destroyAllWindows(); lmkr.close()
    print("\nDone.")


if __name__ == "__main__":
    main()
