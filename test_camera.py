"""
摄像头诊断脚本 — 最小化测试
目的：排除摄像头驱动 / OpenCV GUI 的问题
"""

import cv2
import time
import numpy as np

print("1. 测试摄像头...")
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

if not cap.isOpened():
    print("   ERROR: 无法打开摄像头！尝试不带 CAP_DSHOW...")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("   ERROR: 仍然无法打开摄像头！")
        exit(1)

print("   OK")

print("2. 预热摄像头...")
for _ in range(10):
    ok, _ = cap.read()
    if ok:
        break
    time.sleep(0.1)

print("3. 创建窗口...")
cv2.namedWindow("Camera Test", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Camera Test", 960, 540)

print("4. 开始实时显示（10秒，按 ESC 退出）...")
t0 = time.time()
frames = 0

while True:
    ok, frame = cap.read()
    if not ok:
        print("   WARN: cap.read() 失败")
        time.sleep(0.05)
        continue

    frames += 1
    frame = cv2.flip(frame, 1)

    # 显示 FPS
    elapsed = time.time() - t0
    fps = frames / elapsed if elapsed > 0 else 0
    cv2.putText(frame, f"FPS: {fps:.1f}  Frame: {frames}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(frame, "Press ESC to exit",
                (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    cv2.imshow("Camera Test", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == 27:
        print("\n   ESC pressed, exiting...")
        break

    if frames == 1:
        print(f"   第一帧成功! shape={frame.shape}")

    if elapsed > 10:
        print("\n   10秒测试完成!")
        break

cap.release()
cv2.destroyAllWindows()
print(f"   {frames} frames in {elapsed:.1f}s = {fps:.1f} FPS")
print("   Camera + GUI 测试通过！")
