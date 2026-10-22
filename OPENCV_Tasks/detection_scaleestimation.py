import cv2
import numpy as np
import math
from collections import deque
cap = cv2.VideoCapture(0)
scale = 10
tracker = None
trail = deque(maxlen=100)
def new_tracker():
    if hasattr(cv2, "TrackerCSRT_create"): return cv2.TrackerCSRT_create()
    return cv2.legacy.TrackerCSRT_create()
def click(event, x, y, flags, data):
    global tracker, trail
    if event == cv2.EVENT_LBUTTONDOWN:
        contours, _ = cv2.findContours(data[1], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in contours:
            if cv2.contourArea(c) > 100 and cv2.pointPolygonTest(c, (x, y), False) >= 0:
                tracker = new_tracker()
                tracker.init(data[0], cv2.boundingRect(c))
                trail.clear()
                break
cv2.namedWindow("Tracker")
data = [None, None]
cv2.setMouseCallback("Tracker", click, data)
while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    blue = cv2.inRange(hsv, np.array([90, 70, 40]), np.array([135, 255, 255]))
    yellow = cv2.inRange(hsv, np.array([20, 80, 80]), np.array([45, 255, 255]))
    blue = cv2.morphologyEx(blue, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    data[:] = [frame.copy(), blue.copy()]
    contours, _ = cv2.findContours(yellow, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    balls = []
    for c in contours:
        area = cv2.contourArea(c)
        p = cv2.arcLength(c, True)
        if area < 80 or area > 15000 or p == 0: continue
        circularity = 4 * math.pi * area / (p * p)
        x, y, w, h = cv2.boundingRect(c)
        if circularity < 0.55 or not 0.65 <= w / h <= 1.35: continue
        (cx, cy), r = cv2.minEnclosingCircle(c)
        if r < 3: continue
        cx, cy = int(cx), int(cy)
        balls.append((cx, cy, r))
        cv2.circle(frame, (cx, cy), int(r), (0, 255, 255), 2)
        cv2.putText(frame, f"Ball: ({cx},{cy}) R={r:.1f}px A={math.pi*r*r/scale**2:.2f}cm2", (cx+10, cy),  cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 2)
    contours, _ = cv2.findContours(blue, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if tracker is None:
        for c in contours:
            if cv2.contourArea(c) > 100: cv2.rectangle(frame, cv2.boundingRect(c)[:2], (cv2.boundingRect(c)[0]+cv2.boundingRect(c)[2], cv2.boundingRect(c)[1]+cv2.boundingRect(c)[3]), (255, 150, 0), 1)
    else:
        ok, box = tracker.update(frame)
        if ok:
            x, y, w, h = map(int, box)
            x1, y1 = max(0, x), max(0, y)
            roi = blue[y1:min(y+h, blue.shape[0]), x1:min(x+w, blue.shape[1])]
            cs, _ = cv2.findContours(roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cs = [c for c in cs if cv2.contourArea(c) > 100]
            if cs:
                c = max(cs, key=cv2.contourArea)
                c[:, 0, 0] += x1
                c[:, 0, 1] += y1
                bx, by, bw, bh = cv2.boundingRect(c)
                m = cv2.moments(c)
                if m["m00"]:
                    center = (int(m["m10"]/m["m00"]), int(m["m01"]/m["m00"]))
                    trail.append(center)
                    cv2.drawContours(frame, [c], -1, (0, 255, 0), 2)
                    cv2.rectangle(frame, (bx, by), (bx+bw, by+bh), (0, 255, 0), 2)
                    for i in range(1, len(trail)): cv2.line(frame, trail[i-1], trail[i], (255, 0, 255), 2)
                    cv2.circle(frame, center, 4, (255, 0, 255), -1)
                    cv2.putText(frame, f"Blue: {center} Area={cv2.contourArea(c)/scale**2:.2f}cm2", (bx, max(20, by-25)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    cv2.putText(frame, f"Box=({bx},{by},{bw},{bh})", (bx, by+bh+18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)
                    if balls:
                        tx, ty, r = min(balls, key=lambda b: (b[0]-center[0])**2+(b[1]-center[1])**2)
                        dx, dy = center[0]-tx, center[1]-ty
                        cv2.line(frame, (tx, ty), center, (255, 0, 0), 2)
                        cv2.putText(frame, f"Blue vs ball: dx={dx}px dy={dy}px D={math.hypot(dx,dy):.1f}px", (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    cv2.imshow("Tracker", frame)
    key = cv2.waitKey(1) & 0xFF
    if key in (ord('q'), 27): break
    if key == ord('r'): tracker = None; trail.clear()
cap.release()
cv2.destroyAllWindows()