import cv2
import numpy as np
import argparse
from math import hypot
from collections import deque
BLUE_LOWER = np.array([90, 70, 40], dtype=np.uint8)
BLUE_UPPER = np.array([140, 255, 255], dtype=np.uint8)
BALL_LOWER = np.array([20, 70, 70], dtype=np.uint8)
BALL_UPPER = np.array([45, 255, 255], dtype=np.uint8)
MIN_BLUE_AREA = 300
MIN_BALL_AREA = 150
MAX_BALL_AREA = 30000
MAX_MATCH_DISTANCE = 60
MAX_MISSED_FRAMES = 10
BALL_MAX_MISSED_FRAMES = 30
BALL_MAX_JUMP = 150
BALL_LOOSE_CIRCULARITY = 0.3
BALL_TRAIL_COLOR = (0, 165, 255)
TRAIL_LENGTH = 100
BLUE_COLOR = (255, 100, 0)
BALL_COLOR = (0, 255, 0)
RELATION_COLOR = (0, 255, 255)
TEXT_COLOR = (255, 255, 255)
def create_mask(hsv, lower, upper):
    mask = cv2.inRange(hsv, lower, upper)
 
    kernel = np.ones((5, 5), np.uint8)
 
    mask = cv2.morphologyEx(
        mask, cv2.MORPH_OPEN, kernel
    )
    mask = cv2.morphologyEx(
        mask, cv2.MORPH_CLOSE, kernel
    )
    return mask
def find_blue_objects(mask):
    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )
    detections = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < MIN_BLUE_AREA:
           continue
        x, y, w, h = cv2.boundingRect(contour)
        moments = cv2.moments(contour)
        if moments["m00"] == 0:
            continue
        cx = int(moments["m10"] / moments["m00"])
        cy = int(moments["m01"] / moments["m00"])
        detections.append({
            "centroid": (cx, cy),
            "bbox": (x, y, w, h),
            "contour": contour,
            "area": area
        })
    return detections
def find_tennis_ball(mask, last_center=None):
    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )
    candidates = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if not MIN_BALL_AREA <= area <= MAX_BALL_AREA:
            continue
        perimeter = cv2.arcLength(contour, True)
        if perimeter == 0:
            continue
        circularity = (
            4 * np.pi * area / (perimeter ** 2)
        )
        (cx, cy), radius = cv2.minEnclosingCircle(contour)
        if radius < 5:
            continue
        candidates.append({
            "center": (int(cx), int(cy)),
            "radius": radius,
            "area": area,
            "contour": contour,
            "circularity": circularity
        })
    if last_center is not None:
        near = [
            item for item in candidates
            if item["circularity"] >= BALL_LOOSE_CIRCULARITY
            and hypot(
                item["center"][0] - last_center[0],
                item["center"][1] - last_center[1]
            ) <= BALL_MAX_JUMP
        ]
        if near:
            return min(
                near,
                key=lambda item: hypot(
                    item["center"][0] - last_center[0],
                    item["center"][1] - last_center[1]
                )
            )
    strict = [
        item for item in candidates
        if item["circularity"] >= 0.55
    ]
    if not strict:
        return None
    return max(
        strict,
        key=lambda item: item["circularity"]
    )
class ObjectTracker:
    def __init__(self):
        self.next_id = 1
        self.tracks = {}
    def update(self, detections):
        for track in self.tracks.values():
            track["matched"] = False
            track["missed"] += 1
        for detection in detections:
            cx, cy = detection["centroid"]
            best_id = None
            best_distance = MAX_MATCH_DISTANCE
            for track_id, track in self.tracks.items():
                if track["matched"]:
                    continue
                tx, ty = track["centroid"]
                distance = hypot(cx - tx, cy - ty)
                if distance < best_distance:
                    best_distance = distance
                    best_id = track_id
            if best_id is None:
                best_id = self.next_id
                self.next_id += 1
                self.tracks[best_id] = {
                    **detection,
                    "previous_centroid": (cx, cy),
                    "trail": deque([(cx, cy)], maxlen=TRAIL_LENGTH),
                    "missed": 0,
                    "matched": True
                }
            else:
                track = self.tracks[best_id]
                track["previous_centroid"] = track["centroid"]
                track.update(detection)
                track["trail"].append((cx, cy))
                track["missed"] = 0
                track["matched"] = True
        expired = [
            track_id
            for track_id, track in self.tracks.items()
            if track["missed"] > MAX_MISSED_FRAMES
        ]
        for track_id in expired:
            del self.tracks[track_id]
 
        return self.tracks
def draw_trail(frame, trail, color, max_thickness=4):
    points = list(trail)
    count = len(points)
    if count < 2:
        return
    for i in range(1, count):
        fade = i / (count - 1)
        seg_color = tuple(
            int(c * (0.4 + 0.6 * fade)) for c in color
        )
        thickness = max(
            1, int(round(1 + fade * (max_thickness - 1)))
        )
        cv2.line(
            frame,
            points[i - 1],
            points[i],
            seg_color,
            thickness,
            cv2.LINE_AA
        )
def draw_label(frame, text, position, color=TEXT_COLOR,
               scale=0.48):
    x, y = position
    x = max(0, min(x, frame.shape[1] - 1))
    y = max(18, min(y, frame.shape[0] - 1))
    cv2.putText(
        frame,
        text,
        (x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        1,
        cv2.LINE_AA
    )
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--camera",
        type=int,
        default=0,
        help="Camera index, default 0"
    )
    args = parser.parse_args()
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print("Error: Cannot open camera.")
        return
    tracker = ObjectTracker()
    ball_trail = deque(maxlen=TRAIL_LENGTH)
    ball_missed = 0
    mode = "a"
    show_mask = False
    print("Controls:")
    print("b = Blue objects")
    print("t = Tennis ball")
    print("a = Both")
    print("c = Clear trails")
    print("m = Toggle ball mask window")
    print("q = Quit")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Cannot read camera frame.")
            break
        blurred = cv2.GaussianBlur(
            frame, (5, 5), 0
        )
        hsv = cv2.cvtColor(
            blurred, cv2.COLOR_BGR2HSV
        )
        blue_mask = create_mask(
            hsv, BLUE_LOWER, BLUE_UPPER
        )
        ball_mask = create_mask(
            hsv, BALL_LOWER, BALL_UPPER
        )
        ball = None
        if mode in ("t", "a"):
            last_center = ball_trail[-1] if ball_trail else None
            ball = find_tennis_ball(ball_mask, last_center)
            if ball is not None:
                ball_trail.append(ball["center"])
                ball_missed = 0
            else:
                ball_missed += 1
                if ball_missed > BALL_MAX_MISSED_FRAMES:
                    ball_trail.clear()
            draw_trail(frame, ball_trail, BALL_TRAIL_COLOR, 5)
        else:
            ball_trail.clear()
        if ball is not None:
            bx, by = ball["center"]
            radius = ball["radius"]
            ball_area = ball["area"]
            cv2.drawContours(
                frame,
                [ball["contour"]],
                -1,
                BALL_COLOR,
                2
            )
            cv2.circle(
                frame,
                (bx, by),
                max(1, int(round(radius))),
                BALL_COLOR,
                2
            )
            cv2.circle(
                frame, (bx, by), 4,
                (0, 0, 255), -1
            )
            draw_label(
                frame,
                f"TENNIS BALL Center: ({bx},{by})",
                (bx + 10, by - 35),
                BALL_COLOR
            )
            draw_label(
                frame,
                f"Radius: {radius:.1f}px",
                (bx + 10, by - 18),
                BALL_COLOR
            )
            draw_label(
                frame,
                f"Area: {ball_area:.0f}px^2",
                (bx + 10, by),
                BALL_COLOR
            )
        if mode in ("b", "a"):
            detections = find_blue_objects(blue_mask)
            tracks = tracker.update(detections)
            for track_id, track in tracks.items():
                if track["missed"] > 0:
                    continue
                cx, cy = track["centroid"]
                x, y, w, h = track["bbox"]
                area = track["area"]
                draw_trail(frame, track["trail"], BLUE_COLOR)
                cv2.drawContours(
                    frame,
                    [track["contour"]],
                    -1,
                    BLUE_COLOR,
                    2
                )
                cv2.rectangle(
                    frame,
                    (x, y),
                    (x + w, y + h),
                    BLUE_COLOR,
                    2
                )
                cv2.circle(
                    frame,
                    (cx, cy),
                    5,
                    (255, 255, 0),
                    -1
                )
                px, py = track["previous_centroid"]
                cv2.arrowedLine(
                    frame,
                    (px, py),
                    (cx, cy),
                    BLUE_COLOR,
                    2,
                    tipLength=0.3
                )
                draw_label(
                    frame,
                    f"BLUE ID {track_id}",
                    (x, y - 65),
                    BLUE_COLOR
                )
                draw_label(
                    frame,
                    f"Area: {area:.0f}px^2",
                    (x, y - 48),
                    BLUE_COLOR
                )
                
                draw_label(
                    frame,
                    f"BBox: ({x},{y},{w},{h})",
                    (x, y - 31),
                    BLUE_COLOR
                )
                draw_label(
                    frame,
                    f"Center: ({cx},{cy})",
                    (x, y - 14),
                    BLUE_COLOR
                )
                if ball is not None and mode == "a":
                    dx = cx - bx
                    dy = cy - by
                    distance = hypot(dx, dy)
                    cv2.line(
                        frame,
                        (bx, by),
                        (cx, cy),
                        RELATION_COLOR,
                        2
                    )
                    mx = (bx + cx) // 2
                    my = (by + cy) // 2
                    draw_label(
                        frame,
                        f"Blue {track_id}: dX={dx:+d}px dY={dy:+d}px",
                        (mx + 5, my - 12),
                        RELATION_COLOR,
                        0.45
                    )
                    draw_label(
                        frame,
                        f"Distance: {distance:.1f}px",
                        (mx + 5, my + 5),
                        RELATION_COLOR,
                        0.45
                    )
        cv2.rectangle(
            frame, (0, 0), (frame.shape[1], 45),
            (30, 30, 30), -1
        )
        draw_label(
            frame,
            f"MODE: {mode.upper()} | B: Blue | T: Ball | A: Both | C: Clear | Q: Quit",
            (10, 28),
            TEXT_COLOR,
            0.55
        )
        cv2.imshow(
            "Object Tracking and Measurements",
            frame
        )
        if show_mask:
            cv2.imshow("Ball mask", ball_mask)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("b"):
            mode = "b"
        elif key == ord("t"):
            mode = "t"
        elif key == ord("a"):
            mode = "a"
        elif key == ord("m"):
            show_mask = not show_mask
            if not show_mask:
                cv2.destroyWindow("Ball mask")
        elif key == ord("c"):
            ball_trail.clear()
            for track in tracker.tracks.values():
                track["trail"].clear()
    cap.release()
    cv2.destroyAllWindows()
if __name__ == "__main__":
    main()
 