
import cv2
import numpy as np
import argparse
from math import hypot
BLUE_LOWER = np.array([90, 70, 40], dtype=np.uint8)
BLUE_UPPER = np.array([140, 255, 255], dtype=np.uint8)
BALL_LOWER = np.array([20, 70, 70], dtype=np.uint8)
BALL_UPPER = np.array([45, 255, 255], dtype=np.uint8)
MIN_BLUE_AREA = 300
MIN_BALL_AREA = 150
MAX_BALL_AREA = 30000
MAX_MATCH_DISTANCE = 60
MAX_MISSED_FRAMES = 10
class ObjectTracker:
    def __init__(self):
        self.next_id = 1
        self.tracks = {}
    def update(self, detections):
        """
        detections: list of dictionaries containing:
            centroid: (x, y)
            contour: OpenCV contour
            bbox: (x, y, w, h)
        """
        # Mark all tracks as unmatched for this frame.
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
                    "centroid": (cx, cy),
                    "bbox": detection["bbox"],
                    "contour": detection["contour"],
                    "missed": 0,
                    "matched": True,
                }
            else:
                track = self.tracks[best_id]
                track["centroid"] = (cx, cy)
                track["bbox"] = detection["bbox"]
                track["contour"] = detection["contour"]
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
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
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
        })
    return detections
def find_tennis_ball(mask):
    contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    candidates = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if not MIN_BALL_AREA <= area <= MAX_BALL_AREA:
            continue
        perimeter = cv2.arcLength(contour, True)
        if perimeter == 0:
            continue
        circularity = 4 * np.pi * area / (perimeter ** 2)
        if circularity < 0.55:
            continue
        (cx, cy), radius = cv2.minEnclosingCircle(contour)
        if radius < 5:
            continue
        candidates.append({
            "contour": contour,
            "center": (int(cx), int(cy)),
            "radius": int(radius),
            "area": area,
            "circularity": circularity,
        })
    if not candidates:
        return None
    return max(candidates, key=lambda item: item["circularity"])
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--camera", type=int, default=0,
        help="Camera index (default: 0)"
    )
    args = parser.parse_args()
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print("Error: Could not open the camera.")
        return
    blue_tracker = ObjectTracker()
    mode = "a"
    print("Controls:")
    print("  b : Detect and track blue objects")
    print("  t : Detect tennis ball")
    print("  a : Detect both independently")
    print("  q : Quit")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to read camera frame.")
            break
        blurred = cv2.GaussianBlur(frame, (5, 5), 0)
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
        blue_mask = create_mask(
            hsv, BLUE_LOWER, BLUE_UPPER
        )
        ball_mask = create_mask(
            hsv, BALL_LOWER, BALL_UPPER
        )
        if mode in ("b", "a"):
            detections = find_blue_objects(blue_mask)
            tracks = blue_tracker.update(detections)
            for track_id, track in tracks.items():
                if track["missed"] > 0:
                    continue
                x, y, w, h = track["bbox"]
                cx, cy = track["centroid"]
                cv2.drawContours(
                    frame, [track["contour"]], -1,
                    (255, 0, 0), 2
                )
                cv2.rectangle(
                    frame, (x, y), (x + w, y + h),
                    (255, 0, 0), 2
                )
                cv2.putText(
                    frame, f"Blue ID {track_id}",
                    (x, max(y - 8, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (255, 0, 0), 2
                )
                cv2.circle(
                    frame, (cx, cy), 4, (255, 255, 0), -1
                )
        if mode in ("t", "a"):
            ball = find_tennis_ball(ball_mask)
            if ball is not None:
                cx, cy = ball["center"]
                radius = ball["radius"]
                cv2.circle(
                    frame, (cx, cy), radius,
                    (0, 255, 0), 2
                )
                cv2.circle(
                    frame, (cx, cy), 4,
                    (0, 0, 255), -1
                )
                cv2.putText(
                    frame, "Tennis Ball",
                    (cx - radius, max(cy - radius - 10, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65, (0, 255, 0), 2
                )
        cv2.putText(
            frame, f"Mode: {mode.upper()} | b/t/a/q",
            (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
            0.7, (255, 255, 255), 2
        )
        cv2.imshow("Blue Object and Tennis Ball Tracker", frame)
        cv2.imshow("Blue Mask", blue_mask)
        cv2.imshow("Tennis Ball Mask", ball_mask)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("b"):
            mode = "b"
        elif key == ord("t"):
            mode = "t"
        elif key == ord("a"):
            mode = "a"
    cap.release()
    cv2.destroyAllWindows()
if __name__ == "__main__":
    main()
