import cv2
import numpy as np
# size of aruco marker is 5 cm wide 
marker_size = 0.05
# CAMERA CALIBRATION
calibration = np.load("camera_calibration.npz")
camera_matrix = calibration["camera_matrix"]
distortion = calibration["distortion"]
print("Camera calibration loaded.")
# ARUCO DETECTORS
# 4x4 dictionary
dictionary_4x4 = cv2.aruco.getPredefinedDictionary(
    cv2.aruco.DICT_4X4_50
)
detector_4x4 = cv2.aruco.ArucoDetector(dictionary_4x4)
# 5x5 dictionary
dictionary_5x5 = cv2.aruco.getPredefinedDictionary(
    cv2.aruco.DICT_5X5_50
)
detector_5x5 = cv2.aruco.ArucoDetector(dictionary_5x5)
# 6x6 dictionary
dictionary_6x6 = cv2.aruco.getPredefinedDictionary(
    cv2.aruco.DICT_6X6_50
)
detector_6x6 = cv2.aruco.ArucoDetector(dictionary_6x6)
# OPEN WEBCAM
camera = cv2.VideoCapture(0)
# IDENTIFY 3D POINTS OF THE MARKER
half_size = marker_size / 2
object_points = np.array([
    [-half_size,  half_size, 0],
    [ half_size,  half_size, 0],
    [ half_size, -half_size, 0],
    [-half_size, -half_size, 0]
], dtype=np.float32)
while True:
    # Read webcam frame
    success, frame = camera.read()
    if not success:
        print("Could not access camera.")
        break
       # DETECT 4x4 MARKERS
    corners_4x4, ids_4x4, _ = detector_4x4.detectMarkers(frame)
    # DETECT 5x5 MARKERS
    corners_5x5, ids_5x5, _ = detector_5x5.detectMarkers(frame)
    # DETECT 6x6 MARKERS
    corners_6x6, ids_6x6, _ = detector_6x6.detectMarkers(frame)
    # PROCESS DETECTED MARKERS
    marker_groups = [
        ("4x4", corners_4x4, ids_4x4),
        ("5x5", corners_5x5, ids_5x5),
        ("6x6", corners_6x6, ids_6x6)
    ]
    for dictionary_name, corners, ids in marker_groups:
        # No markers found for this dictionary
        if ids is None:
            continue
        # Draw detected marker borders
        cv2.aruco.drawDetectedMarkers(
            frame,
            corners,
            ids
        )
        # Process each marker
        for i in range(len(ids)):
            # Get the four corners
            image_points = corners[i][0].astype(np.float32)
            # POSE ESTIMATION
            success_pnp, rvec, tvec = cv2.solvePnP(
                object_points,
                image_points,
                camera_matrix,
                distortion,
                flags=cv2.SOLVEPNP_IPPE_SQUARE
            )
            if not success_pnp:
                continue
            # POSITION
            x = float(tvec[0][0])
            y = float(tvec[1][0])
            z = float(tvec[2][0])
            # MARKER ID
            marker_id = int(
                np.asarray(ids[i]).flat[0]
            )
            # DRAW XYZ AXES
            cv2.drawFrameAxes(
                frame,
                camera_matrix,
                distortion,
                rvec,
                tvec,
                marker_size * 0.5
            )
           # GET MARKER CENTER
            center_x = int(
                np.mean(image_points[:, 0])
            )
            center_y = int(
                np.mean(image_points[:, 1])
            )
            # DISPLAY INFORMATION
            cv2.putText(
                frame,
                f"{dictionary_name}  ID: {marker_id}",
                (center_x - 80, center_y - 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 0),
                2
            )
            cv2.putText(
                frame,
                f"X:{x:.2f} Y:{y:.2f} Z:{z:.2f} m",
                (center_x - 80, center_y - 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2
            )
            # Depth in centimeters
            depth_cm = z * 100
            cv2.putText(
                frame,
                f"Depth: {depth_cm:.1f} cm",
                (center_x - 80, center_y + 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 255),
                2
            )
    # DISPLAY WEBCAM
    cv2.imshow(
        "ArUco Pose Estimation - 4x4 5x5 6x6",
        frame
    )
    # QUIT
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
camera.release()
cv2.destroyAllWindows()