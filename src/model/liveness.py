import cv2
import mediapipe as mp
import numpy as np
import time

YAW_THRESHOLD = 20.0
TIMEOUT = 5

mp_face_mesh = mp.solutions.face_mesh
face_mesh_static = mp_face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
face_mesh_stream = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

MODEL_POINTS = np.array([
    (0.0, 0.0, 0.0),
    (-225.0, 170.0, -135.0),
    (225.0, 170.0, -135.0),
    (-150.0, -150.0, -125.0),
    (150.0, -150.0, -125.0),
    (0.0, -330.0, -65.0)
], dtype=np.float64)

LANDMARK_INDICES = {
    "nose_tip": 1,
    "left_eye": 33,
    "right_eye": 263,
    "left_mouth": 61,
    "right_mouth": 291,
    "chin": 175
}

DIST_COEFFS = np.zeros((4, 1), dtype=np.float64)


def get_camera_matrix(frame_width, frame_height):
    focal_length = frame_width
    center_x = frame_width / 2
    center_y = frame_height / 2
    return np.array([
        [focal_length, 0, center_x],
        [0, focal_length, center_y],
        [0, 0, 1]
    ], dtype=np.float64)


def get_head_pose(landmarks, frame_w, frame_h):
    try:
        indices = [
            LANDMARK_INDICES["nose_tip"],
            LANDMARK_INDICES["left_eye"],
            LANDMARK_INDICES["right_eye"],
            LANDMARK_INDICES["left_mouth"],
            LANDMARK_INDICES["right_mouth"],
            LANDMARK_INDICES["chin"]
        ]

        image_points = np.array([
            (landmarks.landmark[i].x * frame_w, landmarks.landmark[i].y * frame_h)
            for i in indices
        ], dtype=np.float64)

        camera_matrix = get_camera_matrix(frame_w, frame_h)

        success, rotation_vector, _ = cv2.solvePnP(
            MODEL_POINTS,
            image_points,
            camera_matrix,
            DIST_COEFFS,
            flags=cv2.SOLVEPNP_ITERATIVE
        )

        if not success:
            return 0.0, False

        rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
        sy = np.sqrt(rotation_matrix[0, 0] ** 2 + rotation_matrix[1, 0] ** 2)
        singular = sy < 1e-6

        if not singular:
            yaw = np.arctan2(rotation_matrix[1, 0], rotation_matrix[0, 0])
        else:
            yaw = np.arctan2(-rotation_matrix[0, 1], rotation_matrix[1, 1])

        yaw_angle = np.degrees(yaw)
        return yaw_angle, True

    except Exception as exc:
        print(f"Error in head pose: {exc}")
        return 0.0, False


def calculate_yaw_from_frame(frame):
    h, w = frame.shape[:2]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh_static.process(rgb)

    if results.multi_face_landmarks:
        face_landmarks = results.multi_face_landmarks[0]
        yaw, success = get_head_pose(face_landmarks, w, h)
        if success:
            return yaw, True

    return 0.0, False


def check_liveness_senior(cap, timeout=TIMEOUT, yaw_threshold=YAW_THRESHOLD):
    start_time = time.time()
    initial_yaw = None
    initial_set = False
    last_frame = None

    while time.time() - start_time < timeout:
        ret, frame = cap.read()
        if not ret:
            blank = np.zeros((480, 640, 3), dtype=np.uint8)
            return False, blank, "Cannot read camera"

        last_frame = frame
        h, w = frame.shape[:2]
        elapsed = time.time() - start_time
        remaining = max(0.0, timeout - elapsed)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh_stream.process(rgb)

        message = "Move face into frame."
        deviation = 0.0

        if results.multi_face_landmarks:
            face_landmarks = results.multi_face_landmarks[0]
            yaw, ok = get_head_pose(face_landmarks, w, h)

            if ok:
                if not initial_set:
                    initial_yaw = yaw
                    initial_set = True
                    message = "Face detected. Turn head left."

                deviation = initial_yaw - yaw
                if deviation >= yaw_threshold:
                    return True, frame.copy(), "Liveness verified"

                progress = int(min(100, (deviation / yaw_threshold) * 100))
                message = f"Turn left... ({progress}%)"

        cv2.putText(frame, "Liveness: turn head left", (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        cv2.putText(frame, message, (20, 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(frame, f"Time: {remaining:.1f}s", (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        if initial_set:
            color = (0, 255, 0) if deviation >= yaw_threshold else (255, 200, 0)
            cv2.putText(frame, f"Yaw: {deviation:.1f}/{yaw_threshold}", (20, 105),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        cv2.imshow("Liveness Detection - Turn Head Left", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            return False, frame, "Canceled"

    if not initial_set:
        return False, last_frame, "No face detected"

    return False, last_frame, "Did not reach required yaw"