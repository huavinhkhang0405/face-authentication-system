import cv2
import mediapipe as mp
import numpy as np

# =====================================
# HẰNG SỐ CẤU HÌNH & KHỞI TẠO
# =====================================
mp_face_mesh = mp.solutions.face_mesh
# Khởi tạo FaceMesh với static_image_mode=True để tránh lỗi timestamp
# Khi static_image_mode=True, MediaPipe sẽ không yêu cầu timestamp tăng dần
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=True,  # ✅ Sửa: True để xử lý từng frame độc lập
    max_num_faces=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

MODEL_POINTS = np.array([
    (0.0, 0.0, 0.0),          # Nose tip
    (-225.0, 170.0, -135.0),  # Left eye corner
    (225.0, 170.0, -135.0),   # Right eye corner
    (-150.0, -150.0, -125.0), # Left Mouth corner
    (150.0, -150.0, -125.0),  # Right mouth corner
    (0.0, -330.0, -65.0),     # Chin
], dtype=np.float64)

LANDMARK_INDICES = {
    'nose_tip': 1,
    'left_eye': 33,
    'right_eye': 263,
    'left_mouth': 61,
    'right_mouth': 291,
    'chin': 175
}

def get_camera_matrix(frame_width, frame_height):
    focal_length = frame_width
    center_x = frame_width / 2
    center_y = frame_height / 2
    return np.array([
        [focal_length, 0, center_x],
        [0, focal_length, center_y],
        [0, 0, 1]
    ], dtype=np.float64)

DIST_COEFFS = np.zeros((4, 1), dtype=np.float64)

# =====================================
# HÀM TÍNH GÓC (Core Logic)
# =====================================
def get_head_pose(landmarks, frame_w, frame_h):
    """
    Tính toán góc quay đầu từ các điểm landmark.
    Trả về: yaw_angle (độ), success (bool)
    """
    try:
        indices = [
            LANDMARK_INDICES['nose_tip'],
            LANDMARK_INDICES['left_eye'],
            LANDMARK_INDICES['right_eye'],
            LANDMARK_INDICES['left_mouth'],
            LANDMARK_INDICES['right_mouth'],
            LANDMARK_INDICES['chin']
        ]

        image_points = np.array([
            (landmarks.landmark[i].x * frame_w, landmarks.landmark[i].y * frame_h)
            for i in indices
        ], dtype=np.float64)

        camera_matrix = get_camera_matrix(frame_w, frame_h)

        success, rotation_vector, translation_vector = cv2.solvePnP(
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

    except Exception as e:
        print(f"Error in head pose: {e}")
        return 0.0, False

# =====================================
# HÀM MỚI CHO WEB (Stateless)
# =====================================
def calculate_yaw_from_frame(frame):
    """
    Hàm này nhận vào 1 khung hình (OpenCV BGR Image).
    Trả về: (yaw_angle, has_face)
    Không vẽ, không loop, không sleep.
    """
    h, w = frame.shape[:2]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Process FaceMesh
    results = face_mesh.process(rgb)

    if results.multi_face_landmarks:
        # Lấy khuôn mặt đầu tiên
        face_landmarks = results.multi_face_landmarks[0]
        yaw, success = get_head_pose(face_landmarks, w, h)
        if success:
            return yaw, True
    
    return 0.0, False