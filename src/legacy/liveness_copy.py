import cv2
import mediapipe as mp
import numpy as np
import time
from PIL import ImageFont, ImageDraw, Image

# =============================
# HÀM VẼ TIẾNG VIỆT
# =============================
def draw_vn_text(frame, text, x, y, color=(0, 255, 0), size=24):
    """
    Vẽ chữ tiếng Việt lên frame bằng Pillow.
    """
    img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)

    try:
        font = ImageFont.truetype("arial.ttf", size)  # Font Unicode
    except:
        font = ImageFont.load_default()

    draw.text((x, y), text, font=font, fill=color)
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)


# =====================================
# HẰNG SỐ CẤU HÌNH
# =====================================
YAW_THRESHOLD = 20.0  
TIMEOUT = 5  

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
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
    (0.0, -330.0, -65.0),
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
# HEAD POSE
# =====================================
def get_head_pose(landmarks, frame_w, frame_h):
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

    except:
        return 0.0, False


# =====================================
# LIVENESS – QUAY ĐẦU SANG TRÁI
# =====================================
def check_liveness_senior(cap, timeout=TIMEOUT, yaw_threshold=YAW_THRESHOLD):
    start_time = time.time()
    initial_yaw = None
    initial_yaw_set = False
    liveness = False
    final_frame = None
    message = ""

    while time.time() - start_time < timeout:
        ret, frame = cap.read()
        if not ret:
            return False, np.zeros((480, 640, 3)), "Không thể đọc camera!"

        h, w = frame.shape[:2]
        elapsed = time.time() - start_time
        remaining = max(0, timeout - elapsed)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)

        instruction_y = 40
        line_h = 35

        if results.multi_face_landmarks:
            for land in results.multi_face_landmarks:
                yaw, ok = get_head_pose(land, w, h)

                if ok:
                    if not initial_yaw_set:
                        initial_yaw = yaw
                        initial_yaw_set = True
                        message = "Đã phát hiện khuôn mặt. Vui lòng quay đầu sang trái..."

                    deviation = initial_yaw - yaw

                    if deviation >= yaw_threshold:
                        liveness = True
                        message = "✔ Xác thực thành công!"
                        final_frame = frame.copy()
                        time.sleep(0.5)
                        break
                    else:
                        progress = min(100, int(deviation / yaw_threshold * 100))
                        message = f"Quay đầu sang trái... ({progress}%)"
        else:
            message = "Vui lòng đưa khuôn mặt vào khung hình..."
            initial_yaw_set = False

        # ======================
        # HIỂN THỊ VIỆT HÓA
        # ======================
        frame = draw_vn_text(frame, "QUAY ĐẦU SANG TRÁI", 20, instruction_y, (255, 255, 0), 30)
        frame = draw_vn_text(frame, message, 20, instruction_y + line_h, (0, 255, 0), 24)
        frame = draw_vn_text(frame, f"Thời gian: {remaining:.1f}s", 20, instruction_y + line_h * 2, (255, 255, 255), 24)

        if initial_yaw_set:
            yaw_text = f"Góc lệch: {deviation:.1f}° / {yaw_threshold}°"
            color = (0,255,0) if deviation >= yaw_threshold else (255,200,0)
            frame = draw_vn_text(frame, yaw_text, 20, instruction_y + line_h * 3, color, 24)

        cv2.imshow("Liveness Detection - Rotate Head Left", frame)

        if liveness:
            return True, final_frame, message

        if cv2.waitKey(1) & 0xFF == ord('q'):
            return False, frame, "Bạn đã hủy"

    # Nếu hết thời gian mà chưa đạt
    if not liveness:
        if not initial_yaw_set:
            message = "Không phát hiện khuôn mặt!"
        else:
            message = "Chưa quay đầu đủ góc yêu cầu!"

        return False, frame, message


# =====================================
# MAIN
# =====================================
if __name__ == '__main__':
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    print("Bắt đầu kiểm tra liveness...")

    ok, final_frame, msg = check_liveness_senior(cap)
    cap.release()
    cv2.destroyAllWindows()

    print("Kết quả:", msg)

    if final_frame is not None:
        cv2.imshow("Kết quả", final_frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
