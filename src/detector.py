import os
import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm
import time
from insightface.app import FaceAnalysis


class FaceDetector:
    def __init__(self, required_size=(160, 160)):
        # Khởi tạo RetinaFace
        self.required_size = required_size
        self.detector = FaceAnalysis(name="buffalo_sc")
        # ctx_id: 0 = GPU, -1 = CPU
        self.detector.prepare(ctx_id=-1, det_size=(480, 480))
        print("⚡ Đang dùng RetinaFace (insightface) để detect khuôn mặt")

    # ==============================
    # 1️⃣ Detect khuôn mặt từ file ảnh
    # ==============================
    def extract_face(self, filename, multiple=False):
        img = cv2.imread(filename)
        if img is None:
            print(f"❌ Không thể đọc file: {filename}")
            return None

        faces, _ = self._detect_faces(img, multiple=multiple)
        if len(faces) == 0:
            print(f"❌ Không phát hiện khuôn mặt trong ảnh: {filename}")
            return None

        return faces if multiple else faces[0]

    # ==============================
    # 2️⃣ Detect từ thư mục ảnh (huấn luyện)
    # ==============================
    def extract_faces_from_dir(self, input_dir, output_dir):
        os.makedirs(output_dir, exist_ok=True)
        person_dirs = [
            d for d in os.listdir(input_dir)
            if os.path.isdir(os.path.join(input_dir, d))
        ]

        for person in tqdm(person_dirs, desc="🔍 Đang xử lý thư mục"):
            person_path = os.path.join(input_dir, person)
            save_path = os.path.join(output_dir, person)
            os.makedirs(save_path, exist_ok=True)

            for file in os.listdir(person_path):
                if not file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    continue
                src = os.path.join(person_path, file)
                faces = self.extract_face(src, multiple=False)
                if faces is None:
                    continue
                cv2.imwrite(
                    os.path.join(save_path, file),
                    cv2.cvtColor(faces, cv2.COLOR_RGB2BGR),
                )

        print("✅ Hoàn tất: đã lưu các khuôn mặt đã detect & resize vào:", output_dir)

    # ==============================
    # 3️⃣ Detect khuôn mặt từ frame (realtime)
    # ==============================
    def detect_from_frame(self, frame, multiple=True):
        """Phát hiện khuôn mặt từ frame (OpenCV BGR)"""
        faces, boxes = self._detect_faces(frame, multiple=multiple)
        return faces, boxes

    # ==============================
    # 4️⃣ Lưu khuôn mặt người mới (gán nhãn)
    # ==============================
    def save_face(self, face, user_name, save_dir="data/raw"):
        """Lưu khuôn mặt người mới vào thư mục tương ứng"""
        user_path = os.path.join(save_dir, user_name)
        os.makedirs(user_path, exist_ok=True)
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        filename = os.path.join(user_path, f"{timestamp}.jpg")
        cv2.imwrite(filename, cv2.cvtColor(face, cv2.COLOR_RGB2BGR))
        print(f"💾 Đã lưu khuôn mặt vào: {filename}")

    # ==============================
    # 🔧 Hàm nội bộ: dùng RetinaFace để detect
    # ==============================
    def _detect_faces(self, frame, multiple=True):
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.detector.get(frame)

        faces, boxes = [], []
        if len(results) == 0:
            return [], []

        for res in results:
            x1, y1, x2, y2 = res.bbox.astype(int)
            face = img_rgb[y1:y2, x1:x2]
            if face.size == 0:
                continue

            face_img = Image.fromarray(face).resize(self.required_size)
            faces.append(np.asarray(face_img))
            boxes.append((x1, y1, x2, y2))

            if not multiple:
                break

        return faces, boxes
