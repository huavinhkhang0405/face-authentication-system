import cv2
import time
from detector import FaceDetector
from config import RAW_DIR
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# =====================================
# CẤU HÌNH
# =====================================
SAVE_INTERVAL = 0.3      # lưu mỗi 0.3s (giữa 2 ảnh)
MAX_IMAGES = 30          # mỗi người lưu 30 ảnh
IMAGE_SIZE = None 
# =====================================

def capture_images():
    # Khởi tạo detector
    detector = FaceDetector(required_size=IMAGE_SIZE)

    # Tạo webcam
    cap = cv2.VideoCapture(0)
    cap.set(3, 640)
    cap.set(4, 480)

    # Nhập thông tin người cần thu
    name = input("📸 Nhập tên hoặc MSSV sinh viên: ").strip()
    save_dir = os.path.join(RAW_DIR, name)
    os.makedirs(save_dir, exist_ok=True)
    print(f"➡️ Ảnh sẽ được lưu vào: {save_dir}")

    count = 0
    last_save = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Không thể mở camera.")
            break

        # Detect khuôn mặt
        faces, boxes = detector.detect_from_frame(frame)
        print(f"Detect: {len(faces)} mặt - Boxes: {boxes}")

        for face, (x1, y1, x2, y2) in zip(faces, boxes):
            # Vẽ khung nhận diện
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
            cv2.putText(frame, f"{name}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

            # Lưu ảnh mỗi SAVE_INTERVAL giây
            if time.time() - last_save > SAVE_INTERVAL and count < MAX_IMAGES:
                filename = os.path.join(save_dir, f"{name}_{count:03d}.jpg")
                cv2.imwrite(filename, cv2.cvtColor(face, cv2.COLOR_RGB2BGR))
                count += 1
                last_save = time.time()
                print(f"💾 Đã lưu {filename}")

        # Hiển thị camera
        cv2.putText(frame, f"Saved: {count}/{MAX_IMAGES}", (10,30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0), 2)
        cv2.imshow("Data Collection", frame)

        # Nhấn q để thoát
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        if count >= MAX_IMAGES:
            print("✅ Đã thu đủ ảnh.")
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    capture_images()
