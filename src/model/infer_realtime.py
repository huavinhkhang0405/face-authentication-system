import cv2
import time
import joblib
import os
import ast
from src.model.detector import FaceDetector
from src.model.embedder import FaceEmbedder
from src.model.liveness import check_liveness_senior, YAW_THRESHOLD, TIMEOUT
from src.utils.config import MODEL_DIR, RESULT_DIR


def recognize_realtime():
    knn = joblib.load(os.path.join(MODEL_DIR, "best_knn_faceid.pkl"))
    le = joblib.load(os.path.join(MODEL_DIR, "label_encoder.pkl"))

    with open(os.path.join(RESULT_DIR, "best_params_faceid.txt"), "r") as f:
        params = ast.literal_eval(f.readline().split("},")[0] + "}")
    metric = params.get("metric", "euclidean")

    threshold = 0.55 if metric == "euclidean" else 0.4 if metric == "cosine" else None

    detector = FaceDetector()
    embedder = FaceEmbedder()

    cap = cv2.VideoCapture(0)
    cap.set(3, 640)
    cap.set(4, 480)

    print("🚀 Bắt đầu nhận diện realtime (nhấn Q để thoát)")

    liveness_passed = False
    last_seen = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        start = time.time()

        faces, boxes = detector.detect_from_frame(frame)

        # Nếu thấy mặt lần đầu → yêu cầu check liveness
        if faces and not liveness_passed:
            print("🔍 Detect face → Bắt đầu kiểm tra liveness (quay đầu sang trái)...")
            print(f"📋 Yêu cầu: Quay đầu sang trái ít nhất {YAW_THRESHOLD}° trong {TIMEOUT}s")
            
            # Gọi hàm kiểm tra liveness mới (head rotation)
            is_live, liveness_frame, liveness_message = check_liveness_senior(
                cap, 
                timeout=TIMEOUT, 
                yaw_threshold=YAW_THRESHOLD
            )

            if is_live:
                print(f"✅ Liveness passed! {liveness_message}")
                print("🎯 Tiếp tục nhận diện khuôn mặt...")
                liveness_passed = True
            else:
                print(f"❌ Liveness failed! {liveness_message}")
                # Hiển thị frame liveness với thông báo lỗi
                if liveness_frame is not None and liveness_frame.size > 0:
                    cv2.putText(liveness_frame, "Liveness Failed - Please try again", 
                               (20, liveness_frame.shape[0] - 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    cv2.imshow("Realtime Face Recognition", liveness_frame)
                    cv2.waitKey(2000)  # Hiển thị 2 giây
                continue

        # Nếu mất mặt → reset liveness
        if not faces:
            if time.time() - last_seen > 1.0:
                liveness_passed = False
        else:
            last_seen = time.time()

        if liveness_passed:
            for face, (x1, y1, x2, y2) in zip(faces, boxes):
                emb = embedder.embed_face(face).reshape(1, -1)
                dist = knn.kneighbors(emb, n_neighbors=1)[0][0][0]
                pred = knn.predict(emb)[0]
                name = le.inverse_transform([pred])[0]
                if dist > threshold:
                    name = "Unknown"

                color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, f"{name} ({dist:.2f})", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        fps = 1 / max(time.time() - start, 1e-5)
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0), 2)
        
        # Hiển thị trạng thái liveness
        if liveness_passed:
            liveness_status = "✅ Liveness: Verified"
            status_color = (0, 255, 0)
        else:
            liveness_status = "⏳ Liveness: Waiting..."
            status_color = (0, 165, 255)
        
        cv2.putText(frame, liveness_status, (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)

        cv2.imshow("Realtime Face Recognition", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    recognize_realtime()
