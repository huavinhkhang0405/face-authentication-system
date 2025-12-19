import torch
import numpy as np
import cv2
from facenet_pytorch import InceptionResnetV1
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder
import os, joblib
#from src.config import ALIGNED_DIR, DATA_DIR, MODEL_DIR, KEY_FILE
from config import ALIGNED_DIR, DATA_DIR, MODEL_DIR, KEY_FILE
from cryptography.fernet import Fernet

# Đường dẫn encrypted_data
ENCRYPTED_ALIGNED_DIR = os.path.join(DATA_DIR, "encrypted_data", "faces_aligned")

class FaceEmbedder:
    def __init__(self, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print("⚙️ Đang sử dụng thiết bị:", self.device)
        self.model = InceptionResnetV1(pretrained='vggface2').eval().to(self.device)

    def embed_face(self, img_rgb):
        img = cv2.resize(img_rgb, (160, 160))
        img = (img - 127.5) / 128.0  # chuẩn hóa theo FaceNet gốc
        tensor = torch.tensor(img).permute(2, 0, 1).float().unsqueeze(0)
        with torch.no_grad():
            emb = self.model(tensor.to(self.device)).cpu().numpy().squeeze()
        emb = emb / np.linalg.norm(emb)
        return emb

    def build_embeddings(self, input_dir=None, out_dir=DATA_DIR):
        """Tạo embeddings - Hỗ trợ cả aligned thường và encrypted aligned
        
        Args:
            input_dir: Nếu None, tự động chọn ENCRYPTED_ALIGNED_DIR hoặc ALIGNED_DIR
            out_dir: Thư mục lưu embeddings.npy và labels.npy
        """
        # Tự động chọn thư mục nguồn
        if input_dir is None:
            if os.path.exists(ENCRYPTED_ALIGNED_DIR):
                input_dir = ENCRYPTED_ALIGNED_DIR
                is_encrypted = True
                print(f"📁 Sử dụng encrypted aligned data: {ENCRYPTED_ALIGNED_DIR}")
            elif os.path.exists(ALIGNED_DIR):
                input_dir = ALIGNED_DIR
                is_encrypted = False
                print(f"📁 Sử dụng aligned data thường: {ALIGNED_DIR}")
            else:
                raise ValueError("Không tìm thấy thư mục aligned data!")
        else:
            # Kiểm tra xem input_dir có phải encrypted không
            is_encrypted = 'encrypted' in input_dir.lower()
        
        # Load khóa nếu cần
        cipher = None
        if is_encrypted:
            if not os.path.exists(KEY_FILE):
                raise ValueError(f"Không tìm thấy file khóa: {KEY_FILE}")
            key = open(KEY_FILE, "rb").read()
            cipher = Fernet(key)
        
        X, y = [], []
        os.makedirs(MODEL_DIR, exist_ok=True)
        persons = [d for d in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, d))]
        print(f"📦 Tổng số người cần xử lý: {len(persons)}")

        for person in tqdm(persons, desc="🔹 Tạo embedding"):
            person_path = os.path.join(input_dir, person)
            
            # Lọc file theo loại
            if is_encrypted:
                img_files = [f for f in os.listdir(person_path) if f.endswith('.enc')]
            else:
                img_files = [f for f in os.listdir(person_path) if f.lower().endswith(('.jpg','.png','.jpeg'))]
            
            for img_name in img_files:
                try:
                    img_path = os.path.join(person_path, img_name)
                    
                    if is_encrypted:
                        # Giải mã ảnh
                        with open(img_path, 'rb') as f:
                            encrypted_data = f.read()
                        decrypted_data = cipher.decrypt(encrypted_data)
                        img_array = np.frombuffer(decrypted_data, dtype=np.uint8)
                        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                    else:
                        # Đọc ảnh thường
                        img = cv2.imread(img_path)
                    
                    if img is None:
                        print(f"⚠️ Không đọc được ảnh: {img_path}")
                        continue
                    
                    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    emb = self.embed_face(img_rgb)
                    
                    if np.isnan(emb).any():
                        print(f"⚠️ Bỏ ảnh lỗi (NaN): {img_name}")
                        continue
                    
                    X.append(emb)
                    y.append(person)
                    
                except Exception as e:
                    print(f"❌ Lỗi xử lý {img_name}: {e}")
                    continue

        X, y = np.array(X), np.array(y)
        le = LabelEncoder()
        y_enc = le.fit_transform(y)

        np.save(os.path.join(out_dir, "embeddings.npy"), X)
        np.save(os.path.join(out_dir, "labels.npy"), y_enc)
        joblib.dump(le, os.path.join(MODEL_DIR, "label_encoder.pkl"))

        print(f"✅ Đã lưu {len(X)} embedding, {len(np.unique(y_enc))} lớp.")
        
        return X, y_enc


if __name__ == "__main__":
    embedder = FaceEmbedder()
    embedder.build_embeddings()
