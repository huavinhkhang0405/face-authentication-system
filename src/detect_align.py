import os, cv2
import unicodedata
from tqdm import tqdm
from config import RAW_DIR, ALIGNED_DIR, IMAGE_SIZE, DATA_DIR, KEY_FILE
from cryptography.fernet import Fernet
import numpy as np

# Đường dẫn encrypted_data
ENCRYPTED_RAW_DIR = os.path.join(DATA_DIR, "encrypted_data", "raw")
ENCRYPTED_ALIGNED_DIR = os.path.join(DATA_DIR, "encrypted_data", "faces_aligned")

def remove_accents(text):
    """Chuyển văn bản tiếng Việt có dấu thành không dấu"""
    nfd = unicodedata.normalize('NFD', text)
    without_accents = ''.join(c for c in nfd if unicodedata.category(c) != 'Mn')
    replacements = {
        'đ': 'd', 'Đ': 'D',
        ' ': '', '-': '', '_': ''
    }
    for old, new in replacements.items():
        without_accents = without_accents.replace(old, new)
    return without_accents

def align_faces():
    """Chuẩn hóa tất cả ảnh - Hỗ trợ encrypted data"""
    
    # Load khóa mã hóa nếu có
    cipher = None
    if os.path.exists(KEY_FILE):
        key = open(KEY_FILE, "rb").read()
        cipher = Fernet(key)
    
    # Kiểm tra nguồn: RAW_DIR hoặc ENCRYPTED_RAW_DIR
    source_dir = None
    is_encrypted = False
    
    if os.path.exists(RAW_DIR) and os.listdir(RAW_DIR):
        source_dir = RAW_DIR
        is_encrypted = False
        print(f"📁 Nguồn: {RAW_DIR}")
    elif os.path.exists(ENCRYPTED_RAW_DIR) and os.listdir(ENCRYPTED_RAW_DIR):
        source_dir = ENCRYPTED_RAW_DIR
        is_encrypted = True
        print(f"📁 Nguồn: {ENCRYPTED_RAW_DIR} (encrypted)")
        if not cipher:
            print("❌ Không tìm thấy file khóa để giải mã!")
            return
    else:
        print("❌ Không tìm thấy dữ liệu ảnh!")
        return
    
    # Tạo thư mục đích
    if is_encrypted:
        output_dir = ENCRYPTED_ALIGNED_DIR
        print(f"💾 Đích: {ENCRYPTED_ALIGNED_DIR} (encrypted)")
    else:
        output_dir = ALIGNED_DIR
        print(f"💾 Đích: {ALIGNED_DIR}")
    
    os.makedirs(output_dir, exist_ok=True)
    
    person_dirs = [d for d in os.listdir(source_dir) if os.path.isdir(os.path.join(source_dir, d))]
    print(f"📦 Tổng số người: {len(person_dirs)}")

    for person in tqdm(person_dirs, desc="📦 Chuẩn hóa khuôn mặt"):
        person_path = os.path.join(source_dir, person)
        save_path = os.path.join(output_dir, person)
        os.makedirs(save_path, exist_ok=True)

        for file in os.listdir(person_path):
            try:
                # Lọc file theo loại
                if is_encrypted and not file.endswith('.enc'):
                    continue
                elif not is_encrypted and not file.lower().endswith(('.jpg','.jpeg','.png')):
                    continue
                
                file_path = os.path.join(person_path, file)
                
                if is_encrypted:
                    # Giải mã ảnh
                    with open(file_path, 'rb') as f:
                        encrypted_data = f.read()
                    decrypted_data = cipher.decrypt(encrypted_data)
                    img_array = np.frombuffer(decrypted_data, dtype=np.uint8)
                    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                else:
                    # Đọc ảnh thường
                    img = cv2.imread(file_path)
                
                if img is None:
                    continue
                
                # Resize
                resized = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
                
                if is_encrypted:
                    # Encode và mã hóa lại
                    _, img_encoded = cv2.imencode('.jpg', resized)
                    img_bytes = img_encoded.tobytes()
                    encrypted_aligned = cipher.encrypt(img_bytes)
                    
                    # Lưu file encrypted
                    output_filename = file if file.endswith('.enc') else file + '.enc'
                    output_path = os.path.join(save_path, output_filename)
                    with open(output_path, 'wb') as f:
                        f.write(encrypted_aligned)
                else:
                    # Lưu ảnh thường
                    cv2.imwrite(os.path.join(save_path, file), resized)
                    
            except Exception as e:
                print(f"❌ Lỗi xử lý {file}: {e}")
                continue

    print(f"✅ Đã chuẩn hóa ảnh vào: {output_dir}")

def align_faces_for_student(student_folder_name):
    """Chuẩn hóa ảnh cho một sinh viên cụ thể - Hỗ trợ encrypted data"""
    
    # Load khóa mã hóa
    if not os.path.exists(KEY_FILE):
        print(f"❌ Không tìm thấy file khóa: {KEY_FILE}")
        return 0
    
    key = open(KEY_FILE, "rb").read()
    cipher = Fernet(key)
    
    # Kiểm tra nguồn: RAW_DIR hoặc ENCRYPTED_RAW_DIR
    person_path = os.path.join(RAW_DIR, student_folder_name)
    is_encrypted = False
    
    if not os.path.exists(person_path):
        # Thử tìm trong encrypted_data
        person_path = os.path.join(ENCRYPTED_RAW_DIR, student_folder_name)
        is_encrypted = True
        
        if not os.path.exists(person_path):
            print(f"❌ Không tìm thấy folder: {student_folder_name}")
            return 0
    
    # Chuẩn bị thư mục đích (luôn lưu vào encrypted_aligned)
    save_path = os.path.join(ENCRYPTED_ALIGNED_DIR, student_folder_name)
    os.makedirs(save_path, exist_ok=True)
    
    aligned_count = 0
    for file in os.listdir(person_path):
        # Nếu là encrypted, file có đuôi .enc
        if is_encrypted and not file.endswith('.enc'):
            continue
        elif not is_encrypted and not file.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue
        
        try:
            file_path = os.path.join(person_path, file)
            
            if is_encrypted:
                # Giải mã ảnh
                with open(file_path, 'rb') as f:
                    encrypted_data = f.read()
                decrypted_data = cipher.decrypt(encrypted_data)
                
                # Decode thành ảnh
                img_array = np.frombuffer(decrypted_data, dtype=np.uint8)
                img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            else:
                # Đọc ảnh thường
                img = cv2.imread(file_path)
            
            if img is None:
                continue
            
            # Resize ảnh
            resized = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
            
            # Encode lại thành bytes
            _, img_encoded = cv2.imencode('.jpg', resized)
            img_bytes = img_encoded.tobytes()
            
            # Mã hóa và lưu
            encrypted_aligned = cipher.encrypt(img_bytes)
            
            # Tên file output (bỏ .enc nếu có, thêm .enc lại)
            output_filename = file.replace('.enc', '') + '.enc' if not file.endswith('.enc') else file
            output_path = os.path.join(save_path, output_filename)
            
            with open(output_path, 'wb') as f:
                f.write(encrypted_aligned)
            
            aligned_count += 1
            
        except Exception as e:
            print(f"❌ Lỗi xử lý {file}: {e}")
            continue
    
    print(f"✅ Đã chuẩn hóa {aligned_count} ảnh cho {student_folder_name}")
    return aligned_count

if __name__ == "__main__":
    align_faces()
