import os
import shutil
import cryptography.fernet as fernet
from cryptography.fernet import Fernet
from src.utils import config

TARGET_FOLDER = [config.RAW_DIR, config.ALIGNED_DIR]

OUTPUT_FOLDER = os.path.join(config.DATA_DIR, "encrypted_data")

KEY_FILE = config.KEY_FILE

def load_or_generate_key():
    """Tự động tạo khóa nếu chưa có, hoặc load khóa cũ"""
    if os.path.exists(KEY_FILE):
        return open(KEY_FILE, "rb").read()
    else:
        key = Fernet.generate_key()
        with open(KEY_FILE, "wb") as f:
            f.write(key)
        print(f"[KEY] Đã tạo khóa mới: {KEY_FILE} (LƯU TRỮ CẨN THẬN)")
        return key

def encrypt_all(delete_original=False):
    """Mã hóa tất cả ảnh trong TARGET_FOLDER
    
    Args:
        delete_original (bool): Nếu True, xóa folder gốc sau khi mã hóa thành công
    
    Returns:
        bool: True nếu thành công, False nếu có lỗi
    """
    key = load_or_generate_key()
    cipher = Fernet(key)

    print(f"\n{'='*60}")
    print(f"🔒 BẮT ĐẦU MÃ HÓA DỮ LIỆU")
    print(f"{'='*60}")
    print(f"📁 Nguồn: {config.DATA_DIR}")
    print(f"💾 Đích : {OUTPUT_FOLDER}")
    print(f"🗑️  Xóa gốc: {'CÓ' if delete_original else 'KHÔNG'}")
    print(f"{'='*60}\n")

    total_encrypted = 0
    total_errors = 0

    for folder_path in TARGET_FOLDER:
        if not os.path.exists(folder_path):
            print(f"⚠️  [CẢNH BÁO] Không tìm thấy folder: {folder_path}")
            continue

        folder_name = os.path.basename(folder_path)
        print(f"\n📂 Đang xử lý: {folder_name}/")
        
        # Duyệt cây thư mục
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                if file.startswith(".") or file.endswith(".enc"): 
                    continue
                
                # 1. Đường dẫn file gốc
                src_path = os.path.join(root, file)
                
                # 2. Tính toán đường dẫn đích để giữ nguyên cấu trúc
                rel_path = os.path.relpath(src_path, start=config.DATA_DIR)
                
                # Ghép với folder đích + thêm đuôi .enc
                dest_path = os.path.join(OUTPUT_FOLDER, rel_path) + ".enc"
                
                # Tạo thư mục cha nếu chưa có
                os.makedirs(os.path.dirname(dest_path), exist_ok=True)

                # 3. Mã hóa và ghi file
                try:
                    with open(src_path, "rb") as f:
                        data = f.read()
                    
                    encrypted_data = cipher.encrypt(data)
                    
                    with open(dest_path, "wb") as f:
                        f.write(encrypted_data)
                    
                    total_encrypted += 1
                    print(f"  ✅ {rel_path}")
                    
                except Exception as e:
                    total_errors += 1
                    print(f"  ❌ [LỖI] {src_path}: {e}")

    print(f"\n{'='*60}")
    print(f"📊 THỐNG KÊ")
    print(f"{'='*60}")
    print(f"✅ Đã mã hóa: {total_encrypted} file")
    print(f"❌ Lỗi: {total_errors} file")
    print(f"💾 Lưu tại: {OUTPUT_FOLDER}")
    
    # Xóa folder gốc nếu được yêu cầu và không có lỗi
    if delete_original and total_errors == 0:
        print(f"\n{'='*60}")
        print(f"🗑️  XÓA DỮ LIỆU GỐC (Đã mã hóa an toàn)")
        print(f"{'='*60}")
        
        for folder_path in TARGET_FOLDER:
            if os.path.exists(folder_path):
                try:
                    shutil.rmtree(folder_path)
                    print(f"✅ Đã xóa: {folder_path}")
                except Exception as e:
                    print(f"❌ Lỗi xóa {folder_path}: {e}")
                    return False
        
        print(f"\n⚠️  LƯU Ý: Chỉ có thể khôi phục dữ liệu bằng file khóa: {KEY_FILE}")
        print(f"🔑 HÃY LƯU TRỮ FILE KHÓA CẨN THẬN!")
    
    elif delete_original and total_errors > 0:
        print(f"\n⚠️  KHÔNG XÓA dữ liệu gốc vì có {total_errors} lỗi mã hóa")
    
    print(f"{'='*60}\n")
    return total_errors == 0

def decrypt_all(encrypted_dir=OUTPUT_FOLDER, output_dir=None):
    """Giải mã dữ liệu đã mã hóa
    
    Args:
        encrypted_dir (str): Thư mục chứa file đã mã hóa
        output_dir (str): Thư mục xuất file giải mã (mặc định là DATA_DIR)
    
    Returns:
        bool: True nếu thành công
    """
    if output_dir is None:
        output_dir = config.DATA_DIR
    
    if not os.path.exists(KEY_FILE):
        print(f"❌ Không tìm thấy file khóa: {KEY_FILE}")
        return False
    
    key = open(KEY_FILE, "rb").read()
    cipher = Fernet(key)
    
    print(f"\n🔓 BẮT ĐẦU GIẢI MÃ DỮ LIỆU")
    print(f"📁 Nguồn: {encrypted_dir}")
    print(f"💾 Đích : {output_dir}\n")
    
    total_decrypted = 0
    
    for root, dirs, files in os.walk(encrypted_dir):
        for file in files:
            if not file.endswith(".enc"):
                continue
            
            src_path = os.path.join(root, file)
            
            # Tính đường dẫn đích (bỏ .enc)
            rel_path = os.path.relpath(src_path, start=encrypted_dir)
            dest_path = os.path.join(output_dir, rel_path[:-4])  # Remove .enc
            
            os.makedirs(os.path.dirname(dest_path), exist_ok=True)
            
            try:
                with open(src_path, "rb") as f:
                    encrypted_data = f.read()
                
                decrypted_data = cipher.decrypt(encrypted_data)
                
                with open(dest_path, "wb") as f:
                    f.write(decrypted_data)
                
                total_decrypted += 1
                print(f"✅ {rel_path[:-4]}")
                
            except Exception as e:
                print(f"❌ Lỗi giải mã {src_path}: {e}")
    
    print(f"\n✅ Đã giải mã {total_decrypted} file")
    return True

if __name__ == "__main__":
    import sys
    
    # Kiểm tra xem config có đúng không trước khi chạy
    print(f"Đang đọc config từ: {config.__file__}")
    if not os.path.exists(config.DATA_DIR):
        print(f"Lỗi: Không tìm thấy DATA_DIR tại {config.DATA_DIR}")
    else:
        # Hỗ trợ tham số dòng lệnh
        if len(sys.argv) > 1 and sys.argv[1] == "decrypt":
            decrypt_all()
        else:
            # Mã hóa và hỏi có xóa gốc không
            delete = input("\n⚠️  Xóa dữ liệu gốc sau khi mã hóa? (yes/no): ").lower() == "yes"
            encrypt_all(delete_original=delete)