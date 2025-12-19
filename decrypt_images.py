import sys
import os

# Thêm thư mục src vào path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(BASE_DIR, "src"))

from image_encryption import decrypt_all
from src.config import DATA_DIR, KEY_FILE

def main():
    print("="*60)
    print("🔓 GIẢI MÃ DỮ LIỆU ẢNH")
    print("="*60)
    
    # Kiểm tra file khóa
    if not os.path.exists(KEY_FILE):
        print(f"❌ KHÔNG TÌM THẤY FILE KHÓA: {KEY_FILE}")
        print("⚠️  Không thể giải mã mà không có file khóa!")
        return
    
    print(f"✅ Đã tìm thấy file khóa: {KEY_FILE}")
    
    # Xác nhận
    print("\n⚠️  CẢNH BÁO:")
    print("   - Dữ liệu sẽ được giải mã về thư mục data/")
    print("   - Các file hiện có có thể bị ghi đè")
    
    confirm = input("\n❓ Bạn có chắc muốn giải mã? (yes/no): ")
    
    if confirm.lower() != "yes":
        print("❌ Hủy thao tác giải mã")
        return
    
    # Giải mã
    success = decrypt_all()
    
    if success:
        print("\n✅ GIẢI MÃ THÀNH CÔNG!")
        print(f"📁 Dữ liệu đã được khôi phục tại: {DATA_DIR}")
    else:
        print("\n❌ GIẢI MÃ THẤT BẠI!")
        print("Vui lòng kiểm tra lại file khóa và dữ liệu mã hóa")

if __name__ == "__main__":
    main()
