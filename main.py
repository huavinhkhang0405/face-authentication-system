import sys
import os

# Thêm đường dẫn project vào PYTHONPATH
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(BASE_DIR, "src")
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

def print_header():
    """In tiêu đề menu"""
    print("\n" + "=" * 70)
    print(" " * 15 + "HỆ THỐNG NHẬN DIỆN KHUÔN MẶT")
    print(" " * 10 + "Face Recognition System with Liveness Detection")
    print("=" * 70)

def print_menu():
    """In menu chính"""
    print("\n" + "-" * 70)
    print(" " * 25 + "MENU CHÍNH")
    print("-" * 70)
    print("\n📊 THU THẬP DỮ LIỆU:")
    print("  1. Thu thập ảnh từ camera")
    print("  2. Cắt và chuẩn hóa ảnh (Detect & Align)")
    print("  3. Tạo embeddings từ ảnh đã chuẩn hóa")
    
    print("\n🤖 HUẤN LUYỆN MÔ HÌNH:")
    print("  4. Tìm tham số tối ưu (Hyperparameter Search)")
    print("  5. Train mô hình KNN với tham số tối ưu")
    
    print("\n👁️  NHẬN DIỆN:")
    print("  6. Nhận diện realtime (với Liveness Detection)")
    
    print("\n❌  Thoát:")
    print("  0. Thoát chương trình")
    print("-" * 70)

def option_1_capture():
    """Thu thập ảnh từ camera"""
    print("\n" + "=" * 70)
    print("📸 THU THẬP ẢNH TỪ CAMERA")
    print("=" * 70)
    try:
        from src.data.data_capture import capture_images
        capture_images()
        print("\n✅ Hoàn thành thu thập ảnh!")
    except Exception as e:
        print(f"\n❌ Lỗi: {e}")
        import traceback
        traceback.print_exc()

def option_2_align():
    """Cắt và chuẩn hóa ảnh"""
    print("\n" + "=" * 70)
    print("✂️  CẮT VÀ CHUẨN HÓA ẢNH")
    print("=" * 70)
    
    # Kiểm tra thư mục raw hoặc encrypted_data/raw
    from src.utils.config import RAW_DIR, DATA_DIR
    encrypted_raw = os.path.join(DATA_DIR, "encrypted_data", "raw")
    
    has_raw_data = os.path.exists(RAW_DIR) and len(os.listdir(RAW_DIR)) > 0
    has_encrypted_data = os.path.exists(encrypted_raw) and len(os.listdir(encrypted_raw)) > 0
    
    if not has_raw_data and not has_encrypted_data:
        print("\n⚠️  Chưa có ảnh trong thư mục raw hoặc encrypted_data/raw!")
        print("   Vui lòng chạy chức năng '1. Thu thập ảnh' trước.")
        input("\nNhấn Enter để tiếp tục...")
        return
    
    if has_encrypted_data:
        print(f"\n📁 Phát hiện {len(os.listdir(encrypted_raw))} folder trong encrypted_data/raw")
    
    try:
        from src.data.detect_align import align_faces
        align_faces()
        print("\n✅ Hoàn thành chuẩn hóa ảnh!")
    except Exception as e:
        print(f"\n❌ Lỗi: {e}")
        import traceback
        traceback.print_exc()

def option_3_embed():
    """Tạo embeddings"""
    print("\n" + "=" * 70)
    print("🔹 TẠO EMBEDDINGS")
    print("=" * 70)
    
    # Kiểm tra thư mục aligned hoặc encrypted_data/faces_aligned
    from src.utils.config import ALIGNED_DIR, DATA_DIR
    encrypted_aligned = os.path.join(DATA_DIR, "encrypted_data", "faces_aligned")
    
    has_aligned_data = os.path.exists(ALIGNED_DIR) and len(os.listdir(ALIGNED_DIR)) > 0
    has_encrypted_aligned = os.path.exists(encrypted_aligned) and len(os.listdir(encrypted_aligned)) > 0
    
    if not has_aligned_data and not has_encrypted_aligned:
        print("\n⚠️  Chưa có ảnh đã chuẩn hóa!")
        print("   Vui lòng chạy chức năng '2. Cắt và chuẩn hóa ảnh' trước.")
        input("\nNhấn Enter để tiếp tục...")
        return
    
    if has_encrypted_aligned:
        print(f"\n📁 Phát hiện {len(os.listdir(encrypted_aligned))} folder trong encrypted_data/faces_aligned")
        print("🔓 Hệ thống sẽ tự động giải mã để tạo embeddings...")
    
    try:
        from src.model.embedder import FaceEmbedder
        embedder = FaceEmbedder()
        embedder.build_embeddings()
        print("\n✅ Hoàn thành tạo embeddings!")
    except Exception as e:
        print(f"\n❌ Lỗi: {e}")
        import traceback
        traceback.print_exc()

def option_4_find_hyperparams():
    """Tìm tham số tối ưu"""
    print("\n" + "=" * 70)
    print("🔍 TÌM THAM SỐ TỐI ƯU (COA Algorithm)")
    print("=" * 70)
    
    # Kiểm tra embeddings
    from src.utils.config import DATA_DIR
    embeddings_file = os.path.join(DATA_DIR, "embeddings.npy")
    if not os.path.exists(embeddings_file):
        print("\n⚠️  Chưa có embeddings!")
        print("   Vui lòng chạy các bước: Thu thập ảnh (1) → Cắt ảnh (2) → Tạo embeddings (3)")
        input("\nNhấn Enter để tiếp tục...")
        return
    
    print("⚠️  Quá trình này có thể mất nhiều thời gian...")
    try:
        # Chạy script find_hyperparams.py bằng subprocess
        import subprocess
        script_path = os.path.join(SRC_DIR, "model", "find_hyperparams.py")
        result = subprocess.run([sys.executable, script_path], 
                              capture_output=False, 
                              text=True,
                              cwd=BASE_DIR)
        if result.returncode == 0:
            print("\n✅ Hoàn thành tìm tham số tối ưu!")
        else:
            print("\n❌ Có lỗi xảy ra trong quá trình tìm tham số!")
    except Exception as e:
        print(f"\n❌ Lỗi: {e}")
        import traceback
        traceback.print_exc()

def option_5_train():
    """Train mô hình KNN"""
    print("\n" + "=" * 70)
    print("🎯 HUẤN LUYỆN MÔ HÌNH KNN")
    print("=" * 70)
    
    # Kiểm tra embeddings và tham số
    from src.utils.config import DATA_DIR, RESULT_DIR
    embeddings_file = os.path.join(DATA_DIR, "embeddings.npy")
    params_file = os.path.join(RESULT_DIR, "best_params_faceid.txt")
    
    if not os.path.exists(embeddings_file):
        print("\n⚠️  Chưa có embeddings!")
        print("   Vui lòng chạy các bước: Thu thập ảnh (1) → Cắt ảnh (2) → Tạo embeddings (3)")
        input("\nNhấn Enter để tiếp tục...")
        return
    
    if not os.path.exists(params_file):
        print("\n⚠️  Chưa có tham số tối ưu!")
        print("   Vui lòng chạy chức năng '4. Tìm tham số tối ưu' trước.")
        input("\nNhấn Enter để tiếp tục...")
        return
    
    try:
        # Chạy script train.py bằng subprocess
        import subprocess
        script_path = os.path.join(SRC_DIR, "train.py")
        result = subprocess.run([sys.executable, script_path], 
                              capture_output=False, 
                              text=True,
                              cwd=BASE_DIR)
        if result.returncode == 0:
            print("\n✅ Hoàn thành huấn luyện mô hình!")
        else:
            print("\n❌ Có lỗi xảy ra trong quá trình train!")
    except Exception as e:
        print(f"\n❌ Lỗi: {e}")
        import traceback
        traceback.print_exc()

def option_6_recognize():
    """Nhận diện realtime"""
    print("\n" + "=" * 70)
    print("👁️  NHẬN DIỆN REALTIME VỚI LIVENESS DETECTION")
    print("=" * 70)
    print("📋 Hướng dẫn:")
    print("  - Hệ thống sẽ yêu cầu bạn quay đầu sang trái để xác thực liveness")
    print("  - Sau khi xác thực thành công, hệ thống sẽ nhận diện khuôn mặt")
    print("  - Nhấn 'Q' để thoát")
    print("\n🚀 Bắt đầu...")
    try:
        from src.model.infer_realtime import recognize_realtime
        recognize_realtime()
        print("\n✅ Đã kết thúc nhận diện!")
    except Exception as e:
        print(f"\n❌ Lỗi: {e}")
        import traceback
        traceback.print_exc()

def main():
    """Hàm main chính"""
    while True:
        print_header()
        print_menu()
        
        try:
            choice = input("\n👉 Chọn chức năng (0-6): ").strip()
            
            if choice == "0":
                print("\n👋 Cảm ơn bạn đã sử dụng hệ thống! Tạm biệt!")
                break
            elif choice == "1":
                option_1_capture()
                input("\nNhấn Enter để quay lại menu...")
            elif choice == "2":
                option_2_align()
                input("\nNhấn Enter để quay lại menu...")
            elif choice == "3":
                option_3_embed()
                input("\nNhấn Enter để quay lại menu...")
            elif choice == "4":
                confirm = input("\n⚠️  Quá trình này có thể mất nhiều thời gian. Tiếp tục? (y/n): ")
                if confirm.lower() == 'y':
                    option_4_find_hyperparams()
                    input("\nNhấn Enter để quay lại menu...")
                else:
                    print("❌ Đã hủy.")
            elif choice == "5":
                # Kiểm tra xem đã có tham số tối ưu chưa
                from src.utils.config import RESULT_DIR
                params_file = os.path.join(RESULT_DIR, "best_params_faceid.txt")
                if not os.path.exists(params_file):
                    print("\n⚠️  Chưa có tham số tối ưu!")
                    print("   Vui lòng chạy chức năng '4. Tìm tham số tối ưu' trước.")
                    input("\nNhấn Enter để tiếp tục...")
                    continue
                option_5_train()
                input("\nNhấn Enter để quay lại menu...")
            elif choice == "6":
                # Kiểm tra xem đã có mô hình chưa
                from src.utils.config import MODEL_DIR
                model_file = os.path.join(MODEL_DIR, "best_knn_faceid.pkl")
                if not os.path.exists(model_file):
                    print("\n⚠️  Chưa có mô hình đã train!")
                    print("   Vui lòng chạy các bước:")
                    print("   1. Thu thập dữ liệu (1, 2, 3)")
                    print("   2. Train mô hình (4, 5)")
                    input("\nNhấn Enter để tiếp tục...")
                    continue
                option_6_recognize()
                input("\nNhấn Enter để quay lại menu...")
            else:
                print("\n❌ Lựa chọn không hợp lệ! Vui lòng chọn từ 0-6.")
                input("\nNhấn Enter để tiếp tục...")
                
        except KeyboardInterrupt:
            print("\n\n⚠️  Đã hủy bởi người dùng (Ctrl+C)")
            break
        except Exception as e:
            print(f"\n❌ Lỗi không mong đợi: {e}")
            import traceback
            traceback.print_exc()
            input("\nNhấn Enter để tiếp tục...")

if __name__ == "__main__":
    main()

