import sys
import os
import cv2
import numpy as np
import base64
import joblib
import ast
import time
import csv
import glob
from datetime import datetime
from flask import Flask, render_template, request, redirect, url_for, flash, session, jsonify, send_file
import json

# --- CẤU HÌNH ĐƯỜNG DẪN ĐỂ IMPORT SRC ---
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE_DIR = ROOT_DIR
SRC_DIR = os.path.join(ROOT_DIR, "src")
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

# Import các module AI của bạn
from src.model.detector import FaceDetector
from src.model.embedder import FaceEmbedder
from src.model.liveness import calculate_yaw_from_frame
from src.utils.config import MODEL_DIR, RESULT_DIR, RAW_DIR, DATA_DIR, KEY_FILE
from cryptography.fernet import Fernet

app = Flask(__name__)
app.secret_key = 'super_secret_key'

# Đường dẫn đến folder encrypted_data
ENCRYPTED_RAW_DIR = os.path.join(DATA_DIR, "encrypted_data", "raw")

def count_encrypted_images(student_folder_name):
    """Đếm số ảnh đã mã hóa trong encrypted_data/raw/student_folder"""
    encrypted_folder = os.path.join(ENCRYPTED_RAW_DIR, student_folder_name)
    if not os.path.exists(encrypted_folder):
        return 0
    
    # Đếm các file .enc (ảnh đã mã hóa)
    encrypted_files = [f for f in os.listdir(encrypted_folder) 
                      if f.endswith('.enc')]
    return len(encrypted_files)

def get_student_folder_path(mssv, check_encrypted=True):
    """Tìm và trả về đường dẫn folder của sinh viên (ưu tiên raw, fallback encrypted)
    
    Args:
        mssv: Mã số sinh viên
        check_encrypted: Có kiểm tra trong encrypted_data không
    
    Returns:
        tuple: (folder_path, is_encrypted, folder_name)
    """
    # Kiểm tra trong RAW_DIR trước
    if os.path.exists(RAW_DIR):
        student_folders = [d for d in os.listdir(RAW_DIR) 
                          if os.path.isdir(os.path.join(RAW_DIR, d)) and mssv in d]
        if student_folders:
            folder_name = student_folders[0]
            return os.path.join(RAW_DIR, folder_name), False, folder_name
    
    # Nếu không tìm thấy và check_encrypted=True, kiểm tra trong encrypted_data
    if check_encrypted and os.path.exists(ENCRYPTED_RAW_DIR):
        student_folders = [d for d in os.listdir(ENCRYPTED_RAW_DIR) 
                          if os.path.isdir(os.path.join(ENCRYPTED_RAW_DIR, d)) and mssv in d]
        if student_folders:
            folder_name = student_folders[0]
            return os.path.join(ENCRYPTED_RAW_DIR, folder_name), True, folder_name
    
    return None, False, None

def encrypt_and_save_image(img, mssv, name_no_accent, index):
    """Mã hóa và lưu ảnh vào encrypted_data/raw
    
    Args:
        img: OpenCV image
        mssv: Mã số sinh viên
        name_no_accent: Tên không dấu
        index: Số thứ tự ảnh
    
    Returns:
        bool: True nếu thành công
    """
    try:
        # Load hoặc tạo khóa mã hóa
        if os.path.exists(KEY_FILE):
            key = open(KEY_FILE, "rb").read()
        else:
            key = Fernet.generate_key()
            with open(KEY_FILE, "wb") as f:
                f.write(key)
            print(f"🔑 Đã tạo khóa mới: {KEY_FILE}")
        
        cipher = Fernet(key)
        
        # Tạo folder cho sinh viên trong encrypted_data
        folder_name = f"{mssv}_{name_no_accent}"
        encrypted_folder = os.path.join(ENCRYPTED_RAW_DIR, folder_name)
        os.makedirs(encrypted_folder, exist_ok=True)
        
        # Encode ảnh thành bytes
        _, img_encoded = cv2.imencode('.jpg', img)
        img_bytes = img_encoded.tobytes()
        
        # Mã hóa
        encrypted_data = cipher.encrypt(img_bytes)
        
        # Lưu file mã hóa
        filename = f"{mssv}_{name_no_accent}_{index:03d}.jpg.enc"
        encrypted_path = os.path.join(encrypted_folder, filename)
        
        with open(encrypted_path, "wb") as f:
            f.write(encrypted_data)
        
        return True
    except Exception as e:
        print(f"❌ Lỗi mã hóa ảnh: {e}")
        return False

# --- 1. KHỞI TẠO MÔ HÌNH (LOAD 1 LẦN) ---
print("⏳ Đang tải mô hình AI...")
detector = FaceDetector()
embedder = FaceEmbedder()

# Load Model KNN & Label Encoder
knn_path = os.path.join(MODEL_DIR, "best_knn_faceid.pkl")
le_path = os.path.join(MODEL_DIR, "label_encoder.pkl")
params_path = os.path.join(RESULT_DIR, "best_params_faceid.txt")

if os.path.exists(knn_path) and os.path.exists(le_path):
    knn = joblib.load(knn_path)
    le = joblib.load(le_path)
    
    # Load tham số metric để set ngưỡng
    with open(params_path, "r") as f:
        params = ast.literal_eval(f.readline().split("},")[0] + "}")
    metric = params.get("metric", "euclidean")
    THRESHOLD = 0.55 if metric == "euclidean" else 0.4
    print("✅ Đã tải mô hình thành công!")
else:
    print("⚠️ Cảnh báo: Chưa tìm thấy file model. Vui lòng train trước!")
    knn = None

# --- 2. QUẢN LÝ TRẠNG THÁI LIVENESS (SESSION STATE) ---
# Vì HTTP stateless, ta cần lưu trạng thái quay đầu của từng Client (theo IP)
client_states = {}

class ClientState:
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.step = "detect" # detect -> liveness -> recognized
        self.initial_yaw = None
        self.start_time = None
        self.detected_name = None
        self.last_active = time.time()

# --- 3. DATABASE GIẢ LẬP & LOGIN ---
users_db = {
    "admin@gmail.com": {"password": "admin123", "name": "Quản trị viên", "role": "Admin"},
    "teacher@gmail.com": {"password": "123456", "name": "Trần Như Ý", "role": "Giảng viên"}
}

@app.route('/', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        if email in users_db and users_db[email]["password"] == password:
            session['logged_in'] = True
            session['email'] = email
            session['name'] = users_db[email]["name"]
            session['role'] = users_db[email]["role"]
            return redirect(url_for('dashboard'))
        else:
            flash('Email hoặc mật khẩu sai.', 'error')
    return render_template('login.html')

@app.route('/dashboard')
def dashboard():
    if not session.get('logged_in'):
        return redirect(url_for('login'))
    return render_template('dashboard.html', user=session, active_page='dashboard')

@app.route('/class')
def class_page():
    if not session.get('logged_in'):
        return redirect(url_for('login'))
    return render_template('class.html', user=session, active_page='class')

@app.route('/students')
def students_page():
    if not session.get('logged_in'):
        return redirect(url_for('login'))
    
    # Load danh sách sinh viên từ JSON
    import json
    import math
    
    student_file = os.path.join(BASE_DIR, 'student_list.json')
    all_students = []
    
    print(f"🔍 Đường dẫn file: {student_file}")
    print(f"📁 File tồn tại: {os.path.exists(student_file)}")
    
    try:
        if os.path.exists(student_file):
            with open(student_file, 'r', encoding='utf-8') as f:
                all_students = json.load(f)
            print(f"✅ Đã load {len(all_students)} sinh viên")
        else:
            print("❌ Không tìm thấy file student_list.json")
    except Exception as e:
        print(f"❌ Lỗi đọc file: {e}")
        all_students = []
    
    # Pagination
    page = request.args.get('page', 1, type=int)
    per_page = 20
    total = len(all_students)
    total_pages = math.ceil(total / per_page) if total > 0 else 1
    
    # Slice students for current page
    start = (page - 1) * per_page
    end = start + per_page
    students = all_students[start:end]
    
    # Thống kê dữ liệu khuôn mặt cho TẤT CẢ sinh viên
    stats = {
        'has_sufficient': 0,  # >=30 ảnh
        'has_insufficient': 0,  # <30 ảnh
        'has_none': 0,  # không có ảnh
        'male_count': 0,
        'female_count': 0
    }
    
    # Kiểm tra dữ liệu khuôn mặt cho mỗi sinh viên trên trang hiện tại
    for student in students:
        mssv = student['mssv']
        # Tìm folder có chứa MSSV trong tên (format: MSSV_TenSV)
        folder_path, is_encrypted, folder_name = get_student_folder_path(mssv, check_encrypted=True)
        
        if folder_path:
            # Đếm số ảnh trong folder
            if is_encrypted:
                # Đếm file .enc trong encrypted_data
                image_count = count_encrypted_images(folder_name)
            else:
                # Đếm file ảnh thông thường
                image_files = [f for f in os.listdir(folder_path) 
                              if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                image_count = len(image_files)
            
            student['has_images'] = True
            student['image_count'] = image_count
            student['is_sufficient'] = image_count >= 30
        else:
            student['has_images'] = False
            student['image_count'] = 0
            student['is_sufficient'] = False
    
    # Thống kê cho TẤT CẢ sinh viên (không chỉ trang hiện tại)
    for student in all_students:
        mssv = student['mssv']
        folder_path, is_encrypted, folder_name = get_student_folder_path(mssv, check_encrypted=True)
        
        if folder_path:
            if is_encrypted:
                image_count = count_encrypted_images(folder_name)
            else:
                image_files = [f for f in os.listdir(folder_path) 
                              if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                image_count = len(image_files)
            
            if image_count >= 30:
                stats['has_sufficient'] += 1
            else:
                stats['has_insufficient'] += 1
        else:
            stats['has_none'] += 1
        
        # Đếm giới tính
        if student.get('gender') == 'Nam':
            stats['male_count'] += 1
        elif student.get('gender') == 'Nữ':
            stats['female_count'] += 1
    
    # Tính phần trăm
    stats['with_images'] = stats['has_sufficient'] + stats['has_insufficient']
    stats['completion_rate'] = round((stats['has_sufficient'] / total * 100), 1) if total > 0 else 0
    stats['missing_count'] = stats['has_none']
    
    print(f"📊 Thống kê: Đủ={stats['has_sufficient']}, Thiếu={stats['has_insufficient']}, Không có={stats['has_none']}")
    print(f"📄 Trang {page}/{total_pages} - Hiển thị {len(students)} sinh viên (từ {start+1} đến {end})")
    
    return render_template('students.html', 
                         user=session, 
                         active_page='students', 
                         students=students,
                         page=page,
                         total_pages=total_pages,
                         total=total,
                         start=start,
                         end=min(end, total),
                         stats=stats)

@app.route('/add-student')
def add_student_page():
    if not session.get('logged_in'):
        return redirect(url_for('login'))
    
    return render_template('add_student.html', 
                         user=session, 
                         active_page='students')

@app.route('/update-images/<mssv>')
def update_images_page(mssv):
    if not session.get('logged_in'):
        return redirect(url_for('login'))
    
    # 1. Load thông tin sinh viên
    student_file = os.path.join(BASE_DIR, 'student_list.json')
    student_info = None
    
    if os.path.exists(student_file):
        try:
            with open(student_file, 'r', encoding='utf-8') as f:
                students = json.load(f)
                # Tìm sinh viên theo MSSV
                for s in students:
                    if s.get('mssv') == mssv:
                        student_info = s
                        break
        except Exception as e:
            print(f"Lỗi đọc file JSON: {e}")
    
    if not student_info:
        flash('Không tìm thấy sinh viên!', 'error')
        return redirect(url_for('students_page'))
    
    # 2. Kiểm tra số lượng ảnh hiện có
    folder_path, is_encrypted, folder_name = get_student_folder_path(mssv, check_encrypted=True)
    
    current_count = 0
    if folder_path:
        # Đếm số ảnh
        if is_encrypted:
            current_count = count_encrypted_images(folder_name)
        else:
            image_files = [f for f in os.listdir(folder_path) 
                          if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            current_count = len(image_files)
    
    # 3. Render template và truyền biến user
    return render_template('update_images.html', 
                           user=session,             # [QUAN TRỌNG] Biến này sửa lỗi 'user' is undefined
                           active_page='students',
                           student=student_info,
                           current_count=current_count)

@app.route('/api/update-images', methods=['POST'])
def api_update_images():
    """API để cập nhật thêm ảnh cho sinh viên đã có"""
    if not session.get('logged_in'):
        return jsonify({'success': False, 'message': 'Chưa đăng nhập'}), 401
    
    try:
        data = request.get_json()
        mssv = data.get('mssv')
        name = data.get('name')
        images = data.get('images', [])
        
        if not mssv or not name or not images:
            return jsonify({'success': False, 'message': 'Thiếu thông tin'}), 400
        
        # Chuyển tên sang không dấu
        name_no_accent = remove_accents(name)
        folder_name = f"{mssv}_{name_no_accent}"
        
        # Đếm số ảnh hiện có trong encrypted_data
        current_count = count_encrypted_images(folder_name)
        start_index = current_count + 1
        
        # Mã hóa và lưu từng ảnh mới
        saved_count = 0
        for idx, img_data in enumerate(images):
            if 'base64,' in img_data:
                img_data = img_data.split('base64,')[1]
            
            img_bytes = base64.b64decode(img_data)
            img_array = np.frombuffer(img_bytes, dtype=np.uint8)
            img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            
            if img is not None:
                # Mã hóa và lưu vào encrypted_data
                if encrypt_and_save_image(img, mssv, name_no_accent, start_index + idx):
                    saved_count += 1
        
        total_count = current_count + saved_count
        
        return jsonify({
            'success': True, 
            'message': f'Đã thêm {saved_count} ảnh. Tổng: {total_count} ảnh',
            'total': total_count
        })
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'message': str(e)}), 500

def remove_accents(text):
    """Chuyển văn bản tiếng Việt có dấu thành không dấu"""
    import unicodedata
    # Chuẩn hóa Unicode về dạng NFD (tách ký tự và dấu)
    nfd = unicodedata.normalize('NFD', text)
    # Loại bỏ các ký tự dấu (Mn = Mark, Nonspacing)
    without_accents = ''.join(c for c in nfd if unicodedata.category(c) != 'Mn')
    # Xử lý các ký tự đặc biệt còn lại
    replacements = {
        'đ': 'd', 'Đ': 'D',
        ' ': '', '-': '', '_': ''
    }
    for old, new in replacements.items():
        without_accents = without_accents.replace(old, new)
    return without_accents

@app.route('/api/capture-images', methods=['POST'])
def api_capture_images():
    """API để lưu ảnh đã chụp từ camera"""
    if not session.get('logged_in'):
        return jsonify({'success': False, 'message': 'Chưa đăng nhập'}), 401
    
    try:
        data = request.get_json()
        mssv = data.get('mssv')
        name = data.get('name')
        images = data.get('images', [])  # Danh sách base64 images
        
        if not mssv or not name or not images:
            return jsonify({'success': False, 'message': 'Thiếu thông tin'}), 400
        
        # Chuyển tên sang không dấu và viết liền
        name_no_accent = remove_accents(name)
        folder_name = f"{mssv}_{name_no_accent}"
        
        # Mã hóa và lưu từng ảnh với format: MSSV_HoTenKhongDau_STT.jpg.enc
        saved_count = 0
        for idx, img_data in enumerate(images):
            # Remove base64 header
            if 'base64,' in img_data:
                img_data = img_data.split('base64,')[1]
            
            # Decode
            img_bytes = base64.b64decode(img_data)
            img_array = np.frombuffer(img_bytes, dtype=np.uint8)
            img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            
            if img is not None:
                # Mã hóa và lưu vào encrypted_data
                if encrypt_and_save_image(img, mssv, name_no_accent, idx + 1):
                    saved_count += 1
        
        return jsonify({
            'success': True, 
            'message': f'Đã lưu {saved_count} ảnh vào {folder_name}',
            'folder': folder_name
        })
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'message': str(e)}), 500

@app.route('/api/align-faces', methods=['POST'])
def api_align_faces():
    """API để cắt và chuẩn hóa ảnh"""
    if not session.get('logged_in'):
        return jsonify({'success': False, 'message': 'Chưa đăng nhập'}), 401
    
    try:
        data = request.get_json()
        mssv = data.get('mssv')
        
        if not mssv:
            return jsonify({'success': False, 'message': 'Thiếu MSSV'}), 400
        
        # Import và chạy detect_align cho sinh viên cụ thể
        from src.data.detect_align import align_faces_for_student
        
        # Tìm folder của sinh viên (hỗ trợ cả encrypted)
        folder_path, is_encrypted, folder_name = get_student_folder_path(mssv, check_encrypted=True)
        
        if not folder_path:
            return jsonify({'success': False, 'message': 'Không tìm thấy folder ảnh'}), 404
        
        # Chạy align với folder_name
        result = align_faces_for_student(folder_name)
        
        return jsonify({
            'success': True,
            'message': 'Đã chuẩn hóa ảnh thành công',
            'aligned_count': result
        })
    
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500

@app.route('/api/create-embeddings', methods=['POST'])
def api_create_embeddings():
    """API để tạo embeddings và tự động mã hóa ảnh"""
    if not session.get('logged_in'):
        return jsonify({'success': False, 'message': 'Chưa đăng nhập'}), 401
    
    try:
        # Rebuild toàn bộ embeddings (bao gồm sinh viên mới)
        embedder.build_embeddings()
        
        # Tự động mã hóa sau khi embedding (đã tích hợp trong embedder.build_embeddings)
        # Không cần gọi riêng nữa vì đã có trong embedder.py
        
        return jsonify({
            'success': True,
            'message': 'Đã tạo embeddings và mã hóa dữ liệu thành công'
        })
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'message': str(e)}), 500

@app.route('/api/train-model', methods=['POST'])
def api_train_model():
    """API để train lại mô hình"""
    if not session.get('logged_in'):
        return jsonify({'success': False, 'message': 'Chưa đăng nhập'}), 401
    
    try:
        # Chạy train.py
        import subprocess
        script_path = os.path.join(SRC_DIR, "train.py")
        result = subprocess.run([sys.executable, script_path], 
                              capture_output=True, 
                              text=True,
                              cwd=BASE_DIR)
        
        if result.returncode == 0:
            # ✅ Reload model mới vào memory
            global knn, le, THRESHOLD
            
            knn_path = os.path.join(MODEL_DIR, "best_knn_faceid.pkl")
            le_path = os.path.join(MODEL_DIR, "label_encoder.pkl")
            params_path = os.path.join(RESULT_DIR, "best_params_faceid.txt")
            
            if os.path.exists(knn_path) and os.path.exists(le_path):
                knn = joblib.load(knn_path)
                le = joblib.load(le_path)
                
                # Reload threshold
                with open(params_path, "r") as f:
                    params = ast.literal_eval(f.readline().split("},")[0] + "}")
                metric = params.get("metric", "euclidean")
                THRESHOLD = 0.55 if metric == "euclidean" else 0.4
                
                print("✅ Đã reload model mới vào memory!")
            
            return jsonify({
                'success': True,
                'message': 'Đã train và reload mô hình thành công'
            })
        else:
            return jsonify({
                'success': False,
                'message': f'Lỗi train: {result.stderr}'
            }), 500
    
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500

@app.route('/api/save-student', methods=['POST'])
def api_save_student():
    """API để lưu thông tin sinh viên vào JSON"""
    if not session.get('logged_in'):
        return jsonify({'success': False, 'message': 'Chưa đăng nhập'}), 401
    
    try:
        import json
        data = request.get_json()
        mssv = data.get('mssv')
        name = data.get('name')
        gender = data.get('gender')
        
        if not all([mssv, name, gender]):
            return jsonify({'success': False, 'message': 'Thiếu thông tin'}), 400
        
        # Load student_list.json
        student_file = os.path.join(BASE_DIR, 'student_list.json')
        students = []
        
        if os.path.exists(student_file):
            with open(student_file, 'r', encoding='utf-8') as f:
                students = json.load(f)
        
        # Kiểm tra trùng MSSV
        if any(s['mssv'] == mssv for s in students):
            return jsonify({'success': False, 'message': 'MSSV đã tồn tại'}), 400
        
        # Thêm sinh viên mới
        students.append({
            'mssv': mssv,
            'name': name,
            'gender': gender
        })
        
        # Lưu lại file
        with open(student_file, 'w', encoding='utf-8') as f:
            json.dump(students, f, ensure_ascii=False, indent=2)
        
        return jsonify({
            'success': True,
            'message': 'Đã lưu thông tin sinh viên'
        })
    
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500

@app.route('/attendance')
def attendance_page():
    if not session.get('logged_in'):
        return redirect(url_for('login'))
    
    try:
        # Load danh sách sinh viên
        import json
        student_file = os.path.join(BASE_DIR, 'student_list.json')
        students = []
        
        if os.path.exists(student_file):
            try:
                with open(student_file, 'r', encoding='utf-8') as f:
                    students = json.load(f)
            except Exception as e:
                print(f"Lỗi đọc file JSON: {e}")
                flash(f"Không thể tải danh sách sinh viên: {e}", "error")
        
        # Load dữ liệu điểm danh từ CSV của ngày hôm nay
        date_str = datetime.now().strftime("%Y-%m-%d")
        csv_filename = f"attendance_{date_str}.csv"
        attended_records = {}  # {mssv: time}
        
        if os.path.exists(csv_filename):
            try:
                with open(csv_filename, 'r', encoding='utf-8') as f:
                    csv_reader = csv.reader(f)
                    next(csv_reader, None)  # Skip header
                    for row in csv_reader:
                        if len(row) >= 3:
                            mssv = row[0]
                            time = row[2]
                            attended_records[mssv] = time
                print(f"✅ Đã load {len(attended_records)} bản ghi điểm danh từ {csv_filename}")
            except Exception as e:
                print(f"⚠️ Lỗi đọc file CSV điểm danh: {e}")
        
        # Đánh dấu sinh viên đã điểm danh
        for student in students:
            if student['mssv'] in attended_records:
                student['attended'] = True
                student['time'] = attended_records[student['mssv']]
            else:
                student['attended'] = False
                student['time'] = None
        
        return render_template('attendance.html', 
                               user=session, 
                               students=students,
                               attended_records=attended_records,
                               active_page='attendance')
    except Exception as e:
        print(f"Lỗi tải trang attendance: {e}")
        flash(f"Lỗi tải trang: {e}", "error")
        return redirect(url_for('dashboard'))

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))

@app.route('/test-students')
def test_students():
    """Route test để kiểm tra load dữ liệu"""
    import json
    student_file = os.path.join(BASE_DIR, 'student_list.json')
    
    if os.path.exists(student_file):
        with open(student_file, 'r', encoding='utf-8') as f:
            students = json.load(f)
        return jsonify({
            "success": True,
            "total": len(students),
            "first_5": students[:5]
        })
    else:
        return jsonify({
            "success": False,
            "error": "File không tồn tại",
            "path": student_file
        })

# --- Route Xuất Báo Cáo ---
@app.route('/export_report')
def export_report():
    if not session.get('logged_in'):
        return redirect(url_for('login'))
    
    try:
        # Tìm file CSV mới nhất
        csv_files = glob.glob("attendance_*.csv")
        
        if not csv_files:
            flash('Chưa có dữ liệu điểm danh nào!', 'error')
            return redirect(url_for('dashboard'))
        
        # Lấy file mới nhất
        latest_file = max(csv_files, key=os.path.getctime)
        
        # Gửi file để download
        return send_file(
            latest_file,
            mimetype='text/csv',
            as_attachment=True,
            download_name=f'BaoCaoDiemDanh_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
        )
    except Exception as e:
        print(f"Lỗi xuất báo cáo: {e}")
        flash('Lỗi khi xuất báo cáo!', 'error')
        return redirect(url_for('dashboard'))

# --- 4. HÀM GHI LOG CSV (ĐIỂM DANH) ---
def get_attendance_status(check_time):
    """Xác định trạng thái điểm danh dựa trên thời gian
    
    Args:
        check_time: datetime object của thời gian điểm danh
    
    Returns:
        str: 'Đã điểm danh' / 'Trễ' / 'Vắng'
    """
    hour = check_time.hour
    minute = check_time.minute
    
    # Chuyển thành số phút từ 0h
    current_minutes = hour * 60 + minute
    
    # 12:30 = 12*60 + 30 = 750 phút
    # 13:00 = 13*60 + 0 = 780 phút  
    # 14:00 = 14*60 + 0 = 840 phút

    if 750 <= current_minutes < 780:  # 12:30 - 13:00
        return "Đã điểm danh"
    elif 780 <= current_minutes < 840:  # 13:00 - 14:00
        return "Trễ"
    else:
        return "Vắng"

def log_attendance_csv(mssv, name):
    """Ghi log điểm danh với trạng thái dựa trên thời gian"""
    now = datetime.now()
    date_str = now.strftime("%Y-%m-%d")
    filename = f"attendance_{date_str}.csv"
    
    # Xác định trạng thái dựa trên thời gian
    status = get_attendance_status(now)
    time_str = now.strftime("%H:%M:%S")
    
    # Check trùng lặp
    already_checked = False
    if os.path.exists(filename):
        with open(filename, 'r', encoding='utf-8') as f:
            if mssv in f.read():
                already_checked = True
    
    if not already_checked:
        with open(filename, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            # Nếu file mới thì ghi header
            if os.stat(filename).st_size == 0:
                writer.writerow(["MSSV", "HoTen", "ThoiGian", "TrangThai"])
            writer.writerow([mssv, name, time_str, status])
        
        print(f"📝 Điểm danh: {mssv} - {name} - {time_str} - [{status}]")
        return True, status  # Ghi mới
    return False, None  # Đã có rồi

# --- 5. API XỬ LÝ AI (CORE LOGIC) ---
@app.route('/api/process_frame', methods=['POST'])
def process_frame():
    # Kiểm tra thời gian trước khi xử lý
    now = datetime.now()
    current_minutes = now.hour * 60 + now.minute
    
    # Nếu chưa tới 12:30 (750 phút), từ chối điểm danh
    if current_minutes < 750:
        return jsonify({
            "status": "too_early",
            "message": "⏰ Chưa tới giờ điểm danh! Vui lòng quay lại sau 12:30.",
            "instruction": "too_early",
            "current_time": now.strftime("%H:%M")
        })
    
    # Lấy IP client để quản lý session
    client_ip = request.remote_addr
    if client_ip not in client_states:
        client_states[client_ip] = ClientState()
    
    state = client_states[client_ip]
    # Reset nếu user không tương tác quá 30s
    if time.time() - state.last_active > 30:
        state.reset()
    state.last_active = time.time()

    try:
        # 1. Decode Ảnh Base64
        data = request.json['image']
        header, encoded = data.split(",", 1)
        nparr = np.frombuffer(base64.b64decode(encoded), np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        print(f"📸 Nhận frame: {frame.shape if frame is not None else 'NULL'} - Client: {client_ip} - Step: {state.step}")
        
        # 2. Detect Face
        faces, boxes = detector.detect_from_frame(frame)
        
        if not faces or len(faces) == 0:
            if state.step == "liveness": # Đang quay đầu mà mất mặt -> làm lại
                state.reset()
            print(f"⚠️ Không phát hiện khuôn mặt - Client: {client_ip}")
            return jsonify({"status": "waiting", "message": "Không thấy khuôn mặt. Hãy nhìn vào camera!", "instruction": "none"})

        # Lấy khuôn mặt to nhất
        face_img = faces[0]
        print(f"✅ Phát hiện {len(faces)} khuôn mặt - Client: {client_ip} - Step: {state.step}")

        # 3. QUY TRÌNH: CHECK LIVENESS -> RECOGNIZE
        
        # Giai đoạn A: Kiểm tra Liveness (Quay trái)
        if state.step == "detect" or state.step == "liveness":
            current_yaw, has_face = calculate_yaw_from_frame(frame)
            
            if state.step == "detect":
                # Bắt đầu ghi nhận góc ban đầu
                state.initial_yaw = current_yaw
                state.start_time = time.time()
                state.step = "liveness"
                return jsonify({
                    "status": "liveness", 
                    "message": "Đã bắt khuôn mặt", 
                    "instruction": "turn_left" # Frontend hiện mũi tên trái
                })
            
            elif state.step == "liveness":
                # Tính độ lệch góc
                deviation = abs(state.initial_yaw - current_yaw)
                YAW_THRESHOLD = 25.0
                
                # Check Timeout (10s)
                if time.time() - state.start_time > 10:
                    state.reset()
                    return jsonify({"status": "error", "message": "Hết giờ! Vui lòng thử lại.", "instruction": "retry"})

                if deviation > YAW_THRESHOLD:
                    # ✅ Liveness Passed
                    state.step = "recognizing"
                    # Không return vội, cho chạy xuống dưới để recognize luôn
                else:
                    progress = int((deviation / YAW_THRESHOLD) * 100)
                    return jsonify({
                        "status": "liveness", 
                        "message": f"Quay trái tiếp... ({progress}%)", 
                        "instruction": "turn_left"
                    })

        # Giai đoạn B: Nhận diện (Khi đã pass liveness)
        if state.step == "recognizing":
            if knn is None:
                return jsonify({"status": "error", "message": "Lỗi Model Server"})
            
            # Embed & Predict
            emb = embedder.embed_face(face_img).reshape(1, -1)
            dist = knn.kneighbors(emb, n_neighbors=1)[0][0][0]
            pred = knn.predict(emb)[0]
            name_label = le.inverse_transform([pred])[0]
            
            if dist > THRESHOLD:
                state.reset()
                return jsonify({
                    "status": "unauthorized", 
                    "message": "⚠️ Người lạ - Không được phép vào!", 
                    "instruction": "unauthorized",
                    "data": {"mssv": "UNKNOWN", "name": "Người lạ"}
                })
            else:
                parts = name_label.split('_')
                if len(parts) >= 2:
                    mssv = parts[0]
                    name_display = " ".join(parts[1:])
                else:
                    mssv = name_label
                    name_display = name_label

                is_new, attendance_status = log_attendance_csv(mssv, name_display)
                
                if is_new:
                    status_msg = f"✅ Điểm danh thành công! Trạng thái: {attendance_status}"
                else:
                    status_msg = "ℹ️ Đã điểm danh trước đó"
                
                state.reset()
                return jsonify({
                    "status": "success", 
                    "message": f"{status_msg} - {name_display}", 
                    "instruction": "success",
                    "data": {
                        "mssv": mssv, 
                        "name": name_display, 
                        "is_new": is_new,
                        "attendance_status": attendance_status if is_new else None
                    }
                })

    except Exception as e:
        print(e)
        return jsonify({"status": "error", "message": "Lỗi xử lý server"})

    return jsonify({"status": "idle", "message": "..."})

@app.route('/api/export-attendance', methods=['POST'])
def export_attendance():
    """API xuất báo cáo sinh viên đã điểm danh"""
    try:
        data = request.json
        students = data.get('students', [])
        timestamp = data.get('timestamp', datetime.now().isoformat())
        
        if not students:
            return jsonify({"success": False, "message": "Không có dữ liệu để xuất"}), 400
        
        # Tạo tên file với timestamp
        now = datetime.now()
        filename = f"DanhSachDiemDanh_{now.strftime('%Y%m%d_%H%M%S')}.csv"
        filepath = os.path.join(BASE_DIR, 'exports', filename)
        
        # Tạo thư mục exports nếu chưa có
        os.makedirs(os.path.join(BASE_DIR, 'exports'), exist_ok=True)
        
        # Ghi file CSV với encoding UTF-8 BOM để Excel hiển thị đúng tiếng Việt
        with open(filepath, 'w', newline='', encoding='utf-8-sig') as csvfile:
            fieldnames = ['STT', 'MSSV', 'Họ và tên', 'Giới tính', 'Thời gian điểm danh']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            writer.writeheader()
            for student in students:
                writer.writerow({
                    'STT': student.get('stt', ''),
                    'MSSV': student.get('mssv', ''),
                    'Họ và tên': student.get('name', ''),
                    'Giới tính': student.get('gender', ''),
                    'Thời gian điểm danh': student.get('time', '')
                })
        
        # Gửi file về client
        return send_file(filepath, 
                        as_attachment=True, 
                        download_name=filename,
                        mimetype='text/csv')
        
    except Exception as e:
        print(f"Lỗi xuất báo cáo: {e}")
        return jsonify({"success": False, "message": str(e)}), 500

@app.route('/api/generate-full-report', methods=['POST'])
def generate_full_report():
    """API tạo báo cáo đầy đủ với trạng thái tự động cho tất cả sinh viên"""
    try:
        # Load danh sách sinh viên từ student_list.json
        student_list_path = os.path.join(BASE_DIR, 'student_list.json')
        if not os.path.exists(student_list_path):
            return jsonify({"success": False, "message": "Không tìm thấy student_list.json"}), 404
        
        with open(student_list_path, 'r', encoding='utf-8') as f:
            all_students = json.load(f)
        
        # Load dữ liệu điểm danh hôm nay
        date_str = datetime.now().strftime("%Y-%m-%d")
        attendance_file = f"attendance_{date_str}.csv"
        
        # Dictionary để lưu thông tin điểm danh
        attendance_dict = {}
        
        if os.path.exists(attendance_file):
            with open(attendance_file, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    mssv = row.get('MSSV', '')
                    attendance_dict[mssv] = {
                        'time': row.get('ThoiGian', ''),
                        'status': row.get('TrangThai', 'Vắng')
                    }
        
        # Tạo báo cáo đầy đủ
        now = datetime.now()
        filename = f"BaoCaoDiemDanh_DayDu_{now.strftime('%Y%m%d_%H%M%S')}.csv"
        filepath = os.path.join(BASE_DIR, 'exports', filename)
        
        # Tạo thư mục exports nếu chưa có
        os.makedirs(os.path.join(BASE_DIR, 'exports'), exist_ok=True)
        
        # Ghi file CSV
        with open(filepath, 'w', newline='', encoding='utf-8-sig') as csvfile:
            fieldnames = ['STT', 'MSSV', 'Họ và tên', 'Giới tính', 'Thời gian điểm danh', 'Trạng thái']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            writer.writeheader()
            for idx, student in enumerate(all_students, 1):
                mssv = student.get('mssv', '')
                attendance_info = attendance_dict.get(mssv, {'time': '', 'status': 'Vắng'})
                
                writer.writerow({
                    'STT': idx,
                    'MSSV': mssv,
                    'Họ và tên': student.get('name', ''),
                    'Giới tính': student.get('gender', ''),
                    'Thời gian điểm danh': attendance_info['time'],
                    'Trạng thái': attendance_info['status']
                })
        
        # Thống kê
        total = len(all_students)
        attended = sum(1 for info in attendance_dict.values() if info['status'] == 'Đã điểm danh')
        late = sum(1 for info in attendance_dict.values() if info['status'] == 'Trễ')
        absent = total - attended - late
        
        print(f"📊 Báo cáo: Tổng {total} | Có mặt {attended} | Trễ {late} | Vắng {absent}")
        
        # Gửi file về client
        return send_file(filepath, 
                        as_attachment=True, 
                        download_name=filename,
                        mimetype='text/csv')
        
    except Exception as e:
        print(f"Lỗi tạo báo cáo đầy đủ: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"success": False, "message": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)