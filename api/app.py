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

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE_DIR = ROOT_DIR
SRC_DIR = os.path.join(ROOT_DIR, "src")
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.model.detector import FaceDetector
from src.model.embedder import FaceEmbedder
from src.model.liveness import calculate_yaw_from_frame
from src.utils.config import MODEL_DIR, RESULT_DIR, RAW_DIR, DATA_DIR, KEY_FILE
from cryptography.fernet import Fernet

app = Flask(__name__)
app.secret_key = 'super_secret_key'

ENCRYPTED_RAW_DIR = os.path.join(DATA_DIR, "encrypted_data", "raw")

def count_encrypted_images(student_folder_name):
    """Đếm số ảnh đã mã hóa trong encrypted_data/raw/student_folder"""
    encrypted_folder = os.path.join(ENCRYPTED_RAW_DIR, student_folder_name)
    if not os.path.exists(encrypted_folder):
        return 0

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
    if os.path.exists(RAW_DIR):
        student_folders = [d for d in os.listdir(RAW_DIR) 
                          if os.path.isdir(os.path.join(RAW_DIR, d)) and mssv in d]
        if student_folders:
            folder_name = student_folders[0]
            return os.path.join(RAW_DIR, folder_name), False, folder_name

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
        if os.path.exists(KEY_FILE):
            key = open(KEY_FILE, "rb").read()
        else:
            key = Fernet.generate_key()
            with open(KEY_FILE, "wb") as f:
                f.write(key)
            print(f"🔑 Đã tạo khóa mới: {KEY_FILE}")
        
        cipher = Fernet(key)

        folder_name = f"{mssv}_{name_no_accent}"
        encrypted_folder = os.path.join(ENCRYPTED_RAW_DIR, folder_name)
        os.makedirs(encrypted_folder, exist_ok=True)
        
        _, img_encoded = cv2.imencode('.jpg', img)
        img_bytes = img_encoded.tobytes()

        encrypted_data = cipher.encrypt(img_bytes)

        filename = f"{mssv}_{name_no_accent}_{index:03d}.jpg.enc"
        encrypted_path = os.path.join(encrypted_folder, filename)
        
        with open(encrypted_path, "wb") as f:
            f.write(encrypted_data)
        
        return True
    except Exception as e:
        print(f"❌ Lỗi mã hóa ảnh: {e}")
        return False

print("⏳ Đang tải mô hình AI...")
detector = FaceDetector()
embedder = FaceEmbedder()

knn_path = os.path.join(MODEL_DIR, "best_knn_faceid.pkl")
le_path = os.path.join(MODEL_DIR, "label_encoder.pkl")
params_path = os.path.join(RESULT_DIR, "best_params_faceid.txt")

if os.path.exists(knn_path) and os.path.exists(le_path):
    knn = joblib.load(knn_path)
    le = joblib.load(le_path)

    with open(params_path, "r") as f:
        params = ast.literal_eval(f.readline().split("},")[0] + "}")
    metric = params.get("metric", "euclidean")
    THRESHOLD = 0.55 if metric == "euclidean" else 0.4
    print("✅ Đã tải mô hình thành công!")
else:
    print("⚠️ Cảnh báo: Chưa tìm thấy file model. Vui lòng train trước!")
    knn = None

client_states = {}

class ClientState:
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.step = "detect"
        self.initial_yaw = None
        self.start_time = None
        self.detected_name = None
        self.last_active = time.time()
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

    page = request.args.get('page', 1, type=int)
    per_page = 20
    total = len(all_students)
    total_pages = math.ceil(total / per_page) if total > 0 else 1

    start = (page - 1) * per_page
    end = start + per_page
    students = all_students[start:end]

    stats = {
        'has_sufficient': 0,  
        'has_insufficient': 0,  
        'has_none': 0,  
        'male_count': 0,
        'female_count': 0
    }

    for student in students:
        mssv = student['mssv']
        folder_path, is_encrypted, folder_name = get_student_folder_path(mssv, check_encrypted=True)
        
        if folder_path:
            if is_encrypted:
                image_count = count_encrypted_images(folder_name)
            else:
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

        if student.get('gender') == 'Nam':
            stats['male_count'] += 1
        elif student.get('gender') == 'Nữ':
            stats['female_count'] += 1

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

    student_file = os.path.join(BASE_DIR, 'student_list.json')
    student_info = None
    
    if os.path.exists(student_file):
        try:
            with open(student_file, 'r', encoding='utf-8') as f:
                students = json.load(f)
                for s in students:
                    if s.get('mssv') == mssv:
                        student_info = s
                        break
        except Exception as e:
            print(f"Lỗi đọc file JSON: {e}")
    
    if not student_info:
        flash('Không tìm thấy sinh viên!', 'error')
        return redirect(url_for('students_page'))

    folder_path, is_encrypted, folder_name = get_student_folder_path(mssv, check_encrypted=True)
    
    current_count = 0
    if folder_path:
        if is_encrypted:
            current_count = count_encrypted_images(folder_name)
        else:
            image_files = [f for f in os.listdir(folder_path) 
                          if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            current_count = len(image_files)

    return render_template('update_images.html', 
                           user=session,
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

        name_no_accent = remove_accents(name)
        folder_name = f"{mssv}_{name_no_accent}"

        current_count = count_encrypted_images(folder_name)
        start_index = current_count + 1

        saved_count = 0
        for idx, img_data in enumerate(images):
            if 'base64,' in img_data:
                img_data = img_data.split('base64,')[1]
            
            img_bytes = base64.b64decode(img_data)
            img_array = np.frombuffer(img_bytes, dtype=np.uint8)
            img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            
            if img is not None:
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
    nfd = unicodedata.normalize('NFD', text)
    without_accents = ''.join(c for c in nfd if unicodedata.category(c) != 'Mn')
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
        images = data.get('images', [])
        
        if not mssv or not name or not images:
            return jsonify({'success': False, 'message': 'Thiếu thông tin'}), 400

        name_no_accent = remove_accents(name)
        folder_name = f"{mssv}_{name_no_accent}"

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

        from src.data.detect_align import align_faces_for_student

        folder_path, is_encrypted, folder_name = get_student_folder_path(mssv, check_encrypted=True)
        
        if not folder_path:
            return jsonify({'success': False, 'message': 'Không tìm thấy folder ảnh'}), 404

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
        embedder.build_embeddings()
        
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
        import subprocess
        script_path = os.path.join(SRC_DIR, "train.py")
        result = subprocess.run([sys.executable, script_path], 
                              capture_output=True, 
                              text=True,
                              cwd=BASE_DIR)
        
        if result.returncode == 0:
            global knn, le, THRESHOLD
            
            knn_path = os.path.join(MODEL_DIR, "best_knn_faceid.pkl")
            le_path = os.path.join(MODEL_DIR, "label_encoder.pkl")
            params_path = os.path.join(RESULT_DIR, "best_params_faceid.txt")
            
            if os.path.exists(knn_path) and os.path.exists(le_path):
                knn = joblib.load(knn_path)
                le = joblib.load(le_path)

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

        student_file = os.path.join(BASE_DIR, 'student_list.json')
        students = []
        
        if os.path.exists(student_file):
            with open(student_file, 'r', encoding='utf-8') as f:
                students = json.load(f)

        if any(s['mssv'] == mssv for s in students):
            return jsonify({'success': False, 'message': 'MSSV đã tồn tại'}), 400

        students.append({
            'mssv': mssv,
            'name': name,
            'gender': gender
        })

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

        date_str = datetime.now().strftime("%Y-%m-%d")
        csv_filename = f"attendance_{date_str}.csv"
        attended_records = {}
        
        if os.path.exists(csv_filename):
            try:
                with open(csv_filename, 'r', encoding='utf-8') as f:
                    csv_reader = csv.reader(f)
                    next(csv_reader, None)
                    for row in csv_reader:
                        if len(row) >= 3:
                            mssv = row[0]
                            time = row[2]
                            attended_records[mssv] = time
                print(f"✅ Đã load {len(attended_records)} bản ghi điểm danh từ {csv_filename}")
            except Exception as e:
                print(f"⚠️ Lỗi đọc file CSV điểm danh: {e}")

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

@app.route('/export_report')
def export_report():
    if not session.get('logged_in'):
        return redirect(url_for('login'))
    
    try:
        csv_files = glob.glob("attendance_*.csv")
        
        if not csv_files:
            flash('Chưa có dữ liệu điểm danh nào!', 'error')
            return redirect(url_for('dashboard'))

        latest_file = max(csv_files, key=os.path.getctime)

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

def get_attendance_status(check_time):
    """Xác định trạng thái điểm danh dựa trên thời gian
    
    Args:
        check_time: datetime object của thời gian điểm danh
    
    Returns:
        str: 'Đã điểm danh' / 'Trễ' / 'Vắng'
    """
    hour = check_time.hour
    minute = check_time.minute
    
    current_minutes = hour * 60 + minute
    

    if 750 <= current_minutes < 780:
        return "Đã điểm danh"
    elif 780 <= current_minutes < 840: 
        return "Trễ"
    else:
        return "Vắng"

def log_attendance_csv(mssv, name):
    """Ghi log điểm danh với trạng thái dựa trên thời gian"""
    now = datetime.now()
    date_str = now.strftime("%Y-%m-%d")
    filename = f"attendance_{date_str}.csv"

    status = get_attendance_status(now)
    time_str = now.strftime("%H:%M:%S")

    already_checked = False
    if os.path.exists(filename):
        with open(filename, 'r', encoding='utf-8') as f:
            if mssv in f.read():
                already_checked = True
    
    if not already_checked:
        with open(filename, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            if os.stat(filename).st_size == 0:
                writer.writerow(["MSSV", "HoTen", "ThoiGian", "TrangThai"])
            writer.writerow([mssv, name, time_str, status])
        
        print(f"📝 Điểm danh: {mssv} - {name} - {time_str} - [{status}]")
        return True, status
    return False, None 

@app.route('/api/process_frame', methods=['POST'])
def process_frame():
    now = datetime.now()
    current_minutes = now.hour * 60 + now.minute

    if current_minutes < 750:
        return jsonify({
            "status": "too_early",
            "message": "⏰ Chưa tới giờ điểm danh! Vui lòng quay lại sau 12:30.",
            "instruction": "too_early",
            "current_time": now.strftime("%H:%M")
        })

    client_ip = request.remote_addr
    if client_ip not in client_states:
        client_states[client_ip] = ClientState()
    
    state = client_states[client_ip]
    if time.time() - state.last_active > 30:
        state.reset()
    state.last_active = time.time()

    try:
        data = request.json['image']
        header, encoded = data.split(",", 1)
        nparr = np.frombuffer(base64.b64decode(encoded), np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        print(f"📸 Nhận frame: {frame.shape if frame is not None else 'NULL'} - Client: {client_ip} - Step: {state.step}")

        faces, boxes = detector.detect_from_frame(frame)
        
        if not faces or len(faces) == 0:
            if state.step == "liveness": 
                state.reset()
            print(f"⚠️ Không phát hiện khuôn mặt - Client: {client_ip}")
            return jsonify({"status": "waiting", "message": "Không thấy khuôn mặt. Hãy nhìn vào camera!", "instruction": "none"})

        face_img = faces[0]
        print(f"✅ Phát hiện {len(faces)} khuôn mặt - Client: {client_ip} - Step: {state.step}")

        if state.step == "detect" or state.step == "liveness":
            current_yaw, has_face = calculate_yaw_from_frame(frame)
            
            if state.step == "detect":
                state.initial_yaw = current_yaw
                state.start_time = time.time()
                state.step = "liveness"
                return jsonify({
                    "status": "liveness", 
                    "message": "Đã bắt khuôn mặt", 
                    "instruction": "turn_left"
                })
            
            elif state.step == "liveness":
                deviation = abs(state.initial_yaw - current_yaw)
                YAW_THRESHOLD = 25.0

                if time.time() - state.start_time > 10:
                    state.reset()
                    return jsonify({"status": "error", "message": "Hết giờ! Vui lòng thử lại.", "instruction": "retry"})

                if deviation > YAW_THRESHOLD:
                    state.step = "recognizing"
                else:
                    progress = int((deviation / YAW_THRESHOLD) * 100)
                    return jsonify({
                        "status": "liveness", 
                        "message": f"Quay trái tiếp... ({progress}%)", 
                        "instruction": "turn_left"
                    })

        if state.step == "recognizing":
            if knn is None:
                return jsonify({"status": "error", "message": "Lỗi Model Server"})

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

        now = datetime.now()
        filename = f"DanhSachDiemDanh_{now.strftime('%Y%m%d_%H%M%S')}.csv"

        os.makedirs(os.path.join(BASE_DIR, 'exports'), exist_ok=True)

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
        student_list_path = os.path.join(BASE_DIR, 'student_list.json')
        if not os.path.exists(student_list_path):
            return jsonify({"success": False, "message": "Không tìm thấy student_list.json"}), 404
        
        with open(student_list_path, 'r', encoding='utf-8') as f:
            all_students = json.load(f)

        date_str = datetime.now().strftime("%Y-%m-%d")
        attendance_file = f"attendance_{date_str}.csv"

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

        now = datetime.now()
        filename = f"BaoCaoDiemDanh_DayDu_{now.strftime('%Y%m%d_%H%M%S')}.csv"
        filepath = os.path.join(BASE_DIR, 'exports', filename)

        os.makedirs(os.path.join(BASE_DIR, 'exports'), exist_ok=True)

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

        total = len(all_students)
        attended = sum(1 for info in attendance_dict.values() if info['status'] == 'Đã điểm danh')
        late = sum(1 for info in attendance_dict.values() if info['status'] == 'Trễ')
        absent = total - attended - late
        
        print(f"📊 Báo cáo: Tổng {total} | Có mặt {attended} | Trễ {late} | Vắng {absent}")

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
