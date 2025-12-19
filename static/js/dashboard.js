tailwind.config = {
    darkMode: 'class',
    theme: {
        extend: {
            colors: {
                primary: '#3b82f6',
                'background-light': '#f3f4f6',
                'surface-light': '#ffffff',
                'background-dark': '#0f172a',
                'surface-dark': '#1e293b',
                'text-main': '#334155',
                'text-secondary': '#64748b'
            }
        }
    }
}


const attendancePanel = document.getElementById('attendance-panel');
const btnToggle = document.getElementById('btn-toggle-cam');
const video = document.getElementById('video');
const canvas = document.getElementById('canvas');
const overlayGuide = document.getElementById('overlay-guide');
const guideText = document.getElementById('guide-text');
const guideIcon = document.getElementById('guide-icon');
const progressBar = document.getElementById('progress-bar');
const logList = document.getElementById('live-log-list');
const statusText = document.getElementById('status-text');
const emptyLogMsg = document.getElementById('empty-log-msg');

let stream = null;
let intervalId = null;

// 1. Hàm bật/tắt Panel
async function toggleAttendancePanel() {
    if (attendancePanel.classList.contains('hidden')) {
        // Mở Panel
        attendancePanel.classList.remove('hidden');
        btnToggle.innerHTML = '<span class="material-symbols-outlined text-[20px]">stop_circle</span> Dừng điểm danh';
        btnToggle.classList.replace('bg-rose-600', 'bg-slate-600');
        btnToggle.classList.replace('hover:bg-rose-700', 'hover:bg-slate-700');

        await startCamera();
    } else {
        // Đóng Panel
        stopCamera();
        attendancePanel.classList.add('hidden');
        btnToggle.innerHTML = '<span class="material-symbols-outlined text-[20px]">videocam</span> Bắt đầu điểm danh';
        btnToggle.classList.replace('bg-slate-600', 'bg-rose-600');
        btnToggle.classList.replace('hover:bg-slate-700', 'hover:bg-rose-700');
    }
}

// 2. Mở Camera
async function startCamera() {
    try {
        stream = await navigator.mediaDevices.getUserMedia({ video: { width: 640, height: 480 } });
        video.srcObject = stream;
        statusText.innerText = "Camera đang hoạt động. Đang tìm khuôn mặt...";
        statusText.className = "text-xs text-green-600 font-bold mt-1 animate-pulse";

        // Gửi ảnh mỗi 400ms
        intervalId = setInterval(captureAndSend, 400);
    } catch (err) {
        console.error(err);
        alert("Không thể mở Camera! Vui lòng kiểm tra quyền truy cập.");
        toggleAttendancePanel(); // Tắt lại nếu lỗi
    }
}

// 3. Tắt Camera
function stopCamera() {
    if (stream) {
        stream.getTracks().forEach(track => track.stop());
        video.srcObject = null;
    }
    clearInterval(intervalId);
    overlayGuide.classList.add('hidden');
}

// 4. Chụp ảnh & Gửi API
function captureAndSend() {
    if (!stream || video.paused || video.ended) return;

    const ctx = canvas.getContext('2d');
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    ctx.drawImage(video, 0, 0);

    // Convert sang base64 (giảm chất lượng xuống 0.6 cho nhẹ)
    const dataURL = canvas.toDataURL('image/jpeg', 0.6);

    fetch('/api/process_frame', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ image: dataURL })
    })
        .then(res => res.json())
        .then(data => updateUI(data))
        .catch(err => console.error("Lỗi API:", err));
}

// 5. Cập nhật Giao diện (Quan trọng nhất)
function updateUI(data) {
    // -- LOGIC OVERLAY (Liveness) --
    if (data.status === 'liveness' || data.status === 'liveness_check') {
        overlayGuide.classList.remove('hidden');
        guideText.innerText = data.message;

        if (data.instruction === 'turn_left') {
            // Hiện icon quay trái
            guideIcon.innerHTML = '<span class="material-symbols-outlined text-9xl">arrow_back</span>';

            // Cập nhật thanh progress (nếu server trả về progress)
            // Nếu server chưa trả về progress, bạn có thể tự tính hoặc bỏ qua
            let percent = data.progress || 0;
            // Hack nhẹ: Nếu đang liveness mà progress 0 thì cho nó nhích 10% cho đẹp
            if (percent === 0) percent = 10;
            progressBar.style.width = percent + "%";
        }
    }
    else if (data.status === 'success') {
        // Hiệu ứng Thành công
        overlayGuide.classList.remove('hidden');
        guideText.innerText = "ĐIỂM DANH THÀNH CÔNG";
        guideIcon.innerHTML = '<span class="material-symbols-outlined text-9xl text-green-400">check_circle</span>';
        progressBar.style.width = "100%";
        progressBar.classList.add('bg-green-400');

        // Thêm vào Log & Tắt overlay sau 1.5s
        setTimeout(() => {
            overlayGuide.classList.add('hidden');
            progressBar.style.width = "0%";
            addLogItem(data.data); // data.data chứa {mssv, name}
        }, 1500);
    }
    else {
        // Trạng thái chờ / detect / lỗi
        overlayGuide.classList.add('hidden');
        progressBar.style.width = "0%";
    }

    // -- LOGIC STATUS TEXT NHỎ --
    if (data.status === 'waiting') {
        statusText.innerText = "Đang tìm khuôn mặt...";
        statusText.className = "text-xs text-orange-500 font-bold mt-1";
    } else if (data.status === 'liveness') {
        statusText.innerText = "Phát hiện người. Đang kiểm tra Liveness...";
        statusText.className = "text-xs text-blue-500 font-bold mt-1";
    } else if (data.status === 'error') {
        statusText.innerText = "Lỗi: " + data.message;
        statusText.className = "text-xs text-red-500 font-bold mt-1";
    }
}

// 6. Thêm Log vào danh sách
function addLogItem(studentData) {
    if (!studentData) return;

    if (emptyLogMsg) emptyLogMsg.style.display = 'none';

    const timeStr = new Date().toLocaleTimeString('vi-VN');
    const newItem = document.createElement('div');

    // Style cho log item
    newItem.className = "flex items-center gap-3 p-3 rounded-xl bg-green-50 dark:bg-green-900/20 border border-green-100 dark:border-green-800 log-item-enter";

    newItem.innerHTML = `
                <div class="w-10 h-10 rounded-full bg-gradient-to-br from-green-400 to-emerald-600 text-white flex items-center justify-center font-bold text-xs shadow-sm">
                    ${studentData.mssv.substring(0, 2)}
                </div>
                <div class="flex-1 min-w-0">
                    <p class="text-sm font-bold text-slate-800 dark:text-slate-100 truncate">${studentData.name}</p>
                    <p class="text-xs text-green-600 dark:text-green-400 flex items-center gap-1">
                        <span class="material-symbols-outlined text-[12px]">check</span> ${timeStr}
                    </p>
                </div>
                <div class="text-xs font-mono text-slate-400 bg-white dark:bg-slate-800 px-2 py-1 rounded border border-slate-100 dark:border-slate-700">
                    ${studentData.mssv}
                </div>
            `;

    logList.prepend(newItem); // Thêm lên đầu danh sách
}