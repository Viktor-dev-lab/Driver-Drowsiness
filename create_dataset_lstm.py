import cv2
import csv
import os
import sys
import glob
import kagglehub
import concurrent.futures

current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from src.features.face_mesh import FaceMeshDetector
from src.features.geometry_calc import calculate_ear, calculate_mar, get_head_pose, LEFT_EYE, RIGHT_EYE, MOUTH
from src.features.patch_extractor import extract_eye_patch
from src.classification.vit_classifier import ViTEyeClassifier

# ==========================================
# KHỞI TẠO MÔ HÌNH TOÀN CỤC CHO TỪNG NHÂN
# ==========================================
global_face_mesh = None
global_vit_model = None

def init_worker():
    """
    Hàm này chỉ chạy ĐÚNG 1 LẦN khi mỗi nhân CPU (Worker) được sinh ra.
    Nó tải AI lên RAM của nhân đó và giữ nguyên trong suốt quá trình chạy.
    """
    global global_face_mesh, global_vit_model
    global_face_mesh = FaceMeshDetector(max_faces=1)
    global_vit_model = ViTEyeClassifier()

# ==========================================
# HÀM XỬ LÝ CHO TỪNG NHÂN
# ==========================================
def process_single_video(args):
    video_path, label = args
    global global_face_mesh, global_vit_model

    # Xài ké mô hình đã được tải sẵn ở hàm init_worker
    face_mesh = global_face_mesh
    vit_model = global_vit_model

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return video_path, []

    fps_video = cap.get(cv2.CAP_PROP_FPS)
    frame_skip = int(fps_video / 10) if fps_video > 10 else 1 
    
    frame_count = 0
    extracted_rows = []

    while True:
        ret, frame = cap.read()
        if not ret: break
        
        frame_count += 1
        if frame_count % frame_skip != 0: 
            continue 

        h_img, w_img, _ = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        landmarks = face_mesh.get_landmarks(rgb_frame)

        if landmarks:
            left_ear = calculate_ear(LEFT_EYE, landmarks, w_img, h_img)
            right_ear = calculate_ear(RIGHT_EYE, landmarks, w_img, h_img)
            avg_ear = round((left_ear + right_ear) / 2.0, 4)
            mar = round(calculate_mar(MOUTH, landmarks, w_img, h_img), 4)
            pitch, yaw, _ = get_head_pose(landmarks, w_img, h_img)

            left_patch = extract_eye_patch(frame, landmarks, LEFT_EYE)
            right_patch = extract_eye_patch(frame, landmarks, RIGHT_EYE)
            
            left_state = vit_model.predict(left_patch)[0] if left_patch is not None else 0
            right_state = vit_model.predict(right_patch)[0] if right_patch is not None else 0
            vit_state = 1 if (left_state == 1 and right_state == 1) else 0

            extracted_rows.append([avg_ear, mar, round(pitch, 2), round(yaw, 2), vit_state, label])

    cap.release()
    return video_path, extracted_rows

# ==========================================
# LUỒNG ĐIỀU KHIỂN CHÍNH
# ==========================================
def main():
    print("[INFO] Đang tải Dataset SUST-DDD từ Kaggle (Vui lòng chờ...).")
    dataset_path = kagglehub.dataset_download("esrakavalci/sust-ddd")
    
    drowsy_videos = glob.glob(os.path.join(dataset_path, '**', 'd_*.mp4'), recursive=True)
    awake_videos = glob.glob(os.path.join(dataset_path, '**', 'n_*.mp4'), recursive=True)
    
    # ==========================================
    # CHỈ LẤY 50 VIDEO MỖI LOẠI ĐỂ TEST
    # ==========================================
    # LIMIT = 50
    # drowsy_videos = drowsy_videos[:LIMIT]
    # awake_videos = awake_videos[:LIMIT]
    
    print(f"[INFO] Đã cắt dữ liệu: Lấy {len(drowsy_videos)} video Ngủ gật và {len(awake_videos)} video Tỉnh táo.")

    tasks = [(video, 1) for video in drowsy_videos] + [(video, 0) for video in awake_videos]
    total_videos = len(tasks)
    
    csv_path = "dataset_lstm.csv"
    max_workers = max(1, os.cpu_count() - 2)
    print(f"\n[🚀] BẮT ĐẦU XỬ LÝ ĐA NHÂN VỚI {max_workers} WORKERS...")

    with open(csv_path, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['ear', 'mar', 'pitch', 'yaw', 'vit_state', 'label'])
        f.flush()

        completed_count = 0
        
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers, initializer=init_worker) as executor:
            futures = {executor.submit(process_single_video, task): task for task in tasks}

            for future in concurrent.futures.as_completed(futures):
                try:
                    video_path, rows = future.result()
                    
                    if rows:
                        writer.writerows(rows)
                        f.flush() 
                        
                    completed_count += 1
                    file_name = os.path.basename(video_path)
                    print(f"[{completed_count}/{total_videos}] ✅ Xong: {file_name} -> Ghi {len(rows)} dòng vào CSV.")
                    
                except Exception as exc:
                    print(f"[LỖI] Xử lý video thất bại. Chi tiết: {exc}")

    print(f"\n[SUCCESS] Hoàn tất! Dữ liệu đã được lưu vào {csv_path}")

if __name__ == "__main__":
    main()