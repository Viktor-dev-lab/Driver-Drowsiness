import cv2
import sys
import os
import time
import glob
import random
import kagglehub
from dotenv import load_dotenv

# Đảm bảo import
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from src.preprocessing.image_enhancement import apply_clahe
from src.detection.rf_detr import RFDETRDetector
from src.detection.tracker import FaceTracker
from src.features.face_mesh import FaceMeshDetector
from src.features.geometry_calc import calculate_ear, calculate_mar, get_head_pose, LEFT_EYE, RIGHT_EYE, MOUTH
from src.features.patch_extractor import extract_eye_patch
from src.classification.vit_classifier import ViTEyeClassifier
from src.fusion.sliding_window import SlidingWindow
from src.fusion.lstm_voter import SpatiotemporalVoter

load_dotenv()
API_KEY = os.getenv("ROBOFLOW_API_KEY")
MODEL_ID = os.getenv("ROBOFLOW_MODEL_ID")

def process_single_video(video_path, label, out_dir, detector, face_mesh, vit_model, lstm_path):
    """
    Xử lý một video duy nhất và xuất ra file video mới có vẽ cảnh báo và landmarks.
    Tên file sẽ được gắn nhãn tự động.
    """
    file_name = os.path.basename(video_path)
    
    # Gắn nhãn vào tên file dựa theo label truyền vào
    prefix = "[DROWSY]" if label == 1 else "[AWAKE]"
    out_path = os.path.join(out_dir, f"{prefix}_result_{file_name}")
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[ERROR] Không thể đọc video: {video_path}")
        return

    # Lấy thông số gốc của video
    fps_video = int(cap.get(cv2.CAP_PROP_FPS))
    w_video = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h_video = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Khởi tạo công cụ ghi video (VideoWriter)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Định dạng .mp4
    out = cv2.VideoWriter(out_path, fourcc, fps_video, (w_video, h_video))

    # Reset Tracker, Cửa sổ trượt và Voter cho MỖI video mới
    tracker = FaceTracker()
    window = SlidingWindow(window_size=30)
    
    # Kích hoạt LSTM. Chú ý truyền đúng fps_video
    voter = SpatiotemporalVoter(lstm_weights_path=lstm_path, fps=fps_video, window_size=30)

    print(f" -> Đang xử lý: {prefix}_{file_name} (FPS: {fps_video}, Size: {w_video}x{h_video})")

    while True:
        ret, frame = cap.read()
        if not ret: break

        processed_frame = apply_clahe(frame)
        display_frame = processed_frame.copy()

        # PHASE 2: Detection (Không sử dụng SKIP_FRAMES, quét trên TỪNG frame)
        raw_coords = detector.detect(processed_frame)
        tracked_coords = tracker.update(raw_coords)
        box_color = (0, 255, 0)

        if tracked_coords:
            x, y, w, h = tracked_coords
            start_point = (int(x - w/2), int(y - h/2))
            end_point = (int(x + w/2), int(y + h/2))
            # Vẽ Box nhận diện khuôn mặt
            cv2.rectangle(display_frame, start_point, end_point, box_color, 2)

            rgb_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
            landmarks = face_mesh.get_landmarks(rgb_frame)

            if landmarks:
                # ==========================================
                # VẼ FACE LANDMARKS LÊN KHUÔN MẶT
                # ==========================================
                for lm in landmarks:
                    lm_x = int(lm.x * w_video)
                    lm_y = int(lm.y * h_video)
                    # Vẽ các chấm nhỏ li ti màu vàng nhạt bao phủ mặt
                    cv2.circle(display_frame, (lm_x, lm_y), 1, (153, 255, 255), -1)

                # PHASE 3: Features
                left_ear = calculate_ear(LEFT_EYE, landmarks, w_video, h_video)
                right_ear = calculate_ear(RIGHT_EYE, landmarks, w_video, h_video)
                avg_ear = (left_ear + right_ear) / 2.0
                mar = calculate_mar(MOUTH, landmarks, w_video, h_video)
                pitch, yaw, roll = get_head_pose(landmarks, w_video, h_video)

                left_eye_patch = extract_eye_patch(processed_frame, landmarks, LEFT_EYE)
                right_eye_patch = extract_eye_patch(processed_frame, landmarks, RIGHT_EYE)

                # PHASE 4: ViT Spatial State
                left_state = vit_model.predict(left_eye_patch)[0] if left_eye_patch is not None else -1
                right_state = vit_model.predict(right_eye_patch)[0] if right_eye_patch is not None else -1
                current_vit_state = 1 if (left_state == 1 and right_state == 1) else 0

                # PHASE 5: Spatiotemporal Fusion
                window.add_data(avg_ear, mar, pitch, yaw, current_vit_state)
                window_data = window.get_window()
                driver_status, status_color = voter.evaluate(window_data)
                
                # Hiển thị kết quả Cảnh báo
                cv2.putText(display_frame, f"STATUS: {driver_status}", (20, 40), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)
                cv2.putText(display_frame, f"EAR: {avg_ear:.2f} | MAR: {mar:.2f} | Pitch: {pitch:.1f}", (20, 80), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                cv2.putText(display_frame, f"Queue: {len(window_data)}/30 frames", (20, 110), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

                # Chèn ảnh mắt đã crop vào góc màn hình (ĐÃ PHÓNG TO)
                if left_eye_patch is not None and right_eye_patch is not None:
                    try:
                        # Phóng to ảnh mắt lên 128x128 
                        eye_h, eye_w = 128, 128 
                        l_eye_resized = cv2.resize(left_eye_patch, (eye_w, eye_h))
                        r_eye_resized = cv2.resize(right_eye_patch, (eye_w, eye_h))
                        
                        # In chữ O/C (Open/Closed) lên mắt
                        cv2.putText(l_eye_resized, "C" if left_state == 1 else "O", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255) if left_state==1 else (0,255,0), 3)
                        cv2.putText(r_eye_resized, "C" if right_state == 1 else "O", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255) if right_state==1 else (0,255,0), 3)

                        # Đè ảnh mắt vào góc phải dưới màn hình
                        display_frame[h_video-eye_h-10 : h_video-10, w_video-eye_w*2-20 : w_video-eye_w-20] = l_eye_resized
                        display_frame[h_video-eye_h-10 : h_video-10, w_video-eye_w-10 : w_video-10] = r_eye_resized
                    except:
                        pass
        else:
            cv2.putText(display_frame, "WARNING: Face Lost!", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # Ghi frame vào video đầu ra
        out.write(display_frame)

    cap.release()
    out.release()
    print(f" [SUCCESS] Đã lưu video kết quả tại: {out_path}")

def main():
    # 1. Tải Dataset
    print("[INFO] Đang kiểm tra/tải Dataset SUST-DDD từ Kaggle...")
    dataset_path = kagglehub.dataset_download("esrakavalci/sust-ddd")
    
    drowsy_videos = glob.glob(os.path.join(dataset_path, '**', 'd_*.mp4'), recursive=True)
    awake_videos = glob.glob(os.path.join(dataset_path, '**', 'n_*.mp4'), recursive=True)

    if not drowsy_videos or not awake_videos:
        print("[ERROR] Không tìm thấy đủ video trong Dataset. Vui lòng kiểm tra lại.")
        return

    # BỐC MẪU NGẪU NHIÊN: Lấy 5 video ngủ và 5 video tỉnh
    LIMIT = 5
    sampled_drowsy = random.sample(drowsy_videos, min(LIMIT, len(drowsy_videos)))
    sampled_awake = random.sample(awake_videos, min(LIMIT, len(awake_videos)))

    tasks = [(v, 1) for v in sampled_drowsy] + [(v, 0) for v in sampled_awake]

    # 2. Tạo thư mục chứa video đầu ra
    out_dir = os.path.join(current_dir, "output_videos")
    os.makedirs(out_dir, exist_ok=True)
    print(f"[INFO] Bốc ngẫu nhiên {len(tasks)} video để test. Kết quả sẽ lưu tại: {out_dir}")

    # 3. Khởi tạo Mô hình AI
    print("[INFO] Đang khởi tạo các mô hình AI (RF-DETR, FaceMesh, ViT)...")
    detector = RFDETRDetector(model_id=MODEL_ID, api_key=API_KEY)
    face_mesh = FaceMeshDetector(max_faces=1)
    vit_model = ViTEyeClassifier()
    
    lstm_path = os.path.join(current_dir, "src", "model", "lstm_drowsiness_best.pth")
    if not os.path.exists(lstm_path):
        print(f"[WARNING] Chưa tìm thấy file LSTM tại {lstm_path}. Hệ thống sẽ dùng Soft Voting!")

    # 4. Chạy vòng lặp xử lý tất cả video
    print(f"\n[🚀] BẮT ĐẦU XỬ LÝ {len(tasks)} VIDEO...")
    for idx, (video_path, label) in enumerate(tasks):
        print(f"\n[{idx+1}/{len(tasks)}] Video nhãn: {'DROWSY' if label == 1 else 'AWAKE'}")
        process_single_video(video_path, label, out_dir, detector, face_mesh, vit_model, lstm_path)

    print("\n[🎉] HOÀN TẤT TOÀN BỘ QUÁ TRÌNH KIỂM TRA!")

if __name__ == "__main__":
    main()