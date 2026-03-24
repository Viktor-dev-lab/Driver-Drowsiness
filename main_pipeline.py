import cv2
import sys
import os
import time
from dotenv import load_dotenv

# Đảm bảo đường dẫn import hoạt động không cần __init__.py
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# Import Giai đoạn 1 & 2
from src.preprocessing.camera_stream import CameraStream
from src.preprocessing.image_enhancement import apply_clahe
from src.detection.rf_detr import RFDETRDetector
from src.detection.tracker import FaceTracker

# Import Giai đoạn 3
from src.features.face_mesh import FaceMeshDetector
from src.features.geometry_calc import calculate_ear, calculate_mar, get_head_pose, LEFT_EYE, RIGHT_EYE, MOUTH
from src.features.patch_extractor import extract_eye_patch

# Tải biến môi trường từ file .env
load_dotenv()
API_KEY = os.getenv("ROBOFLOW_API_KEY")
MODEL_ID = os.getenv("ROBOFLOW_MODEL_ID")

# Cấu hình tối ưu hệ thống
SKIP_FRAMES = 5  # Cứ 5 frame chạy AI (RF-DETR) 1 lần

def main():
    # 1. Khởi tạo các module (Models & Trackers)
    cam = CameraStream(src=0)
    detector = RFDETRDetector(model_id=MODEL_ID, api_key=API_KEY)
    tracker = FaceTracker()
    face_mesh = FaceMeshDetector(max_faces=1)

    print("[INFO] Starting Full Pipeline (Phase 1-3)... Press 'q' to exit.")

    frame_counter = 0
    start_time = time.time()
    fps = 0

    while True:
        ret, frame = cam.read()
        if not ret:
            print("[WARNING] Dropped frame or end of stream.")
            break

        # ==========================================
        # GIAI ĐOẠN 1: TIỀN XỬ LÝ
        # ==========================================
        processed_frame = apply_clahe(frame)
        display_frame = processed_frame.copy()

        # ==========================================
        # GIAI ĐOẠN 2: DETECT & TRACKING (Khung mặt)
        # ==========================================
        if frame_counter % SKIP_FRAMES == 0:
            raw_coords = detector.detect(processed_frame)
            tracked_coords = tracker.update(raw_coords)
            box_color = (0, 255, 0) # Xanh lá (AI RF-DETR chạy)
        else:
            tracked_coords = tracker.update(None)
            box_color = (0, 255, 255) # Vàng (Kalman Filter dự đoán)

        # ==========================================
        # GIAI ĐOẠN 3: ĐẶC TRƯNG 3D & CẮT MẮT
        # ==========================================
        if tracked_coords:
            # 3.1 Vẽ Bounding Box chứa mặt
            x, y, w, h = tracked_coords
            start_point = (int(x - w/2), int(y - h/2))
            end_point = (int(x + w/2), int(y + h/2))
            cv2.rectangle(display_frame, start_point, end_point, box_color, 2)

            # 3.2 Gọi MediaPipe trích xuất 468 điểm 3D
            # Chú ý: MediaPipe cần ảnh RGB
            rgb_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
            landmarks = face_mesh.get_landmarks(rgb_frame)

            if landmarks:
                h_img, w_img, _ = processed_frame.shape

                # 3.3 Tính toán các chỉ số toán học
                left_ear = calculate_ear(LEFT_EYE, landmarks, w_img, h_img)
                right_ear = calculate_ear(RIGHT_EYE, landmarks, w_img, h_img)
                avg_ear = (left_ear + right_ear) / 2.0
                mar = calculate_mar(MOUTH, landmarks, w_img, h_img)
                pitch, yaw, roll = get_head_pose(landmarks, w_img, h_img)

                # 3.4 Cắt Eye Patch (Chuẩn bị cho GĐ 4)
                left_eye_patch = extract_eye_patch(processed_frame, landmarks, LEFT_EYE)
                right_eye_patch = extract_eye_patch(processed_frame, landmarks, RIGHT_EYE)

                # Hiển thị số liệu lên màn hình
                cv2.putText(display_frame, f"EAR: {avg_ear:.2f}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(display_frame, f"MAR: {mar:.2f}", (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(display_frame, f"Pitch: {pitch:.1f}  Yaw: {yaw:.1f}", (20, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                # (Tuỳ chọn) Bạn có thể imshow thêm left_eye_patch ra một cửa sổ nhỏ để kiểm tra xem cắt chuẩn chưa
                if left_eye_patch is not None:
                    cv2.imshow("Left Eye Crop (64x64)", left_eye_patch)

        else:
            cv2.putText(display_frame, "WARNING: Track Lost!", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # ==========================================
        # FPS COUNTER
        # ==========================================
        frame_counter += 1
        elapsed_time = time.time() - start_time
        if elapsed_time >= 1.0:
            fps = frame_counter / elapsed_time
            frame_counter = 0
            start_time = time.time()

        cv2.putText(display_frame, f"FPS: {int(fps)}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        # Hiển thị luồng chính
        cv2.imshow("Drosiness Driver - Phase 1 to 3", display_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cam.stop()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()