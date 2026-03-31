import cv2
import sys
import os
import time
from dotenv import load_dotenv

# Đảm bảo import
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# Import GĐ 1 & 2
from src.preprocessing.camera_stream import CameraStream
from src.preprocessing.image_enhancement import apply_clahe
from src.detection.rf_detr import RFDETRDetector
from src.detection.tracker import FaceTracker

# Import GĐ 3
from src.features.face_mesh import FaceMeshDetector
from src.features.geometry_calc import calculate_ear, calculate_mar, get_head_pose, LEFT_EYE, RIGHT_EYE, MOUTH
from src.features.patch_extractor import extract_eye_patch

# Import GĐ 4
from src.classification.vit_classifier import ViTEyeClassifier

# Import GĐ 5 (MỚI)
from src.fusion.sliding_window import SlidingWindow
from src.fusion.lstm_voter import SpatiotemporalVoter

load_dotenv()
API_KEY = os.getenv("ROBOFLOW_API_KEY")
MODEL_ID = os.getenv("ROBOFLOW_MODEL_ID")
IP_CAMERA_URL = "http://admin:123@192.168.2.51:8081/video"
SKIP_FRAMES = 5

def main():
    cam = CameraStream(src=IP_CAMERA_URL)
    detector = RFDETRDetector(model_id=MODEL_ID, api_key=API_KEY)
    tracker = FaceTracker()
    face_mesh = FaceMeshDetector(max_faces=1)
    vit_model = ViTEyeClassifier() 
    
    # Khởi tạo Cửa sổ trượt và Bộ ra quyết định (Phase 5)
    window = SlidingWindow(window_size=60)
    voter = SpatiotemporalVoter(fps=5)

    print("[INFO] Starting Full System (Phase 1-5)... Press 'q' to exit.")

    frame_counter = 0
    start_time = time.time()
    fps = 0

    while True:
        ret, frame = cam.read()
        if not ret: break

        processed_frame = apply_clahe(frame)
        display_frame = processed_frame.copy()
        h_img, w_img, _ = processed_frame.shape

        # PHASE 2: Detection
        if frame_counter % SKIP_FRAMES == 0:
            raw_coords = detector.detect(processed_frame)
            tracked_coords = tracker.update(raw_coords)
            box_color = (0, 255, 0)
        else:
            tracked_coords = tracker.update(None)
            box_color = (0, 255, 255)

        if tracked_coords:
            x, y, w, h = tracked_coords
            start_point = (int(x - w/2), int(y - h/2))
            end_point = (int(x + w/2), int(y + h/2))
            cv2.rectangle(display_frame, start_point, end_point, box_color, 2)

            rgb_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
            landmarks = face_mesh.get_landmarks(rgb_frame)

            if landmarks:
                # PHASE 3: Features
                left_ear = calculate_ear(LEFT_EYE, landmarks, w_img, h_img)
                right_ear = calculate_ear(RIGHT_EYE, landmarks, w_img, h_img)
                avg_ear = (left_ear + right_ear) / 2.0
                mar = calculate_mar(MOUTH, landmarks, w_img, h_img)
                pitch, yaw, roll = get_head_pose(landmarks, w_img, h_img)

                left_eye_patch = extract_eye_patch(processed_frame, landmarks, LEFT_EYE)
                right_eye_patch = extract_eye_patch(processed_frame, landmarks, RIGHT_EYE)

                # PHASE 4: ViT Spatial State (Khoảnh khắc)
                left_state = vit_model.predict(left_eye_patch)[0] if left_eye_patch is not None else -1
                right_state = vit_model.predict(right_eye_patch)[0] if right_eye_patch is not None else -1
                
                # 1: Nhắm, 0: Mở. Chỉ ghi nhận "Nhắm" nếu cả 2 mắt cùng nhắm
                current_vit_state = 1 if (left_state == 1 and right_state == 1) else 0

                if left_eye_patch is not None:
                    # Phóng to ảnh mắt lên một chút (từ 64x64 lên 128x128) để dễ nhìn hơn
                    left_display = cv2.resize(left_eye_patch, (128, 128))
                    
                    # In trạng thái ViT lên cửa sổ mắt trái
                    txt = "CLOSED" if left_state == 1 else "OPEN"
                    color = (0, 0, 255) if left_state == 1 else (0, 255, 0)
                    cv2.putText(left_display, txt, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                    
                    cv2.imshow("Left Eye Crop", left_display)
                    
                if right_eye_patch is not None:
                    right_display = cv2.resize(right_eye_patch, (128, 128))
                    
                    # In trạng thái ViT lên cửa sổ mắt phải
                    txt = "CLOSED" if right_state == 1 else "OPEN"
                    color = (0, 0, 255) if right_state == 1 else (0, 255, 0)
                    cv2.putText(right_display, txt, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                    
                    cv2.imshow("Right Eye Crop", right_display)
                    
                # PHASE 5: Spatiotemporal Fusion (Thời gian)
                # Nạp vector vào hàng đợi
                window.add_data(avg_ear, mar, pitch, yaw, current_vit_state)
                
                # Rút dữ liệu từ cửa sổ trượt ra đánh giá
                window_data = window.get_window()
                driver_status, status_color = voter.evaluate(window_data)
                
                # Hiển thị kết quả Cảnh báo cuối cùng
                cv2.putText(display_frame, f"STATUS: {driver_status}", (20, 40), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)

                # Hiển thị số liệu nhỏ giọt để debug
                cv2.putText(display_frame, f"EAR: {avg_ear:.2f} | Pitch: {pitch:.1f}", (20, 80), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                cv2.putText(display_frame, f"Queue: {len(window_data)}/60 frames", (20, 110), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

        else:
            cv2.putText(display_frame, "WARNING: Track Lost!", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # Đếm FPS
        frame_counter += 1
        elapsed_time = time.time() - start_time
        if elapsed_time >= 1.0:
            fps = frame_counter / elapsed_time
            frame_counter = 0
            start_time = time.time()

        cv2.putText(display_frame, f"FPS: {int(fps)}", (w_img - 120, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

        cv2.imshow("Drosiness Driver - Phase 1 to 5", display_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cam.stop()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()