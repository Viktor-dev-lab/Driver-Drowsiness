import cv2
import numpy as np

def extract_eye_patch(image, landmarks, eye_indices, size=(64, 64), padding=10):
    """
    Cắt vùng ảnh chứa con mắt dựa trên tọa độ landmarks.
    """
    h_img, w_img, _ = image.shape
    
    # Lấy tọa độ x, y của tất cả các điểm quanh mắt
    x_coords = [int(landmarks[i].x * w_img) for i in eye_indices]
    y_coords = [int(landmarks[i].y * h_img) for i in eye_indices]
    
    # Tìm khung chữ nhật bao quanh mắt
    x_min, x_max = min(x_coords), max(x_coords)
    y_min, y_max = min(y_coords), max(y_coords)
    
    # Thêm padding để lấy được cả lông mày và bóng đổ (rất tốt cho AI)
    x_min = max(0, x_min - padding)
    y_min = max(0, y_min - padding)
    x_max = min(w_img, x_max + padding)
    y_max = min(h_img, y_max + padding)
    
    # Cắt ảnh
    eye_crop = image[y_min:y_max, x_min:x_max]
    
    # Chuẩn hóa về đúng 64x64 nếu cắt thành công
    if eye_crop.size != 0:
        eye_crop_resized = cv2.resize(eye_crop, size)
        return eye_crop_resized
    return None