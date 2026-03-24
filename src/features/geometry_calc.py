import math
import cv2
import numpy as np

# Các chỉ số (indices) chuẩn của MediaPipe Face Mesh
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]
MOUTH = [13, 14, 78, 308] # Top, Bottom, Left, Right

def calculate_distance(p1, p2):
    return math.hypot(p2[0] - p1[0], p2[1] - p1[1])

def calculate_ear(eye_indices, landmarks, w, h):
    """Tính tỷ lệ nhắm/mở của mắt (EAR)"""
    eye_points = [(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in eye_indices]
    v1 = calculate_distance(eye_points[1], eye_points[5])
    v2 = calculate_distance(eye_points[2], eye_points[4])
    h_dist = calculate_distance(eye_points[0], eye_points[3])
    return (v1 + v2) / (2.0 * h_dist) if h_dist != 0 else 0

def calculate_mar(mouth_indices, landmarks, w, h):
    """Tính tỷ lệ nháp miệng (MAR) để phát hiện ngáp"""
    p_top = (int(landmarks[mouth_indices[0]].x * w), int(landmarks[mouth_indices[0]].y * h))
    p_bottom = (int(landmarks[mouth_indices[1]].x * w), int(landmarks[mouth_indices[1]].y * h))
    p_left = (int(landmarks[mouth_indices[2]].x * w), int(landmarks[mouth_indices[2]].y * h))
    p_right = (int(landmarks[mouth_indices[3]].x * w), int(landmarks[mouth_indices[3]].y * h))
    v_dist = calculate_distance(p_top, p_bottom)
    h_dist = calculate_distance(p_left, p_right)
    return v_dist / h_dist if h_dist != 0 else 0

def get_head_pose(landmarks, w, h):
    """Tính toán góc quay của đầu bằng thuật toán PnP"""
    # 1. Trích xuất 6 điểm 2D từ MediaPipe
    face_2d = []
    face_3d = []
    
    # Mũi, Cằm, Mắt trái, Mắt phải, Mép trái, Mép phải
    key_indices = [1, 152, 33, 263, 61, 291] 
    
    for idx, lm in enumerate(landmarks):
        if idx in key_indices:
            x, y = int(lm.x * w), int(lm.y * h)
            face_2d.append([x, y])
            face_3d.append([x, y, lm.z]) # Lấy thêm trục Z giả lập từ MediaPipe
            
    face_2d = np.array(face_2d, dtype=np.float64)
    face_3d = np.array(face_3d, dtype=np.float64)

    # 2. Ma trận camera (Giả lập focal length bằng chiều rộng ảnh)
    focal_length = 1 * w
    cam_matrix = np.array([
        [focal_length, 0, h / 2],
        [0, focal_length, w / 2],
        [0, 0, 1]
    ])
    dist_matrix = np.zeros((4, 1), dtype=np.float64)

    # 3. Giải bài toán PnP
    success, rot_vec, trans_vec = cv2.solvePnP(face_3d, face_2d, cam_matrix, dist_matrix)
    
    if not success:
        return 0, 0, 0
        
    # Chuyển vector xoay thành ma trận xoay, rồi tính góc Euler
    rmat, _ = cv2.Rodrigues(rot_vec)
    angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)
    
    pitch = angles[0] * 360 # Cúi / Ngửa (Gật gù)
    yaw = angles[1] * 360   # Quay trái / phải
    roll = angles[2] * 360  # Nghiêng đầu
    
    return pitch, yaw, roll