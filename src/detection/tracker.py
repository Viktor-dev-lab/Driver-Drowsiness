import numpy as np
from filterpy.kalman import KalmanFilter

class FaceTracker:
    def __init__(self):
        # Khởi tạo Kalman Filter với 8 biến trạng thái (state) và 4 biến đo lường (measurement)
        self.kf = KalmanFilter(dim_x=8, dim_z=4)
        
        # 1. Ma trận chuyển đổi trạng thái (State Transition Matrix - F)
        # Công thức vật lý: Vị trí mới = Vị trí cũ + Vận tốc * thời gian (dt=1)
        self.kf.F = np.array([
            [1, 0, 0, 0, 1, 0, 0, 0],
            [0, 1, 0, 0, 0, 1, 0, 0],
            [0, 0, 1, 0, 0, 0, 1, 0],
            [0, 0, 0, 1, 0, 0, 0, 1],
            [0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 1]
        ])
        
        # 2. Ma trận đo lường (Measurement Function - H)
        # Cách chúng ta "nhìn" vào hệ thống (chỉ lấy x, y, w, h, bỏ qua vận tốc)
        self.kf.H = np.array([
            [1, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0]
        ])
        
        # 3. Thiết lập các ma trận độ nhiễu (Tối ưu cho Tracking khuôn mặt)
        self.kf.P *= 1000.  # P: Bắt đầu với độ không chắc chắn cao
        self.kf.R *= 10.    # R: Nhiễu đo lường từ mô hình AI (RF-DETR có thể hơi rung)
        self.kf.Q *= 0.01   # Q: Nhiễu hệ thống (tài xế ít khi giật đầu quá nhanh)

        self.is_initialized = False

    def update(self, detected_box):
        """
        Nhận Bounding Box từ RF-DETR.
        Nếu detected_box là None (AI nghỉ), Kalman Filter tự nội suy frame tiếp theo.
        """
        if detected_box is not None:
            # Giải nén tọa độ AI gửi sang
            x, y, w, h = detected_box
            z = np.array([[x], [y], [w], [h]])

            if not self.is_initialized:
                # Lần đầu tiên tìm thấy mặt, ép tọa độ thực tế vào bộ lọc
                self.kf.x[:4] = z
                self.is_initialized = True
            else:
                # Nếu đã theo dõi rồi, thì vừa dự đoán vừa cập nhật (sửa sai)
                self.kf.predict()
                self.kf.update(z)
        else:
            # Khúc này AI bị tắt (Skip frame) hoặc mất dấu
            if self.is_initialized:
                # Tự đoán vị trí tiếp theo dựa trên quán tính vật lý
                self.kf.predict()
            else:
                # Chưa từng thấy mặt bao giờ thì không thể đoán được
                return None

        # Trích xuất tọa độ đã được làm mượt / dự đoán từ vector trạng thái
        pred_x, pred_y, pred_w, pred_h = self.kf.x[:4].flatten()
        
        # Đảm bảo chiều rộng và chiều cao không bao giờ bị âm do sai số toán học
        pred_w = max(1, pred_w)
        pred_h = max(1, pred_h)
        
        return (pred_x, pred_y, pred_w, pred_h) if self.is_initialized else None