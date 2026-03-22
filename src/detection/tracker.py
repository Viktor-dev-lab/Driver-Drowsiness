class FaceTracker:
    def __init__(self):
        # Nơi khởi tạo Kalman Filter sau này
        pass
        
    def update(self, detected_box):
        """
        Nhận Bounding Box từ RF-DETR.
        Nếu detected_box là None (mất dấu), Kalman Filter sẽ dự đoán tọa độ tiếp theo.
        """
        # TODO: Code logic ByteTrack ở đây
        return detected_box