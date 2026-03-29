import cv2
import sys

class CameraStream:
    def __init__(self, src=0, width=640, height=480):
        self.src = src
        self.width = width
        self.height = height
        
        print(f"[INFO] Connecting to camera source: {self.src}...")
        self.cap = cv2.VideoCapture(self.src)
        
        # Mặc định set độ phân giải
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        
        if not self.cap.isOpened():
            print(f"[ERROR] Failed to connect to camera: {self.src}")
            sys.exit(1)

        self.is_ip_camera = isinstance(self.src, str)

    def read(self):
        success, frame = self.cap.read()
        if not success:
            return False, None
        
        # Chỉ xoay và lật nếu bạn đang dùng Camera điện thoại (dạng URL)
        # Vì thường app điện thoại hay bị ngược trục tọa độ
        if self.is_ip_camera:
            frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
            frame = cv2.flip(frame, 1)
            
        frame = cv2.resize(frame, (self.width, self.height))
        
        return True, frame
    
    def stop(self):
        """
        Giải phóng camera, đổi tên thành stop() thay vì release()
        """
        self.cap.release()