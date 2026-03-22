import cv2

class CameraStream:
    def __init__(self, src=0):
        self.cap = cv2.VideoCapture(src)
        if not self.cap.isOpened():
            raise ValueError(f"[ERROR] Cannot open video source: {src}")

    def read(self):
        ret, frame = self.cap.read()
        return ret, frame

    def stop(self):
        self.cap.release()