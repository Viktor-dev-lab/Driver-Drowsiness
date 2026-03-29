# Cách import an toàn cho mediapipe 0.10.x
import mediapipe as mp

mp_face_mesh = mp.solutions.face_mesh

class FaceMeshDetector:
    def __init__(self, max_faces=1):
        # Khởi tạo trực tiếp từ module đã import
        self.face_mesh = mp_face_mesh.FaceMesh(
            max_num_faces=max_faces,
            refine_landmarks=True, # Lấy chi tiết con ngươi
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

    def get_landmarks(self, frame_rgb):
        """
        Nhận ảnh RGB và trả về 468 điểm tọa độ khuôn mặt.
        """
        results = self.face_mesh.process(frame_rgb)
        if results.multi_face_landmarks:
            return results.multi_face_landmarks[0].landmark
        return None