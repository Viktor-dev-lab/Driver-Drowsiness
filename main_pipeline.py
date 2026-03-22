import cv2
import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from src.preprocessing.camera_stream import CameraStream
from src.preprocessing.image_enhancement import apply_clahe
from src.detection.rf_detr import RFDETRDetector
from src.detection.tracker import FaceTracker

API_KEY = "g3YO1gZV4kABhraGFSCB"
MODEL_ID = "driver-face-detection-dzicg-1pf6o/1"

def main():
    cam = CameraStream(src=0)
    detector = RFDETRDetector(model_id=MODEL_ID, api_key=API_KEY)
    tracker = FaceTracker()

    print("[INFO] Starting pipeline... Press 'q' to exit.")

    while True:
        # Lấy frame từ luồng camera
        ret, frame = cam.read()
        if not ret:
            print("[WARNING] Dropped frame or end of stream.")
            break

        processed_frame = apply_clahe(frame)
        raw_coords = detector.detect(processed_frame)
        tracked_coords = tracker.update(raw_coords)
        display_frame = processed_frame.copy()
        
        if tracked_coords:
            x, y, w, h = tracked_coords
            start_point = (int(x - w/2), int(y - h/2))
            end_point = (int(x + w/2), int(y + h/2))
            
            cv2.rectangle(display_frame, start_point, end_point, (0, 255, 0), 2)
            cv2.putText(display_frame, "Face Locked", (start_point[0], start_point[1] - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        else:
            cv2.putText(display_frame, "WARNING: Track Lost!", (20, 40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        cv2.imshow("Drosiness Driver - Phase 1 & 2", display_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Dọn dẹp
    cam.stop()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()