from inference import get_model

class RFDETRDetector:
    def __init__(self, model_id, api_key):
        print(f"[INFO] Loading RF-DETR Face Detection Model...")
        self.model = get_model(model_id=model_id, api_key=api_key)
        print("[INFO] RF-DETR Loaded Successfully!")

    def detect(self, frame, conf_threshold=0.5):
        results = self.model.infer(frame)[0]
        
        for prediction in results.predictions:
            if prediction.confidence > conf_threshold:
                return (prediction.x, prediction.y, prediction.width, prediction.height)
        
        return None