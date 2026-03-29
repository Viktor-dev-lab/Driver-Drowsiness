import os
import torch
import torch.nn as nn
from torchvision.models import vit_b_16, ViT_B_16_Weights
import torchvision.transforms as transforms
import cv2
from PIL import Image

class ViTEyeClassifier:
    def __init__(self, model_weights_path=None, device=None):
        """
        Khởi tạo mô hình Vision Transformer phân loại mắt.
        """
        # 1. Tự động nhận diện phần cứng
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        print(f"[INFO] Loading Vision Transformer on {self.device}...")
        
        # 2. Khởi tạo ViT Base
        self.model = vit_b_16(weights=ViT_B_16_Weights.DEFAULT)

        # 3. Đổi lớp Fully Connected cuối cùng
        num_ftrs = self.model.heads.head.in_features
        self.model.heads.head = nn.Linear(num_ftrs, 2)

        # 4. Xử lý đường dẫn file Weights
        if model_weights_path is None:
            # Tự động trỏ đến thư mục src/model/vit_eye_best.pth
            current_dir = os.path.dirname(os.path.abspath(__file__))
            model_weights_path = os.path.join(current_dir, "..", "model", "vit_eye_best.pth")

        # Tải trọng số đã huấn luyện
        if os.path.exists(model_weights_path):
            self.model.load_state_dict(torch.load(model_weights_path, map_location=self.device))
            print(f"[INFO] Tải thành công weights từ: {model_weights_path}")
        else:
            print(f"[WARNING] Không tìm thấy file weights tại {model_weights_path}. Model sẽ đoán bừa!")

        self.model.to(self.device)
        self.model.eval() # Bật chế độ suy luận

        # 5. Pipeline tiền xử lý ảnh
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)), 
            transforms.ToTensor(),         
            transforms.Normalize(          
                mean=[0.485, 0.456, 0.406], 
                std=[0.229, 0.224, 0.225]
            )
        ])

    def predict(self, eye_patch):
        """
        Nhận vào mảng numpy (ảnh cắt từ MediaPipe) và trả về trạng thái.
        Quy ước: 0 = Mở (Open), 1 = Nhắm (Closed)
        """
        if eye_patch is None or eye_patch.size == 0:
            return -1, 0.0 

        # ==========================================
        # XỬ LÝ DOMAIN SHIFT (Ngày -> Đêm)
        # ==========================================
        # 1. Ép ảnh màu BGR của OpenCV về ảnh xám (Grayscale)
        gray_eye = cv2.cvtColor(eye_patch, cv2.COLOR_BGR2GRAY)
        
        # 2. Chuyển ngược ảnh xám về dạng 3 kênh (RGB ảo) để ViT đọc được
        fake_rgb_eye = cv2.cvtColor(gray_eye, cv2.COLOR_GRAY2RGB)
        
        pil_img = Image.fromarray(fake_rgb_eye)

        # Tiền xử lý: Thêm chiều Batch (từ [C, H, W] thành [1, C, H, W])
        input_tensor = self.transform(pil_img).unsqueeze(0).to(self.device)

        # Suy luận bằng AI
        with torch.no_grad():
            outputs = self.model(input_tensor)
            
            # Tính xác suất bằng hàm Softmax
            probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
            confidence, predicted_class = torch.max(probabilities, 0)

        state = predicted_class.item()
        conf_score = confidence.item()

        return state, conf_score