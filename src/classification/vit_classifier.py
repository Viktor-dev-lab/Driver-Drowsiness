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
        # 1. Tự động nhận diện phần cứng (Ưu tiên GPU nếu có)
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        print(f"[INFO] Loading Vision Transformer on {self.device}...")
        
        # 2. Khởi tạo ViT Base (chia ảnh thành các patch 16x16)
        self.model = vit_b_16(weights=ViT_B_16_Weights.DEFAULT)

        # 3. Đổi lớp Fully Connected cuối cùng (Heads) từ 1000 class (ImageNet) xuống 2 class
        num_ftrs = self.model.heads.head.in_features
        self.model.heads.head = nn.Linear(num_ftrs, 2)

        # 4. Tải trọng số đã huấn luyện (Nếu bạn có file .pth / .pt tự train trên Roboflow/Colab)
        if model_weights_path:
            self.model.load_state_dict(torch.load(model_weights_path, map_location=self.device))
            print(f"[INFO] Loaded custom ViT weights from {model_weights_path}")
        else:
            print("[WARNING] No custom weights provided. Using untrained classification head.")

        self.model.to(self.device)
        self.model.eval() # Bật chế độ suy luận (tắt Dropout, BatchNorm)

        # 5. Pipeline tiền xử lý ảnh (Đưa Eye Patch về đúng chuẩn đầu vào của ViT)
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)), # Resize từ 64x64 lên 224x224
            transforms.ToTensor(),         # Chuyển thành PyTorch Tensor (0-1)
            transforms.Normalize(          # Chuẩn hóa theo phân phối của ImageNet
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
            return -1, 0.0 # Báo lỗi nếu không có ảnh mắt

        # OpenCV dùng chuẩn BGR, PyTorch và PIL dùng RGB
        eye_patch_rgb = cv2.cvtColor(eye_patch, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(eye_patch_rgb)

        # Tiền xử lý: Thêm chiều Batch (từ [C, H, W] thành [1, C, H, W])
        input_tensor = self.transform(pil_img).unsqueeze(0).to(self.device)

        # Suy luận bằng AI
        with torch.no_grad():
            outputs = self.model(input_tensor)
            
            # Tính xác suất bằng hàm Softmax
            probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
            confidence, predicted_class = torch.max(probabilities, 0)

        # Lấy giá trị int (0 hoặc 1) và độ tự tin (float)
        state = predicted_class.item()
        conf_score = confidence.item()

        return state, conf_score