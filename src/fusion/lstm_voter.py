import torch
import torch.nn as nn
import numpy as np

# ==========================================
# KIẾN TRÚC MẠNG LSTM
# ==========================================
class DrowsinessLSTM(nn.Module):
    # Đã nâng input_size lên 5 (EAR, MAR, Pitch, Yaw, ViT)
    def __init__(self, input_size=5, hidden_size=64, num_layers=2, num_classes=2):
        super(DrowsinessLSTM, self).__init__()
        # Mạng nhận đầu vào chuỗi thời gian (Batch, Sequence, Features)
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :]) # Chỉ lấy quyết định ở bước thời gian cuối cùng
        return out

# ==========================================
# BỘ XỬ LÝ LOGIC (SOFT VOTING + LSTM)
# ==========================================
class SpatiotemporalVoter:
    def __init__(self, lstm_weights_path=None, fps=15, window_size=60):
        # Lưu ý: Điều chỉnh fps ở đây (hoặc truyền vào từ main) cho sát với số FPS thực tế trên góc màn hình
        self.fps = fps
        self.window_size = window_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # --- Định nghĩa các ngưỡng (Thresholds) ---
        # 1. Ngưỡng cho Microsleep (Nhắm mắt)
        self.microsleep_frames = int(1.5 * fps)  # VD: FPS=15 -> 22 frames (duy trì 1.5 giây)
        
        # 2. Ngưỡng cho Ngủ gật (Gật gù)
        self.pitch_threshold = 20.0              # Hạ ngưỡng cúi đầu xuống 20 độ cho nhạy
        self.look_down_frames = int(0.5 * fps)   # Lọc nhiễu nhìn vô lăng (0.5 giây)
        
        # 3. Ngưỡng cho Mệt mỏi (Ngáp)
        self.mar_threshold = 0.40                # Độ há miệng
        self.yawn_frames = int(1.5 * fps)        # Ngáp thường kéo dài ít nhất 1.5 giây

        # Khởi tạo LSTM
        self.lstm_model = DrowsinessLSTM().to(self.device)
        self.use_lstm = False
        
        # Nếu có file weights LSTM thì nạp vào, nếu không thì dùng Soft Voting
        if lstm_weights_path:
            try:
                self.lstm_model.load_state_dict(torch.load(lstm_weights_path, map_location=self.device))
                self.lstm_model.eval()
                self.use_lstm = True
            except Exception as e:
                pass 

    def evaluate(self, window_data):
        # Chưa gom đủ dữ liệu tối thiểu (1 giây) thì báo trạng thái chờ
        if len(window_data) < self.fps:
            return "AWAKE (Gathering Data...)", (0, 255, 0)

        # Bóc tách Vector thành các mảng độc lập
        ears = window_data[:, 0]
        mars = window_data[:, 1]
        pitches = window_data[:, 2]
        yaws = window_data[:, 3]     # Thêm Yaw để dự phòng cho bài toán Ngủ trắng sau này
        vit_states = window_data[:, 4] # ViT lùi xuống vị trí cuối cùng

        # --------------------------------------------------
        # KỊCH BẢN 1: PHÁT HIỆN NHẮM MẮT (MICROSLEEP)
        # --------------------------------------------------
        # Nhìn vào chuỗi ViT trong 1.5 giây qua
        recent_vit = vit_states[-self.microsleep_frames:]
        is_eyes_closed = False
        
        if len(recent_vit) >= self.microsleep_frames:
            # Nếu 80% thời gian trong 1.5s qua là nhắm mắt -> Báo động
            if np.sum(recent_vit == 1) >= (self.microsleep_frames * 0.80):
                is_eyes_closed = True

        # --------------------------------------------------
        # KỊCH BẢN 2: PHÁT HIỆN GẬT GÙ (NGỦ TRẮNG) & LỌC NHIỄU
        # --------------------------------------------------
        # Nhìn vào chuỗi Pitch trong 1 giây gần nhất
        recent_pitches = pitches[-self.fps:] 
        is_nodding_off = False
        
        if len(recent_pitches) == self.fps:
            # Đếm số khung hình bị gập cổ quá ngưỡng
            pitch_down_count = np.sum(recent_pitches > self.pitch_threshold)
            
            # LỌC NHIỄU: Nếu cúi đầu vượt quá 0.5s, xác nhận là gật gù vô hồn
            if pitch_down_count > self.look_down_frames:
                is_nodding_off = True

        # --------------------------------------------------
        # KỊCH BẢN 3: PHÁT HIỆN NGÁP (YAWNING)
        # --------------------------------------------------
        # Nhìn vào chuỗi MAR trong 2 giây qua
        recent_mars = mars[-self.yawn_frames:]
        is_yawning = False
        
        if len(recent_mars) == self.yawn_frames:
            # Nếu miệng há to (MAR > threshold) trong hơn 50% thời gian của 2 giây đó
            if np.sum(recent_mars > self.mar_threshold) > (self.yawn_frames * 0.5):
                is_yawning = True

        # --------------------------------------------------
        # KỊCH BẢN 4: DỰ ĐOÁN TỪ LSTM (Hiện tại đang tắt)
        # --------------------------------------------------
        lstm_prediction = 0
        if self.use_lstm and len(window_data) == self.window_size:
            seq_tensor = torch.tensor(window_data, dtype=torch.float32).unsqueeze(0).to(self.device)
            with torch.no_grad():
                outputs = self.lstm_model(seq_tensor)
                _, predicted = torch.max(outputs, 1)
                lstm_prediction = predicted.item()

        # ==========================================
        # CHỐT TRẠNG THÁI CUỐI CÙNG
        # ==========================================
        if is_eyes_closed:
            return "DANGER: MICRO-SLEEP!", (0, 0, 255)         # ĐỎ
        elif is_nodding_off:
            return "WARNING: NODDING OFF!", (0, 165, 255)      # CAM
        elif is_yawning:
            return "CAUTION: YAWNING (FATIGUE)", (0, 255, 255) # VÀNG
        elif lstm_prediction == 1:
            return "WARNING: DROWSY PATTERN", (0, 0, 255) 
        
        return "DRIVER AWAKE", (0, 255, 0)                     # XANH LÁ