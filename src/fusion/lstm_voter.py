import torch
import torch.nn as nn
import numpy as np

# ==========================================
# 1. KIẾN TRÚC MẠNG CHUỖI THỜI GIAN LSTM
# ==========================================
class DrowsinessLSTM(nn.Module):
    def __init__(self, input_size=4, hidden_size=64, num_layers=2, num_classes=2):
        super(DrowsinessLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        # Mạng LSTM nhận đầu vào: (Batch, Sequence_length, Features)
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        # Lớp Linear cuối cùng phân loại 0 (Awake) hoặc 1 (Drowsy)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out, _ = self.lstm(x)
        # Chỉ lấy output của timestep cuối cùng (frame thứ 60) để dự đoán
        out = self.fc(out[:, -1, :])
        return out

# ==========================================
# 2. BỘ ĐIỀU PHỐI: SOFT VOTING KẾT HỢP LSTM
# ==========================================
class SpatiotemporalVoter:
    def __init__(self, lstm_weights_path=None, fps=30, window_size=60):
        self.fps = fps
        self.window_size = window_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # --- Thiết lập Ngưỡng Thời gian (Lọc Nhiễu) ---
        self.pitch_threshold = 25.0
        self.microsleep_frames = int(1.5 * fps)  # ~45 frames (1.5s nhắm mắt)
        self.look_down_frames = int(0.5 * fps)   # ~15 frames (0.5s nhìn vô lăng)

        # --- Khởi tạo LSTM ---
        self.lstm_model = DrowsinessLSTM().to(self.device)
        self.use_lstm = False
        
        if lstm_weights_path:
            try:
                self.lstm_model.load_state_dict(torch.load(lstm_weights_path, map_location=self.device))
                self.lstm_model.eval()
                self.use_lstm = True
                print("[INFO] LSTM model loaded successfully.")
            except Exception as e:
                print(f"[WARNING] Could not load LSTM. Using Soft Voting fallback: {e}")

    def evaluate(self, window_data):
        """
        Nhận vào mảng dữ liệu cửa sổ trượt và đưa ra phán quyết cuối cùng.
        """
        # Cần thu thập đủ ít nhất 1 giây dữ liệu mới bắt đầu phán xét
        if len(window_data) < self.fps:
            return "AWAKE (Gathering Data...)", (0, 255, 0)

        # Trích xuất các cột dữ liệu
        vit_states = window_data[:, 3] # Cột ViT (0: Mở, 1: Nhắm)
        pitches = window_data[:, 2]    # Cột Góc cúi đầu

        # --------------------------------------------------
        # LUẬT A: NHẮM MẮT (Lọc nháy mắt)
        # --------------------------------------------------
        recent_vit = vit_states[-self.microsleep_frames:]
        is_eyes_closed = False
        if len(recent_vit) >= self.microsleep_frames:
            # Nếu trong 1.5s qua, > 80% số khung hình là mắt nhắm -> Đích thị là ngủ
            if np.sum(recent_vit == 1) >= (self.microsleep_frames * 0.80):
                is_eyes_closed = True

        # --------------------------------------------------
        # LUẬT B: GẬT GÙ / NGỦ TRẮNG (Lọc hành vi nhìn vô lăng)
        # --------------------------------------------------
        recent_pitches = pitches[-self.fps:] # Xét trong 1s gần nhất
        is_nodding_off = False
        if len(recent_pitches) == self.fps:
            pitch_down_count = np.sum(recent_pitches > self.pitch_threshold)
            # Chỉ báo động nếu cúi đầu vượt quá 0.5 giây. Nếu < 0.5s thì coi là đang liếc nhìn màn hình xe.
            if pitch_down_count > self.look_down_frames:
                is_nodding_off = True

        # --------------------------------------------------
        # LSTM INFERENCE (Nếu có weights)
        # --------------------------------------------------
        lstm_prediction = 0
        if self.use_lstm and len(window_data) == self.window_size:
            seq_tensor = torch.tensor(window_data, dtype=torch.float32).unsqueeze(0).to(self.device)
            with torch.no_grad():
                outputs = self.lstm_model(seq_tensor)
                _, predicted = torch.max(outputs, 1)
                lstm_prediction = predicted.item()

        # ==========================================
        # CHỐT TRẠNG THÁI & MÀU SẮC (Soft Voting)
        # ==========================================
        if is_eyes_closed:
            return "DANGER: MICRO-SLEEP!", (0, 0, 255) # ĐỎ (Ngủ gật)
        elif is_nodding_off:
            return "WARNING: NODDING OFF!", (0, 165, 255) # CAM (Ngủ trắng)
        elif lstm_prediction == 1:
            return "WARNING: DROWSY PATTERN", (0, 0, 255) # AI phát hiện quy luật mệt mỏi
        
        return "DRIVER AWAKE", (0, 255, 0) # XANH LÁ