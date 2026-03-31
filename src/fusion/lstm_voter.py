import torch
import torch.nn as nn
import numpy as np
import pickle
import os

# ==========================================
# KIẾN TRÚC MẠNG LSTM
# ==========================================
class DrowsinessLSTM(nn.Module):
    def __init__(self, input_size=5, hidden_size=64, num_layers=2, num_classes=2):
        super(DrowsinessLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :]) 
        return out

# ==========================================
# BỘ XỬ LÝ LOGIC (SOFT VOTING + LSTM)
# ==========================================
class SpatiotemporalVoter:
    def __init__(self, lstm_weights_path=None, fps=15, window_size=30):
        self.fps = fps
        self.window_size = window_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # --- ĐÃ CHỈNH SỬA: Tối ưu các ngưỡng Soft Voting ---
        # Nới lỏng một chút để hệ thống dễ bắt lỗi hơn
        self.microsleep_frames = int(1.0 * fps)  # Giảm từ 1.5s xuống 1.0s (Chỉ cần nhắm 1s là báo)
        self.pitch_threshold = 15.0              # Giảm từ 20 độ xuống 15 độ (Chỉ cần cúi nhẹ là bắt)
        self.look_down_frames = int(0.3 * fps)   # Chỉ cần cúi 0.3s liên tục
        self.mar_threshold = 0.35                # Hạ ngưỡng há miệng xuống 0.35
        self.yawn_frames = int(1.0 * fps)        # Ngáp 1s là đủ để cảnh báo

        # Khởi tạo LSTM và Scaler
        self.lstm_model = DrowsinessLSTM().to(self.device)
        self.scaler = None
        self.use_lstm = False
        
        # Nạp tệp trọng số (Weights) và Scaler
        if lstm_weights_path:
            try:
                self.lstm_model.load_state_dict(torch.load(lstm_weights_path, map_location=self.device))
                self.lstm_model.eval()
                
                scaler_path = os.path.join(os.path.dirname(lstm_weights_path), "lstm_scaler.pkl")
                if os.path.exists(scaler_path):
                    with open(scaler_path, 'rb') as f:
                        self.scaler = pickle.load(f)
                    self.use_lstm = True
                    print(f"[INFO] Bộ Voter đã nạp thành công LSTM và Scaler.")
                else:
                    print(f"[ERROR] Bị thiếu file Scaler tại {scaler_path}")
            except Exception as e:
                print(f"[WARNING] Không nạp được LSTM. Lý do: {e}")

    def evaluate(self, window_data):
        if len(window_data) < self.fps:
            return "AWAKE (Gathering Data...)", (0, 255, 0)

        ears = window_data[:, 0]
        mars = window_data[:, 1]
        pitches = window_data[:, 2]
        yaws = window_data[:, 3]
        vit_states = window_data[:, 4]

        # --------------------------------------------------
        # KỊCH BẢN 1: MICROSLEEP (Nhắm mắt)
        # --------------------------------------------------
        recent_vit = vit_states[-self.microsleep_frames:]
        is_eyes_closed = False
        if len(recent_vit) >= self.microsleep_frames:
            # 70% thời gian trong 1s là nhắm mắt -> Nhắm
            if np.sum(recent_vit == 1) >= (self.microsleep_frames * 0.70):
                is_eyes_closed = True

        # --------------------------------------------------
        # KỊCH BẢN 2: GẬT GÙ (Ngủ gật)
        # --------------------------------------------------
        recent_pitches = pitches[-self.fps:] 
        is_nodding_off = False
        if len(recent_pitches) == self.fps:
            pitch_down_count = np.sum(recent_pitches > self.pitch_threshold)
            if pitch_down_count > self.look_down_frames:
                is_nodding_off = True

        # --------------------------------------------------
        # KỊCH BẢN 3: NGÁP
        # --------------------------------------------------
        recent_mars = mars[-self.yawn_frames:]
        is_yawning = False
        if len(recent_mars) == self.yawn_frames:
            # Há miệng 50% thời gian trong 1s -> Ngáp
            if np.sum(recent_mars > self.mar_threshold) > (self.yawn_frames * 0.5):
                is_yawning = True

        # --------------------------------------------------
        # KỊCH BẢN 4: DỰ ĐOÁN TỪ MẠNG LSTM (SIẾT CHẶT)
        # --------------------------------------------------
        lstm_prediction = 0
        if self.use_lstm and len(window_data) == self.window_size:
            scaled_data = self.scaler.transform(window_data)
            seq_tensor = torch.tensor(scaled_data, dtype=torch.float32).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                outputs = self.lstm_model(seq_tensor)
                
                # Áp dụng Softmax để lấy xác suất cụ thể từ 0 đến 1
                probabilities = torch.softmax(outputs, dim=1)
                prob_drowsy = probabilities[0][1].item()
                
                # NGƯỠNG AN TOÀN: LSTM phải chắc chắn trên 85% mới được báo động
                if prob_drowsy > 0.85:
                    lstm_prediction = 1

        # ==========================================
        # CHỐT TRẠNG THÁI CUỐI CÙNG (ƯU TIÊN)
        # ==========================================
        # Ưu tiên 1: Các hành vi sinh học rõ ràng (Soft Voting)
        if is_eyes_closed:
            return "DANGER: MICRO-SLEEP!", (0, 0, 255)         
        elif is_nodding_off:
            return "WARNING: NODDING OFF!", (0, 165, 255)      
        elif is_yawning:
            return "CAUTION: YAWNING (FATIGUE)", (0, 255, 255) 
            
        # Ưu tiên 2: Trạng thái lơ đãng vô hình (LSTM)
        elif lstm_prediction == 1:
            return "DANGER: WHITE SLEEP (TRANCE)!", (255, 0, 255) 
        
        return "DRIVER AWAKE", (0, 255, 0)