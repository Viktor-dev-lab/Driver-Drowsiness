import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler # <--- THƯ VIỆN CHUẨN HÓA MỚI
import matplotlib.pyplot as plt
import os
import pickle

# ==========================================
# 1. CẤU HÌNH THÔNG SỐ HUẤN LUYỆN
# ==========================================
CSV_PATH = "dataset_lstm.csv"
SEQ_LENGTH = 30        
BATCH_SIZE = 64        
EPOCHS = 30            
LEARNING_RATE = 0.001

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Bắt đầu huấn luyện trên thiết bị: {device.type.upper()}")

# ==========================================
# 2. XỬ LÝ DỮ LIỆU & CHUẨN HÓA
# ==========================================
def create_sequences(df, seq_length, scaler=None):
    sequences = []
    labels = []
    features = ['ear', 'mar', 'pitch', 'yaw', 'vit_state']
    
    df = df.dropna()
    
    # --------------------------------------------------
    # CHUẨN HÓA DỮ LIỆU: Đưa tất cả về cùng một thang đo
    # --------------------------------------------------
    # Trích xuất riêng phần đặc trưng
    X_raw = df[features].values 
    
    # Huấn luyện Scaler nếu chưa có
    if scaler is None:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_raw)
        # Bắt buộc phải lưu lại Scaler này để lúc dùng cam thật còn gọi ra dùng
        scaler_path = os.path.join("src", "model", "lstm_scaler.pkl")
        os.makedirs(os.path.dirname(scaler_path), exist_ok=True)
        with open(scaler_path, 'wb') as f:
            pickle.dump(scaler, f)
        print(f"[INFO] Đã tạo và lưu StandardScaler tại: {scaler_path}")
    else:
        # Nếu đã có, chỉ áp dụng để transform
        X_scaled = scaler.transform(X_raw)

    lbls = df['label'].values
    print(f"[INFO] Tổng số frame gốc trong CSV: {len(X_scaled)}")
    
    # Tiến hành cắt khối (Sliding Window) trên dữ liệu ĐÃ CHUẨN HÓA
    for i in range(len(X_scaled) - seq_length):
        seq = X_scaled[i : i + seq_length] # <-- Dùng X_scaled thay vì data gốc
        label = 1 if np.sum(lbls[i : i + seq_length]) >= (seq_length * 0.6) else 0
        
        sequences.append(seq)
        labels.append(label)
        
    return np.array(sequences, dtype=np.float32), np.array(labels, dtype=np.int64), scaler

class LSTMDataset(Dataset):
    def __init__(self, sequences, labels):
        self.X = torch.tensor(sequences)
        self.y = torch.tensor(labels)
        
    def __len__(self): return len(self.X)
    def __getitem__(self, idx): return self.X[idx], self.y[idx]

print("[INFO] Đang nạp, chuẩn hóa và cắt chuỗi...")
df = pd.read_csv(CSV_PATH)

# Lấy X_seq, y_seq và đối tượng scaler
X_seq, y_seq, fitted_scaler = create_sequences(df, SEQ_LENGTH)

print(f"[INFO] Đã tạo thành công {len(X_seq)} chuỗi dữ liệu. Kích thước Tensor: {X_seq.shape}")

# Trộn và chia Tập Train/Val
indices = np.random.permutation(len(X_seq))
split_idx = int(len(X_seq) * 0.8)
train_idx, val_idx = indices[:split_idx], indices[split_idx:]

train_loader = DataLoader(LSTMDataset(X_seq[train_idx], y_seq[train_idx]), batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(LSTMDataset(X_seq[val_idx], y_seq[val_idx]), batch_size=BATCH_SIZE, shuffle=False)

# ==========================================
# 3. ĐỊNH NGHĨA MẠNG LSTM
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

model = DrowsinessLSTM().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)

# ==========================================
# 4. VÒNG LẶP HUẤN LUYỆN
# ==========================================
save_path = os.path.join("src", "model", "lstm_drowsiness_best.pth")
best_val_loss = float('inf')
train_losses, val_losses, val_accuracies = [], [], []

print("\n[🚀] BẮT ĐẦU HUẤN LUYỆN LSTM...")
for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        
    avg_train_loss = running_loss / len(train_loader)
    train_losses.append(avg_train_loss)
    
    model.eval()
    val_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            val_loss += criterion(outputs, labels).item()
            
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
    avg_val_loss = val_loss / len(val_loader)
    val_losses.append(avg_val_loss)
    
    acc = 100 * correct / total
    val_accuracies.append(acc)
    
    print(f"Epoch [{epoch+1:02d}/{EPOCHS}] | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val Acc: {acc:.2f}%", end="")
    
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save(model.state_dict(), save_path)
        print(" 💾 (Đã lưu Model tốt hơn!)")
    else:
        print("")

print(f"\n[SUCCESS] Huấn luyện xong! Trọng số đã lưu tại: {save_path}")

# ==========================================
# 5. VẼ BIỂU ĐỒ TRỰC QUAN
# ==========================================
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Validation Loss')
plt.title('Đồ thị Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(val_accuracies, label='Validation Accuracy', color='green')
plt.title('Đồ thị Độ chính xác')
plt.xlabel('Epochs')
plt.ylabel('Accuracy (%)')
plt.legend()

plt.tight_layout()
plt.show()