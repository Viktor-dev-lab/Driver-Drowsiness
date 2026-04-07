import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler 
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
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

# Tạo thư mục lưu kết quả hình ảnh
RESULTS_DIR = os.path.join("src", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Bắt đầu huấn luyện trên thiết bị: {device.type.upper()}")

# ==========================================
# 2. XỬ LÝ DỮ LIỆU & CẮT CHUỖI THEO VIDEO
# ==========================================
def create_sequences(df, seq_length, scaler=None):
    sequences, labels, seq_vids = [], [], [] 
    features = ['ear', 'mar', 'pitch', 'yaw', 'vit_state']
    
    df = df.dropna()
    X_raw = df[features].values 
    
    if scaler is None:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_raw)
        scaler_path = os.path.join("src", "model", "lstm_scaler.pkl")
        os.makedirs(os.path.dirname(scaler_path), exist_ok=True)
        with open(scaler_path, 'wb') as f:
            pickle.dump(scaler, f)
    else:
        X_scaled = scaler.transform(X_raw)

    df_scaled = df.copy()
    df_scaled[features] = X_scaled
    
    grouped = df_scaled.groupby('video_id')
    print(f"[INFO] Đang tiến hành trượt cửa sổ (Sliding Window) trên {len(grouped)} video độc lập...")
    
    for video_id, group in grouped:
        group_features = group[features].values
        group_labels = group['label'].values
        
        if len(group_features) < seq_length:
            continue
            
        for i in range(len(group_features) - seq_length):
            seq = group_features[i : i + seq_length]
            label = 1 if np.sum(group_labels[i : i + seq_length]) >= (seq_length * 0.6) else 0
            
            sequences.append(seq)
            labels.append(label)
            seq_vids.append(video_id) 
            
    return np.array(sequences, dtype=np.float32), np.array(labels, dtype=np.int64), scaler, np.array(seq_vids)

class LSTMDataset(Dataset):
    def __init__(self, sequences, labels):
        self.X, self.y = torch.tensor(sequences), torch.tensor(labels)
    def __len__(self): return len(self.X)
    def __getitem__(self, idx): return self.X[idx], self.y[idx]

print("[INFO] Đang nạp dữ liệu từ CSV...")
df = pd.read_csv(CSV_PATH)
X_seq, y_seq, fitted_scaler, vid_seq = create_sequences(df, SEQ_LENGTH)
print(f"[INFO] Tổng số chuỗi gốc tạo được: {len(X_seq)} chuỗi.")

# ==========================================
# 3. LÀM LỆCH DATA & CHIA TRAIN/VAL/TEST THEO TÊN VIDEO
# ==========================================
print("\n[INFO] Đang xử lý làm lệch dữ liệu và chia tập Train/Val/Test...")

awake_idx = np.where(y_seq == 0)[0]
drowsy_idx = np.where(y_seq == 1)[0]

np.random.seed(42)
keep_drowsy_size = int(len(drowsy_idx) * 0.15) 
drowsy_idx_sampled = np.random.choice(drowsy_idx, size=keep_drowsy_size, replace=False)

imbalanced_idx = np.concatenate([awake_idx, drowsy_idx_sampled])

X_seq_imb = X_seq[imbalanced_idx]
y_seq_imb = y_seq[imbalanced_idx]
vid_seq_imb = vid_seq[imbalanced_idx]

print(f"[INFO] Tỉ lệ sau khi ép lệch - Awake (0): {len(awake_idx)} mẫu | Drowsy (1): {keep_drowsy_size} mẫu")

unique_vids = np.unique(vid_seq_imb)
drowsy_vids = [v for v in unique_vids if v.startswith('d_')]
awake_vids = [v for v in unique_vids if not v.startswith('d_')]

np.random.shuffle(drowsy_vids)
np.random.shuffle(awake_vids)

def split_vids(lst):
    n = len(lst)
    return lst[:int(n*0.7)], lst[int(n*0.7):int(n*0.85)], lst[int(n*0.85):]

train_d, val_d, test_d = split_vids(drowsy_vids)
train_a, val_a, test_a = split_vids(awake_vids)

train_vids = set(train_d + train_a)
val_vids = set(val_d + val_a)
test_vids = set(test_d + test_a)

train_idx = [i for i, vid in enumerate(vid_seq_imb) if vid in train_vids]
val_idx = [i for i, vid in enumerate(vid_seq_imb) if vid in val_vids]
test_idx = [i for i, vid in enumerate(vid_seq_imb) if vid in test_vids]

print(f"[INFO] Số lượng chuỗi - Train: {len(train_idx)} | Val: {len(val_idx)} | Test: {len(test_idx)}")

train_loader = DataLoader(LSTMDataset(X_seq_imb[train_idx], y_seq_imb[train_idx]), batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(LSTMDataset(X_seq_imb[val_idx], y_seq_imb[val_idx]), batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(LSTMDataset(X_seq_imb[test_idx], y_seq_imb[test_idx]), batch_size=BATCH_SIZE, shuffle=False)

# ==========================================
# 4. ĐỊNH NGHĨA MẠNG LSTM
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
# 5. VÒNG LẶP HUẤN LUYỆN
# ==========================================
save_path = os.path.join("src", "model", "lstm_drowsiness_best.pth")
os.makedirs(os.path.dirname(save_path), exist_ok=True) 

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
# 6. VẼ & LƯU BIỂU ĐỒ QUÁ TRÌNH HUẤN LUYỆN
# ==========================================
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Validation Loss')
plt.title('Đồ thị Loss (Train vs Val)')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(val_accuracies, label='Validation Accuracy', color='green')
plt.title('Đồ thị Độ chính xác (Validation)')
plt.xlabel('Epochs')
plt.ylabel('Accuracy (%)')
plt.legend()

plt.tight_layout()

# LƯU FILE HÌNH ẢNH TRƯỚC KHI SHOW
train_plot_path = os.path.join(RESULTS_DIR, "training_history.png")
plt.savefig(train_plot_path, dpi=300, bbox_inches='tight')
print(f"[INFO] Đã xuất file biểu đồ huấn luyện tại: {train_plot_path}")

plt.show() 

# ==========================================
# 7. ĐÁNH GIÁ CHI TIẾT TRÊN TẬP TEST
# ==========================================
print("\n[INFO] Đang tiến hành đánh giá chi tiết trên tập TEST (Dữ liệu chưa từng thấy)...")

model.load_state_dict(torch.load(save_path))
model.eval()

all_labels = []
all_preds = []
all_probs = []

with torch.no_grad():
    for inputs, labels in test_loader: 
        inputs = inputs.to(device)
        outputs = model(inputs)
        
        probs = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()
        _, predicted = torch.max(outputs, 1)
        
        all_labels.extend(labels.numpy())
        all_preds.extend(predicted.cpu().numpy())
        all_probs.extend(probs)

# --- A. CLASSIFICATION REPORT ---
print("\n--- BÁO CÁO PHÂN LOẠI TRÊN TẬP TEST ---")
report = classification_report(all_labels, all_preds, target_names=["Awake (0)", "Drowsy (1)"])
print(report)

# --- B. VẼ & LƯU CONFUSION MATRIX & ROC CURVE ---
plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
cm = confusion_matrix(all_labels, all_preds)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=["Awake (0)", "Drowsy (1)"], 
            yticklabels=["Awake (0)", "Drowsy (1)"],
            annot_kws={"size": 14})
plt.title("Confusion Matrix (Test Set)", fontsize=14, fontweight='bold')
plt.xlabel("Predicted Label (Dự đoán)", fontsize=12)
plt.ylabel("True Label (Thực tế)", fontsize=12)

plt.subplot(1, 2, 2)
fpr, tpr, thresholds = roc_curve(all_labels, all_probs)
roc_auc = auc(fpr, tpr)

plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--') 
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate (FPR)', fontsize=12)
plt.ylabel('True Positive Rate (TPR)', fontsize=12)
plt.title('ROC Curve (Test Set)', fontsize=14, fontweight='bold')
plt.legend(loc="lower right", fontsize=12)
plt.grid(alpha=0.3)

plt.tight_layout()

# LƯU FILE HÌNH ẢNH TRƯỚC KHI SHOW
eval_plot_path = os.path.join(RESULTS_DIR, "evaluation_metrics.png")
plt.savefig(eval_plot_path, dpi=300, bbox_inches='tight')
print(f"[INFO] Đã xuất file biểu đồ đánh giá tại: {eval_plot_path}")

plt.show()