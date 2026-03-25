import numpy as np
from collections import deque

class SlidingWindow:
    def __init__(self, window_size=60):
        """
        Khởi tạo cửa sổ trượt. Với camera 30FPS, 60 frames tương đương 2 giây.
        """
        self.window_size = window_size
        # Hàng đợi tự động đẩy phần tử cũ ra khi vượt quá maxlen
        self.data_queue = deque(maxlen=window_size)

    def add_data(self, ear, mar, pitch, vit_state):
        """
        Thêm Vector đặc trưng của frame hiện tại vào hàng đợi.
        Format: [EAR, MAR, Pitch, ViT_State]
        """
        self.data_queue.append([ear, mar, pitch, vit_state])

    def get_window(self):
        """
        Trả về toàn bộ dữ liệu trong cửa sổ dưới dạng ma trận Numpy (shape: N x 4).
        """
        return np.array(self.data_queue)

    def is_full(self):
        return len(self.data_queue) == self.window_size