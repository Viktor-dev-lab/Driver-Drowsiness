import numpy as np
from collections import deque

class SlidingWindow:
    def __init__(self, window_size=60):
        self.window_size = window_size
        self.data_queue = deque(maxlen=window_size)

    # ĐÃ SỬA: Thêm 'yaw' vào tham số truyền vào
    def add_data(self, ear, mar, pitch, yaw, vit_state):
        # ĐÃ SỬA: Lưu mảng 5 phần tử
        self.data_queue.append([ear, mar, pitch, yaw, vit_state])

    def get_window(self):
        return np.array(self.data_queue)

    def is_full(self):
        return len(self.data_queue) == self.window_size