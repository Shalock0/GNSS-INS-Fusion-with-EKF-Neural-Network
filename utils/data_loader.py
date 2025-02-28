import numpy as np
import torch
from torch.utils.data import Dataset


def load_imu_gnss_data(imu_path, gnss_path):
    """加载原始IMU和GNSS数据"""
    # 加载IMU数据 (时间戳, dθ_x, dθ_y, dθ_z, dv_x, dv_y, dv_z)
    imu_data = np.loadtxt(imu_path, delimiter=',')
    # 加载GNSS数据 (时间戳, lat, lon, alt)
    gnss_data = np.loadtxt(gnss_path, delimiter=',')
    return imu_data, gnss_data


class GNSSINSDataset(Dataset):
    def __init__(self, imu_path, gnss_path, window_size=4, stride=1):
        """
        imu_path: 预处理后的IMU数据路径(.npy)
        gnss_path: 预处理后的GNSS数据路径(.npy)
        window_size: 时间窗口大小（秒）
        stride: 滑动步长（秒）
        """
        self.imu_data = np.load(imu_path)  # (N, 6)
        self.gnss_data = np.load(gnss_path)  # (N, 3)
        self.window_size = window_size
        self.stride = stride
        self.sample_indices = self._generate_indices()

    def _generate_indices(self):
        indices = []
        n_samples = len(self.imu_data)
        for i in range(0, n_samples - self.window_size, self.stride):
            if i + self.window_size <= n_samples:
                indices.append(i)
        return indices

    def __len__(self):
        return len(self.sample_indices)

    def __getitem__(self, idx):
        start = self.sample_indices[idx]
        end = start + self.window_size

        # 提取窗口数据
        imu_window = self.imu_data[start:end]  # (T, 6)
        gnss_window = self.gnss_data[start:end]  # (T, 3)

        # 真值：窗口中心点位置
        target = self.gnss_data[start + self.window_size // 2]

        return (
            torch.FloatTensor(imu_window),  # (T, 6)
            torch.FloatTensor(gnss_window),  # (T, 3)
            torch.FloatTensor(target)  # (3,)
        )

    @staticmethod
    def collate_fn(batch):
        imu = torch.stack([item[0] for item in batch])  # (B, T, 6)
        gnss = torch.stack([item[1] for item in batch])  # (B, T, 3)
        target = torch.stack([item[2] for item in batch])  # (B, 3)
        return imu, gnss, target