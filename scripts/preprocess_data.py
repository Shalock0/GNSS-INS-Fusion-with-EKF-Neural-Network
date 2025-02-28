import numpy as np
from utils.data_loader import load_imu_gnss_data


def time_alignment(imu_data, gnss_data):
    """时间对齐IMU(200Hz)和GNSS(1Hz)数据"""
    aligned_imu = []
    aligned_gnss = []

    # 提取GNSS时间戳（假设每秒一个）
    gnss_timestamps = np.arange(gnss_data.shape[0])

    for t in gnss_timestamps:
        # 取对应时刻前后0.5秒内的IMU数据（索引计算）
        start_idx = max(0, int((t - 0.5) * 200))
        end_idx = int((t + 0.5) * 200)
        imu_chunk = imu_data[start_idx:end_idx]

        # 计算均值作为该秒的IMU特征
        aligned_imu.append(np.mean(imu_chunk, axis=0))
        aligned_gnss.append(gnss_data[t])

    return np.array(aligned_imu), np.array(aligned_gnss)


if __name__ == "__main__":
    # 加载原始数据（示例路径）
    imu_raw, gnss_raw = load_imu_gnss_data("data/raw/ICM20602.txt", "data/raw/gnss_pos.txt")

    # 时间对齐处理
    imu_aligned, gnss_aligned = time_alignment(imu_raw, gnss_raw)

    # 保存处理后的数据
    np.save("data/processed/aligned_imu.npy", imu_aligned)
    np.save("data/processed/aligned_gnss.npy", gnss_aligned)