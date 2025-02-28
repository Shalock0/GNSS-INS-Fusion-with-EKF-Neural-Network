import numpy as np


class ExtendedKalmanFilter:
    def __init__(self, dt=1.0):
        # 初始化状态向量 [x, y, z, vx, vy, vz, bgx, bgy, bgz, bax, bay, baz]
        self.state = np.zeros(12)
        self.dt = dt  # GNSS更新间隔（秒）

        # 初始化协方差矩阵
        self.P = np.eye(12) * 0.1

        # 过程噪声协方差
        self.Q = np.diag([
            0.1, 0.1, 0.1,  # 位置噪声
            0.3, 0.3, 0.3,  # 速度噪声
            0.01, 0.01, 0.01,  # 陀螺零偏噪声
            0.05, 0.05, 0.05  # 加速度计零偏噪声
        ])

        # 观测矩阵 (仅观测位置)
        self.H = np.zeros((3, 12))
        self.H[0:3, 0:3] = np.eye(3)

    def predict(self, imu_data):
        """IMU数据驱动预测"""
        # 解析IMU数据 (假设已经转换为ENU坐标系)
        gyro = imu_data[:3]  # 角速度 (rad/s)
        acc = imu_data[3:]  # 加速度 (m/s²)

        # 状态转移矩阵F
        F = np.eye(12)
        F[0:3, 3:6] = np.eye(3) * self.dt
        F[3:6, 6:9] = -self._skew_symmetric(acc) * self.dt
        F[3:6, 9:12] = np.eye(3) * self.dt

        # 过程噪声矩阵G
        G = np.zeros((12, 6))
        G[3:6, 0:3] = np.eye(3) * self.dt
        G[6:9, 3:6] = np.eye(3) * self.dt

        # 状态预测
        self.state = F @ self.state
        self.P = F @ self.P @ F.T + G @ self.Q @ G.T

    def update(self, z, R):
        """观测更新"""
        # 计算卡尔曼增益
        S = self.H @ self.P @ self.H.T + R
        K = self.P @ self.H.T @ np.linalg.inv(S)

        # 更新状态
        y = z - self.H @ self.state
        self.state = self.state + K @ y
        self.P = (np.eye(12) - K @ self.H) @ self.P

    def get_position(self):
        return self.state[0:3]

    def _skew_symmetric(self, v):
        """生成反对称矩阵"""
        return np.array([
            [0, -v[2], v[1]],
            [v[2], 0, -v[0]],
            [-v[1], v[0], 0]
        ])