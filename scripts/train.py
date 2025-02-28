import torch
from torch.utils.data import DataLoader
from models.res_emsa import ResEMSAModel
from models.ekf import ExtendedKalmanFilter
from utils.data_loader import GNSSINSDataset
from utils.logger import TrainingLogger

# 配置参数
config = {
    'batch_size': 32,
    'lr': 0.001,
    'epochs': 100,
    'imu_dim': 6,
    'gnss_dim': 3
}

# 初始化
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ResEMSAModel(imu_dim=6, gnss_dim=3).to(device)
ekf = ExtendedKalmanFilter()
optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])
criterion = torch.nn.SmoothL1Loss()
logger = TrainingLogger("outputs/logs/train.log")

# 数据加载
dataset = GNSSINSDataset("data/processed/aligned_imu.npy",
                         "data/processed/aligned_gnss.npy")
dataloader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=True)

# 训练循环
for epoch in range(config['epochs']):
    for batch_idx, (imu, gnss, target) in enumerate(dataloader):
        imu, gnss = imu.to(device), gnss.to(device)

        # EKF预测步骤
        ekf.predict(imu)

        # 神经网络生成修正
        delta = model(imu, gnss)

        # EKF更新步骤
        ekf.update(gnss + delta.detach().cpu().numpy())

        # 计算损失
        loss = criterion(ekf.get_position(), target)

        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 日志记录
        logger.log({
            'epoch': epoch,
            'batch': batch_idx,
            'loss': loss.item()
        })

    # 保存模型
    if epoch % 10 == 0:
        torch.save(model.state_dict(), f"outputs/checkpoints/epoch_{epoch}.pth")