import torch
import numpy as np
from utils.data_loader import GNSSINSDataset
from utils.visualization import plot_trajectory


def evaluate(model_path, data_path):
    # 加载模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.load(model_path).to(device)
    model.eval()

    # 加载数据
    dataset = GNSSINSDataset(
        f"{data_path}/processed/aligned_imu.npy",
        f"{data_path}/processed/aligned_gnss.npy"
    )

    # 存储结果
    gt_positions = []
    pred_positions = []

    with torch.no_grad():
        for imu, gnss, target in dataset:
            imu = imu.unsqueeze(0).to(device)
            gnss = gnss.unsqueeze(0).to(device)

            # 神经网络修正
            delta = model(imu, gnss)
            corrected_gnss = gnss[0, -1].cpu().numpy() + delta.squeeze().cpu().numpy()

            # 记录结果
            gt_positions.append(target.numpy())
            pred_positions.append(corrected_gnss)

    # 转换为NumPy数组
    gt = np.array(gt_positions)
    pred = np.array(pred_positions)

    # 计算误差
    rmse = np.sqrt(np.mean((gt - pred) ** 2, axis=0))
    print(f"RMSE Errors (X/Y/Z): {rmse}")

    # 可视化
    plot_trajectory(gt, pred, "Evaluation Result")


if __name__ == "__main__":
    evaluate(
        model_path="outputs/checkpoints/best_model.pth",
        data_path="data"
    )