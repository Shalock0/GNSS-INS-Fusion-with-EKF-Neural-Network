import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1),
                nn.BatchNorm1d(out_channels)
            )

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(residual)
        return F.relu(out)


class EMSA(nn.Module):
    def __init__(self, dim, heads=8):
        super().__init__()
        self.heads = heads
        self.scale = dim ​ ** ​ -0.5
        self.to_qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        B, T, C = x.shape
        qkv = self.to_qkv(x).reshape(B, T, 3, self.heads, C // self.heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(B, T, C)
        return self.proj(out)


class ResEMSAModel(nn.Module):
    def __init__(self, imu_dim=6, gnss_dim=3, output_dim=3):
        super().__init__()
        # IMU特征提取
        self.imu_resnet = nn.Sequential(
            ResBlock(imu_dim, 64),
            ResBlock(64, 128),
            ResBlock(128, 256)
        )
        # GNSS特征提取
        self.gnss_resnet = nn.Sequential(
            ResBlock(gnss_dim, 64),
            ResBlock(64, 128),
            ResBlock(128, 256)
        )
        # 融合与回归
        self.emsa = EMSA(512)
        self.regressor = nn.Linear(512, output_dim)

    def forward(self, imu, gnss):
        # 输入形状: (B, T, C)
        imu_feat = self.imu_resnet(imu.permute(0, 2, 1)).permute(0, 2, 1)
        gnss_feat = self.gnss_resnet(gnss.permute(0, 2, 1)).permute(0, 2, 1)
        fused = torch.cat([imu_feat, gnss_feat], dim=-1)
        attn_out = self.emsa(fused)
        return self.regressor(attn_out[:, -1, :])  # 取最后时间步输出