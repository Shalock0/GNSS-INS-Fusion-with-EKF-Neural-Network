# GNSS-INS-Fusion-with-EKF-Neural-Network
GNSS-INS-Fusion/
│
├── configs/                  # 配置文件
│   ├── train_config.yaml     # 训练超参数配置
│   └── model_config.yaml     # 模型结构配置
│
├── data/                     # 数据目录
│   ├── raw/                  # 原始数据（从数据集下载）
│   │   ├── ICM20602.txt      # IMU原始数据（200Hz）
│   │   └── gnss_pos.txt      # GNSS位置数据（1Hz）
│   │
│   └── processed/            # 预处理后的数据
│       ├── aligned_imu.npy   # 时间对齐后的IMU数据
│       └── aligned_gnss.npy  # 时间对齐后的GNSS数据
│
├── models/                   # 模型定义
│   ├── res_emsa.py           # Res-EMSA网络模型
│   └── ekf.py                # EKF实现
│
├── utils/                    # 工具函数
│   ├── data_loader.py        # 数据加载与预处理
│   ├── visualization.py      # 结果可视化
│   └── logger.py             # 训练日志记录
│
├── scripts/                  # 执行脚本
│   ├── preprocess_data.py    # 数据预处理脚本
│   ├── train.py              # 模型训练脚本
│   └── evaluate.py           # 模型评估脚本
│
├── outputs/                  # 输出结果
│   ├── checkpoints/          # 模型权重保存
│   ├── logs/                 # 训练日志
│   └── results/              # 评估结果与图表
│
└── requirements.txt          # 依赖库列表