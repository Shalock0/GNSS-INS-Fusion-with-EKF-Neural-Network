import logging
from datetime import datetime


class TrainingLogger:
    def __init__(self, log_path):
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)

        # 文件Handler
        file_handler = logging.FileHandler(log_path)
        file_formatter = logging.Formatter('%(asctime)s - %(message)s')
        file_handler.setFormatter(file_formatter)

        # 控制台Handler
        console_handler = logging.StreamHandler()
        console_formatter = logging.Formatter('%(message)s')
        console_handler.setFormatter(console_formatter)

        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)

    def log(self, info_dict):
        msg = f"Epoch {info_dict['epoch']} | Batch {info_dict['batch']} | Loss: {info_dict['loss']:.4f}"
        self.logger.info(msg)