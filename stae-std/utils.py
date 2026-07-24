"""实验环境和日志工具。"""

import os
import random
from pathlib import Path

import numpy as np
import torch


def seed_everything(seed: int) -> None:
    """固定常见随机源，便于学习时复现实验。"""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class Logger:
    """同时打印到终端和日志文件的最小 logger。"""

    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.file = path.open("w", encoding="utf-8")

    def write(self, message: str = "") -> None:
        print(message)
        print(message, file=self.file, flush=True)

    def close(self) -> None:
        self.file.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
