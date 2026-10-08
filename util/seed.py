from __future__ import annotations

import os
import random

import torch


def seed_everything(seed: int) -> None:
    # 固定 Python、哈希和 PyTorch 随机源，提升实验可复现性。
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # 让 cuDNN 使用确定性实现，牺牲少量速度换取稳定复现实验结果。
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
