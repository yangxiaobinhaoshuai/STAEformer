"""交通预测指标。

数据集中 0 通常代表缺失值，因此 loss 和指标都需要忽略真实值为 0 的位置。
"""

import numpy as np
import torch


def masked_mae_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    null_value: float = 0.0,
) -> torch.Tensor:
    """供反向传播使用的 PyTorch masked MAE。"""
    if np.isnan(null_value):
        mask = ~torch.isnan(target)
    else:
        mask = target != null_value

    mask = mask.float()
    # 让 mask 的均值为 1，最终 mean 才等价于只对有效位置求平均。
    mask = mask / mask.mean()
    mask = torch.nan_to_num(mask)

    error = torch.abs(prediction - target)
    return torch.nan_to_num(error * mask).mean()


def _normalized_mask(target: np.ndarray) -> np.ndarray:
    mask = (target != 0).astype(np.float32)
    mask /= mask.mean()
    return np.nan_to_num(mask)


def rmse(target: np.ndarray, prediction: np.ndarray) -> float:
    with np.errstate(divide="ignore", invalid="ignore"):
        squared_error = np.square(prediction - target)
        return float(np.sqrt(np.mean(np.nan_to_num(squared_error * _normalized_mask(target)))))


def mae(target: np.ndarray, prediction: np.ndarray) -> float:
    with np.errstate(divide="ignore", invalid="ignore"):
        absolute_error = np.abs(prediction - target)
        return float(np.mean(np.nan_to_num(absolute_error * _normalized_mask(target))))


def mape(target: np.ndarray, prediction: np.ndarray) -> float:
    with np.errstate(divide="ignore", invalid="ignore"):
        percentage_error = np.abs((prediction - target) / target)
        masked_error = np.nan_to_num(percentage_error * _normalized_mask(target))
        return float(np.mean(masked_error) * 100)


def all_metrics(
    target: np.ndarray,
    prediction: np.ndarray,
) -> tuple[float, float, float]:
    return rmse(target, prediction), mae(target, prediction), mape(target, prediction)
