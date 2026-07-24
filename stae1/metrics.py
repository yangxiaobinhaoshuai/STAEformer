import numpy as np
import torch


def masked_mae_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    null_value: float = 0.0,
) -> torch.Tensor:

    if np.isnan(null_value):
        mask = ~torch.isnan(target)
    else:
        mask = target != null_value

    mask = mask.float()
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
        absolute_error = np.abs(prediction - target)
        return float(np.mean(np.nan_to_num(absolute_error * _normalized_mask(target))))


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
