"""STAEformer 教学版训练入口。

从仓库根目录运行：
    uv run python stae-std/train.py -d PEMS08 -g 0
"""

import argparse
import copy
import datetime
import os
import time
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from config import ExperimentConfig, get_config
from data import StandardScaler, create_dataloaders
from metrics import all_metrics, masked_mae_loss
from model import STAEformer
from utils import Logger, seed_everything


LossFunction = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


def build_loss(config: ExperimentConfig) -> LossFunction:
    if config.loss_name == "masked_mae":
        return masked_mae_loss
    if config.loss_name == "huber":
        return nn.HuberLoss()
    raise ValueError(f"未知 loss：{config.loss_name}")


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    scaler: StandardScaler,
    optimizer: torch.optim.Optimizer,
    criterion: LossFunction,
    device: torch.device,
    clip_grad: float | None,
) -> float:
    model.train()
    batch_losses = []

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        # 模型学习的是标准化空间中的输出。
        normalized_prediction = model(x)
        # 标签没有被标准化，所以计算 loss 前必须恢复原始交通值尺度。
        prediction = scaler.inverse_transform(normalized_prediction)
        loss = criterion(prediction, y)

        optimizer.zero_grad()
        loss.backward()
        if clip_grad is not None:
            nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
        optimizer.step()

        batch_losses.append(loss.item())

    return float(np.mean(batch_losses))


@torch.no_grad()
def evaluate_loss(
    model: nn.Module,
    loader: DataLoader,
    scaler: StandardScaler,
    criterion: LossFunction,
    device: torch.device,
) -> float:
    model.eval()
    batch_losses = []

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        prediction = scaler.inverse_transform(model(x))
        batch_losses.append(criterion(prediction, y).item())

    return float(np.mean(batch_losses))


@torch.no_grad()
def predict(
    model: nn.Module,
    loader: DataLoader,
    scaler: StandardScaler,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    targets = []
    predictions = []

    for x, y in loader:
        x = x.to(device)
        prediction = scaler.inverse_transform(model(x))

        targets.append(y.numpy())
        predictions.append(prediction.cpu().numpy())

    # 只移除 output_dim=1，不能用无参数 squeeze()，
    # 否则最后一个 batch 恰好为 1 时会误删 batch 维。
    target = np.concatenate(targets, axis=0).squeeze(-1)
    prediction = np.concatenate(predictions, axis=0).squeeze(-1)
    return target, prediction


def format_metrics(
    target: np.ndarray,
    prediction: np.ndarray,
    include_each_step: bool,
) -> str:
    rmse, mae, mape = all_metrics(target, prediction)
    lines = [f"All steps: RMSE={rmse:.5f}, MAE={mae:.5f}, MAPE={mape:.5f}"]

    if include_each_step:
        for step in range(prediction.shape[1]):
            rmse, mae, mape = all_metrics(
                target[:, step, :],
                prediction[:, step, :],
            )
            lines.append(
                f"Step {step + 1:02d}: RMSE={rmse:.5f}, "
                f"MAE={mae:.5f}, MAPE={mape:.5f}"
            )
    return "\n".join(lines)


def fit(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    scaler: StandardScaler,
    config: ExperimentConfig,
    device: torch.device,
    logger: Logger,
    max_epochs: int,
) -> tuple[nn.Module, int]:
    criterion = build_loss(config)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=list(config.milestones),
        gamma=config.lr_decay_rate,
    )

    model.to(device)
    best_state = None
    best_epoch = -1
    best_val_loss = float("inf")
    wait = 0

    for epoch in range(max_epochs):
        train_loss = train_one_epoch(
            model,
            train_loader,
            scaler,
            optimizer,
            criterion,
            device,
            config.clip_grad,
        )
        val_loss = evaluate_loss(
            model,
            val_loader,
            scaler,
            criterion,
            device,
        )
        scheduler.step()

        current_lr = optimizer.param_groups[0]["lr"]
        logger.write(
            f"Epoch {epoch + 1:03d} | train={train_loss:.5f} | "
            f"val={val_loss:.5f} | lr={current_lr:.6g}"
        )

        # 验证集只用于选择模型，不参与梯度更新。
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            wait = 0
        else:
            wait += 1
            if wait >= config.early_stop:
                logger.write(f"Early stop at epoch {epoch + 1}")
                break

    if best_state is None:
        raise RuntimeError("训练未产生可用的模型参数")

    # 训练结束时的模型不一定最好，必须恢复验证 loss 最小的那一轮。
    model.load_state_dict(best_state)
    logger.write(
        f"Best epoch: {best_epoch + 1}, best val loss: {best_val_loss:.5f}"
    )
    return model, best_epoch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="教学版 STAEformer")
    parser.add_argument("-d", "--dataset", default="PEMS08")
    parser.add_argument("-g", "--gpu", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--max-epochs",
        type=int,
        default=None,
        help="覆盖配置中的 epoch，做冒烟测试时可传 1",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset = args.dataset.upper()
    config = get_config(dataset)
    max_epochs = args.max_epochs or config.max_epochs

    seed_everything(args.seed)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # stae-std 位于仓库根目录下，因此 parents[1] 就是仓库根目录。
    project_root = Path(__file__).resolve().parents[1]
    data_dir = project_root / "data" / dataset
    if not data_dir.exists():
        raise FileNotFoundError(f"找不到数据目录：{data_dir}")

    train_loader, val_loader, test_loader, scaler = create_dataloaders(
        data_dir,
        batch_size=config.batch_size,
    )
    model = STAEformer(**config.model_kwargs)

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    log_path = project_root / "logs" / f"STAEformer-STD-{dataset}-{timestamp}.log"
    checkpoint_path = (
        project_root
        / "saved_models"
        / f"STAEformer-STD-{dataset}-{timestamp}.pt"
    )
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    with Logger(log_path) as logger:
        logger.write(f"Dataset: {dataset}")
        logger.write(f"Device: {device}")
        logger.write(f"Seed: {args.seed}")
        logger.write(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

        model, _ = fit(
            model,
            train_loader,
            val_loader,
            scaler,
            config,
            device,
            logger,
            max_epochs,
        )
        torch.save(model.state_dict(), checkpoint_path)
        logger.write(f"Checkpoint: {checkpoint_path}")

        # 恢复最佳模型后，再统一报告 train/val/test 指标。
        train_target, train_prediction = predict(
            model, train_loader, scaler, device
        )
        val_target, val_prediction = predict(model, val_loader, scaler, device)
        logger.write("\nTrain\n" + format_metrics(
            train_target, train_prediction, include_each_step=False
        ))
        logger.write("\nValidation\n" + format_metrics(
            val_target, val_prediction, include_each_step=False
        ))

        start = time.perf_counter()
        test_target, test_prediction = predict(
            model, test_loader, scaler, device
        )
        inference_seconds = time.perf_counter() - start
        logger.write("\nTest\n" + format_metrics(
            test_target, test_prediction, include_each_step=True
        ))
        logger.write(f"Inference time: {inference_seconds:.2f}s")


if __name__ == "__main__":
    main()
