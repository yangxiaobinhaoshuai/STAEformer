from dataclasses import dataclass
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import torch
from pathlib import Path


@dataclass(frozen=True)
class StandardScalar:
    mean: float
    std: float

    def transform(self, values):
        return (values - self.mean) / self.std

    def inverse_transform(self, values):
        return values * self.std + self.mean


def make_ranges(starts: np.ndarray, stops: np.ndarray) -> np.ndarray:

    lengths = stops - starts
    if lengths.min() != lengths.max():
        raise ValueError("同一批窗口的长度必须相同")

    offset = np.arange(lengths[0])
    return starts[:, None] + offset[None, :]


def _build_split(
    data: np.ndarray,
    split_index: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:

    x_index = make_ranges(split_index[:, 0], split_index[:, 1])
    y_index = make_ranges(split_index[:, 1], split_index[:, 2])

    x = data[x_index]
    y = data[y_index][..., :1]
    return x, y


def _to_loader(
    x: np.ndarray,
    y: np.ndarray,
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    dataset = TensorDataset(torch.from_numpy(x), torch.from_numpy(y))
    return DataLoader(dataset, batch_size, shuffle=shuffle)

def create_dataloaders(
        data_dir: Path,
        batch_size: int,
        use_time_of_day: bool = True,
        use_day_of_week: bool = True,

) -> tuple[DataLoader, DataLoader, DataLoader, StandardScalar]

    raw_data = np.load(data_dir / "data.npz")["data"].astype(np.float32)

    selected_features = [0]
    if use_time_of_day:
        selected_features.append(1)
    if use_day_of_week:
        selected_features.append(2)
    data = raw_data[..., selected_features]

    indexes = np.load(data_dir / "index.npz")
    x_train, y_train = _build_split(data, indexes["train"])
    x_val, y_val = _build_split(data, indexes["val"])
    x_test, y_test = _build_split(data, indexes["test"])

    scaler = StandardScalar(
        mean=float(x_train[...,0].mean()),
        std =float(x_train[...,0].std()),
    )

    if scaler.std == 0:
        raise ValueError("训练集交通值的标准差为 0， 无法进行标准化")

    x_train[..., 0] = scaler.transform(x_train[..., 0])
    x_val[..., 0] = scaler.transform(x_val[..., 0])
    x_test[..., 0] = scaler.transform(x_test[..., 0])

    print(f"Train: x={x_train.shape}, y={y_train.shape}")
    print(f"Val:   x={x_val.shape}, y={y_val.shape}")
    print(f"Test:  x={x_test.shape}, y={y_test.shape}")
    print(f"Scaler: mean={scaler.mean:.4f}, std={scaler.std:.4f}")

    return (
        _to_loader(x_train, y_train, batch_size, shuffle=True),
        _to_loader(x_val, y_val, batch_size, shuffle=False),
        _to_loader(x_test, y_test, batch_size, shuffle=False),
        scaler,
    )