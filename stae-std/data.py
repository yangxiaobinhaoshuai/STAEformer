"""从连续交通序列构造 STAEformer 所需的滑动窗口样本。"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset


@dataclass(frozen=True)
class StandardScaler:
    """只保存训练集交通值的均值和标准差。"""

    mean: float
    std: float

    def transform(self, values):
        return (values - self.mean) / self.std

    def inverse_transform(self, values):
        # values 可以是 numpy.ndarray，也可以是 torch.Tensor；
        # 两者都支持这里的乘法和加法。
        return values * self.std + self.mean


def make_ranges(starts: np.ndarray, stops: np.ndarray) -> np.ndarray:
    """把多组 [start, stop) 边界展开成二维索引。

    例：
        starts = [1, 5], stops = [4, 8]
        结果为 [[1, 2, 3], [5, 6, 7]]

    STAEformer 的每个样本窗口长度必须相同。
    """
    lengths = stops - starts
    if lengths.min() != lengths.max():
        raise ValueError("同一批窗口的长度必须相同")

    offsets = np.arange(lengths[0])
    return starts[:, None] + offsets[None, :]


def _build_split(
    data: np.ndarray,
    split_index: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """根据 [x_start, x_end, y_end] 构造一个数据集分区。"""
    x_index = make_ranges(split_index[:, 0], split_index[:, 1])
    y_index = make_ranges(split_index[:, 1], split_index[:, 2])

    # 高级索引后：
    # x: (samples, in_steps, num_nodes, input_features)
    # y: (samples, out_steps, num_nodes, 1)
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
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def create_dataloaders(
    data_dir: Path,
    batch_size: int,
    use_time_of_day: bool = True,
    use_day_of_week: bool = True,
) -> tuple[DataLoader, DataLoader, DataLoader, StandardScaler]:
    """读取 npz，切分样本，标准化输入并返回三个 DataLoader。"""
    raw_data = np.load(data_dir / "data.npz")["data"].astype(np.float32)

    # 原数据的通道约定：
    # 0=交通值，1=一天中的归一化位置，2=星期几。
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

    # 关键点：统计量只能来自训练集，否则验证/测试信息会泄漏到训练过程。
    scaler = StandardScaler(
        mean=float(x_train[..., 0].mean()),
        std=float(x_train[..., 0].std()),
    )
    if scaler.std == 0:
        raise ValueError("训练集交通值的标准差为 0，无法进行标准化")

    # 只标准化交通值；tod 和 dow 是离散时间标记，后面会进入 Embedding。
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
