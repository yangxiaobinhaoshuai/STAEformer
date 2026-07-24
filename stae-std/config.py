"""教学版 STAEformer 的配置。

原项目把配置写在 YAML 中。这里改成 dataclass，是为了让你在抄写时能同时看清：
1. 哪些参数属于数据集；
2. 哪些参数会真正传给模型；
3. 每个字段的类型和默认值。
"""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class ExperimentConfig:
    # ------------------------------- 数据形状 ------------------------------- #
    num_nodes: int
    in_steps: int = 12
    out_steps: int = 12
    steps_per_day: int = 288
    input_dim: int = 3
    output_dim: int = 1

    # ------------------------------- 模型维度 ------------------------------- #
    input_embedding_dim: int = 24
    tod_embedding_dim: int = 24
    dow_embedding_dim: int = 24
    spatial_embedding_dim: int = 0
    adaptive_embedding_dim: int = 80
    feed_forward_dim: int = 256
    num_heads: int = 4
    num_layers: int = 3
    dropout: float = 0.1
    use_mixed_proj: bool = True

    # ------------------------------- 训练参数 ------------------------------- #
    batch_size: int = 16
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    milestones: tuple[int, ...] = field(default_factory=tuple)
    lr_decay_rate: float = 0.1
    max_epochs: int = 200
    early_stop: int = 20
    clip_grad: float | None = None
    loss_name: str = "huber"

    @property
    def model_kwargs(self) -> dict:
        """只返回 STAEformer.__init__ 真正需要的参数。"""
        return {
            "num_nodes": self.num_nodes,
            "in_steps": self.in_steps,
            "out_steps": self.out_steps,
            "steps_per_day": self.steps_per_day,
            "input_dim": self.input_dim,
            "output_dim": self.output_dim,
            "input_embedding_dim": self.input_embedding_dim,
            "tod_embedding_dim": self.tod_embedding_dim,
            "dow_embedding_dim": self.dow_embedding_dim,
            "spatial_embedding_dim": self.spatial_embedding_dim,
            "adaptive_embedding_dim": self.adaptive_embedding_dim,
            "feed_forward_dim": self.feed_forward_dim,
            "num_heads": self.num_heads,
            "num_layers": self.num_layers,
            "dropout": self.dropout,
            "use_mixed_proj": self.use_mixed_proj,
        }


# 参数与 model/STAEformer.yaml 保持一致。
DATASET_CONFIGS: dict[str, ExperimentConfig] = {
    "METRLA": ExperimentConfig(
        num_nodes=207,
        weight_decay=3e-4,
        milestones=(20, 30),
        max_epochs=200,
        early_stop=30,
        loss_name="masked_mae",
    ),
    "PEMSBAY": ExperimentConfig(
        num_nodes=325,
        weight_decay=1e-4,
        milestones=(10, 30),
        max_epochs=300,
        early_stop=20,
        loss_name="masked_mae",
    ),
    "PEMS03": ExperimentConfig(
        num_nodes=358,
        weight_decay=5e-4,
        milestones=(15, 30, 40),
        max_epochs=300,
        early_stop=20,
    ),
    "PEMS04": ExperimentConfig(
        num_nodes=307,
        weight_decay=5e-4,
        milestones=(15, 30, 50),
        max_epochs=300,
        early_stop=20,
    ),
    "PEMS07": ExperimentConfig(
        num_nodes=883,
        weight_decay=1e-3,
        milestones=(15, 35, 50),
        max_epochs=300,
        early_stop=20,
    ),
    "PEMS08": ExperimentConfig(
        num_nodes=170,
        weight_decay=1.5e-3,
        milestones=(25, 45, 65),
        max_epochs=300,
        early_stop=30,
    ),
}


def get_config(dataset: str) -> ExperimentConfig:
    """取得数据集配置，同时把输入统一转成大写。"""
    dataset = dataset.upper()
    if dataset not in DATASET_CONFIGS:
        choices = ", ".join(DATASET_CONFIGS)
        raise ValueError(f"不支持数据集 {dataset!r}，可选值：{choices}")
    return DATASET_CONFIGS[dataset]
