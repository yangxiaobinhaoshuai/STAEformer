"""STAEformer 教学版模型。

全文件只围绕一个张量契约：
    x: (batch, time, node, feature)
"""

import torch
from torch import nn


class MultiHeadAttention(nn.Module):
    """在倒数第二维 length 上执行多头注意力。"""

    def __init__(
        self,
        model_dim: int,
        num_heads: int,
        causal_mask: bool = False,
    ) -> None:
        super().__init__()
        if model_dim % num_heads != 0:
            raise ValueError("model_dim 必须能被 num_heads 整除")

        self.num_heads = num_heads
        self.head_dim = model_dim // num_heads
        self.causal_mask = causal_mask

        self.query_projection = nn.Linear(model_dim, model_dim)
        self.key_projection = nn.Linear(model_dim, model_dim)
        self.value_projection = nn.Linear(model_dim, model_dim)
        self.output_projection = nn.Linear(model_dim, model_dim)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        """把特征维拆成多个头，并把 head 合并到 batch 维。

        (B, ..., L, D) -> (B * H, ..., L, D/H)

        这种写法与原项目一致，优点是后面的矩阵乘法不必单独处理 head 维。
        """
        return torch.cat(torch.split(x, self.head_dim, dim=-1), dim=0)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> torch.Tensor:
        batch_size = query.shape[0]
        target_length = query.shape[-2]
        source_length = key.shape[-2]

        query = self._split_heads(self.query_projection(query))
        key = self._split_heads(self.key_projection(key))
        value = self._split_heads(self.value_projection(value))

        # QK^T 得到“每个目标位置应该关注每个来源位置多少”的分数。
        attention_score = query @ key.transpose(-1, -2)
        attention_score = attention_score / (self.head_dim**0.5)

        if self.causal_mask:
            # 下三角 mask 禁止当前位置看到未来位置。
            # STAEformer 默认关闭：输入的 12 步全是已知历史，不包含预测标签。
            mask = torch.ones(
                target_length,
                source_length,
                dtype=torch.bool,
                device=query.device,
            ).tril()
            attention_score.masked_fill_(~mask, -torch.inf)

        attention_weight = torch.softmax(attention_score, dim=-1)
        output = attention_weight @ value

        # 把合并进 batch 的多个 head 重新拼回特征维：
        # (B * H, ..., L, D/H) -> (B, ..., L, D)
        output = torch.cat(torch.split(output, batch_size, dim=0), dim=-1)
        return self.output_projection(output)


class TransformerEncoderBlock(nn.Module):
    """自注意力 + 前馈网络，并在两处使用残差连接和 LayerNorm。"""

    def __init__(
        self,
        model_dim: int,
        feed_forward_dim: int,
        num_heads: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.attention = MultiHeadAttention(model_dim, num_heads)
        self.feed_forward = nn.Sequential(
            nn.Linear(model_dim, feed_forward_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feed_forward_dim, model_dim),
        )
        self.attention_norm = nn.LayerNorm(model_dim)
        self.feed_forward_norm = nn.LayerNorm(model_dim)
        self.attention_dropout = nn.Dropout(dropout)
        self.feed_forward_dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, attention_dim: int) -> torch.Tensor:
        # MultiHeadAttention 永远沿倒数第二维计算。
        # 把 time 或 node 换到倒数第二维，就能复用同一套注意力代码。
        x = x.transpose(attention_dim, -2)

        residual = x
        x = self.attention(x, x, x)
        x = self.attention_norm(residual + self.attention_dropout(x))

        residual = x
        x = self.feed_forward(x)
        x = self.feed_forward_norm(residual + self.feed_forward_dropout(x))

        return x.transpose(attention_dim, -2)


class STAEformer(nn.Module):
    def __init__(
        self,
        num_nodes: int,
        in_steps: int = 12,
        out_steps: int = 12,
        steps_per_day: int = 288,
        input_dim: int = 3,
        output_dim: int = 1,
        input_embedding_dim: int = 24,
        tod_embedding_dim: int = 24,
        dow_embedding_dim: int = 24,
        spatial_embedding_dim: int = 0,
        adaptive_embedding_dim: int = 80,
        feed_forward_dim: int = 256,
        num_heads: int = 4,
        num_layers: int = 3,
        dropout: float = 0.1,
        use_mixed_proj: bool = True,
    ) -> None:
        super().__init__()
        self.num_nodes = num_nodes
        self.in_steps = in_steps
        self.out_steps = out_steps
        self.steps_per_day = steps_per_day
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.tod_embedding_dim = tod_embedding_dim
        self.dow_embedding_dim = dow_embedding_dim
        self.spatial_embedding_dim = spatial_embedding_dim
        self.adaptive_embedding_dim = adaptive_embedding_dim
        self.use_mixed_proj = use_mixed_proj

        self.model_dim = (
            input_embedding_dim
            + tod_embedding_dim
            + dow_embedding_dim
            + spatial_embedding_dim
            + adaptive_embedding_dim
        )
        if self.model_dim % num_heads != 0:
            raise ValueError("所有 embedding 维度之和必须能被 num_heads 整除")

        # 当前配置 input_dim=3，因此原始交通值、tod、dow 都会经过这个线性投影。
        self.input_projection = nn.Linear(input_dim, input_embedding_dim)

        if tod_embedding_dim > 0:
            self.time_of_day_embedding = nn.Embedding(
                steps_per_day,
                tod_embedding_dim,
            )
        if dow_embedding_dim > 0:
            self.day_of_week_embedding = nn.Embedding(7, dow_embedding_dim)

        if spatial_embedding_dim > 0:
            # 静态节点 embedding：同一个节点在所有时间位置共享。
            self.node_embedding = nn.Parameter(
                torch.empty(num_nodes, spatial_embedding_dim)
            )
            nn.init.xavier_uniform_(self.node_embedding)

        if adaptive_embedding_dim > 0:
            # 核心：每个“历史相对时间位置 × 节点”都有一个可学习向量。
            # 它不依赖输入值，训练时从数据中自动学出潜在时空模式。
            self.adaptive_embedding = nn.Parameter(
                torch.empty(in_steps, num_nodes, adaptive_embedding_dim)
            )
            nn.init.xavier_uniform_(self.adaptive_embedding)

        self.temporal_blocks = nn.ModuleList(
            TransformerEncoderBlock(
                self.model_dim,
                feed_forward_dim,
                num_heads,
                dropout,
            )
            for _ in range(num_layers)
        )
        self.spatial_blocks = nn.ModuleList(
            TransformerEncoderBlock(
                self.model_dim,
                feed_forward_dim,
                num_heads,
                dropout,
            )
            for _ in range(num_layers)
        )

        if use_mixed_proj:
            # 同时混合历史时间信息和特征信息，一次性输出未来所有步。
            self.output_projection = nn.Linear(
                in_steps * self.model_dim,
                out_steps * output_dim,
            )
        else:
            # 原项目保留的另一种输出头：先映射时间，再映射特征。
            self.temporal_projection = nn.Linear(in_steps, out_steps)
            self.output_projection = nn.Linear(self.model_dim, output_dim)

    def _make_embeddings(self, x: torch.Tensor) -> torch.Tensor:
        """把输入投影和各种 embedding 拼成最终 model_dim。"""
        batch_size, in_steps, num_nodes, input_features = x.shape
        expected_shape = (self.in_steps, self.num_nodes, self.input_dim)
        actual_shape = (in_steps, num_nodes, input_features)
        if actual_shape != expected_shape:
            raise ValueError(
                f"输入后三维应为 {expected_shape}，实际为 {actual_shape}"
            )

        # 必须在覆盖 x 之前保存时间标记。
        time_of_day = x[..., 1] if self.tod_embedding_dim > 0 else None
        day_of_week = x[..., 2] if self.dow_embedding_dim > 0 else None

        features = [self.input_projection(x[..., : self.input_dim])]

        if time_of_day is not None:
            # data 中 tod 约在 [0, 1)，乘 288 后变成当天第几个 5 分钟槽。
            tod_index = (time_of_day * self.steps_per_day).long()
            # 浮点舍入可能产生边界值，clamp 能给出更明确、稳定的行为。
            tod_index = tod_index.clamp(0, self.steps_per_day - 1)
            features.append(self.time_of_day_embedding(tod_index))

        if day_of_week is not None:
            dow_index = day_of_week.long().clamp(0, 6)
            features.append(self.day_of_week_embedding(dow_index))

        if self.spatial_embedding_dim > 0:
            node_embedding = self.node_embedding.view(
                1,
                1,
                self.num_nodes,
                self.spatial_embedding_dim,
            )
            features.append(
                node_embedding.expand(batch_size, self.in_steps, -1, -1)
            )

        if self.adaptive_embedding_dim > 0:
            features.append(
                self.adaptive_embedding.unsqueeze(0).expand(batch_size, -1, -1, -1)
            )

        # (B, Tin, N, 24+24+24+80) = (B, Tin, N, 152)
        return torch.cat(features, dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        x = self._make_embeddings(x)

        # 固定节点，沿 time=1 维学习同一节点的历史依赖。
        for block in self.temporal_blocks:
            x = block(x, attention_dim=1)

        # 固定时间，沿 node=2 维学习不同传感器之间的关系。
        for block in self.spatial_blocks:
            x = block(x, attention_dim=2)

        if self.use_mixed_proj:
            # 每个节点单独把 Tin × model_dim 展平并预测 Tout。
            x = x.transpose(1, 2)
            x = x.reshape(batch_size, self.num_nodes, self.in_steps * self.model_dim)
            x = self.output_projection(x)
            x = x.view(
                batch_size,
                self.num_nodes,
                self.out_steps,
                self.output_dim,
            )
            return x.transpose(1, 2)

        x = x.transpose(1, 3)
        x = self.temporal_projection(x)
        return self.output_projection(x.transpose(1, 3))
