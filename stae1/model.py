import torch
from torch import nn, Tensor


class MultiHeadAttn(nn.Module):
    def __init__(self, d_model: int, n_heads: int, causal_mask: bool = False):
        super().__init__()

        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisialbe by n_heads")

        self.n_heads = n_heads
        self.d_model = d_model
        self.d_k = d_model // n_heads
        self.causal_mask = causal_mask

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.o_proj = nn.Linear(d_model, d_model)

    def _split_heads(self, x: Tensor) -> Tensor:
        return torch.cat(torch.split(x, self.d_k, dim=-1), dim=0)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> Tensor:
        batch_size = query.shape[0]
        target_length = query.shape[-2]
        source_length = key.shape[-2]

        query = self._split_heads(self.q_proj(query))
        key = self._split_heads(self.k_proj(key))
        value = self._split_heads(self.v_proj(value))

        attn_score = query @ key.transpose(-1, -2)
        attn_score = attn_score / (self.d_k**0.5)

        if self.causal_mask:
            mask = torch.ones(
                target_length,
                source_length,
                dtype=torch.bool,
                device=query.device,
            ).tril()
            attn_score.masked_fill_(~mask, -torch.inf)

        attn_w = torch.softmax(attn_score, dim=-1)
        output = attn_w @ value

        output = torch.cat(torch.split(output, batch_size, dim=0), dim=-1)

        return self.o_proj(output)


class Transformer(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()

        self.self_attn = MultiHeadAttn(d_model, n_heads, False)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.ReLU(inplace=True), nn.Linear(d_ff, d_model)
        )

        self.attn_norm = nn.LayerNorm(d_model)
        self.ffn_norm = nn.LayerNorm(d_model)
        self.attn_dropout = nn.Dropout(dropout)
        self.ffn_dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor, attn_dim: int) -> Tensor:

        x = x.transpose(attn_dim, -2)

        residual = x
        x = self.self_attn(x, x, x)
        x = self.attn_norm(residual + self.attn_dropout(x))

        residual = x
        x = self.ffn(x)
        x = self.ffn_norm(residual + self.ffn_norm(x))

        return x.transpose(attn_dim, -2)


class STAEFormer(nn.Module):
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
    ):
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

        if self.model_dim % num_heads == 1:
            raise ValueError("All dimens sum must be disvisable by n_heads")

        self.input_proj = nn.Linear(input_dim, input_embedding_dim)

        if dow_embedding_dim > 0:
            self.time_of_day_embedding = nn.Embedding(steps_per_day, tod_embedding_dim)
        if tod_embedding_dim > 0:
            self.day_of_week_embedding = nn.Embedding(7, dow_embedding_dim)

        if spatial_embedding_dim > 0:
            self.node_embedding = nn.Parameter(
                torch.empty(num_nodes, spatial_embedding_dim)
            )
            nn.init.xavier_uniform_(self.node_embedding)

        if self.adaptive_embedding_dim > 0:
            self.adaptive_embedding = nn.Parameter(
                torch.empty(in_steps, num_nodes, adaptive_embedding_dim)
            )
            nn.init.xavier_uniform_(self.adaptive_embedding)

        self.temporal_blocks = nn.ModuleList(
            Transformer(self.model_dim, feed_forward_dim, num_heads, dropout)
            for _ in range(num_layers)
        )

        self.spatial_blocks = nn.ModuleList(
            Transformer(self.model_dim, feed_forward_dim, num_heads, dropout)
            for _ in range(num_layers)
        )

        if use_mixed_proj:
            self.output_porj = nn.Linear(
                in_steps * self.model_dim,
                out_steps * output_dim,
            )
        else:
            self.temporal_proj = nn.Linear(in_steps, out_steps)
            self.output_proj = nn.Linear(self.model_dim, output_dim)

    def _make_embeddings(self, x: Tensor) -> Tensor:

        bs, in_steps, num_heads, input_features = x.shape
        expected_shape = (self.in_steps, self.num_nodes, self.input_dim)
        actual_shape = (in_steps, self.num_nodes, input_features)
        if actual_shape != expected_shape:
            raise ValueError

        time_of_day = x[..., 1] if self.tod_embedding_dim > 0 else None
        day_of_week = x[..., 2] if self.dow_embedding_dim > 0 else None

        features = [self.input_proj(x[..., : self.input_dim])]

        if time_of_day is not None:
            tod_index = (time_of_day * self.steps_per_day).long()
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
            features.append(node_embedding.expand(bs, in_steps, -1, -1))

        if self.adaptive_embedding > 0:
            features.append(self.adaptive_embedding.unsqueeze(0).expand(bs, -1, -1, -1))

        return torch.cat(features, dim=-1)

    def forward(self, x: Tensor) -> Tensor:
        bs = x.shape[0]
        x = self._make_embeddings(x)

        for block in self.temporal_blocks:
            x = block(x, attn_dim=1)

        if self.use_mixed_proj:
            x = x.transpose(1, 2)
            x = x.reshape(bs, self.num_nodes, self.in_steps * self.model_dim)
            x = self.output_porj(x)
            x = x.view(bs, self.num_nodes, self.out_steps, self.output_dim)

            return x.transpose(1, 2)

        x = x.transpose(1, 3)
        x = self.temporal_blocks(x)
        return self.output_proj(x.transpose(1, 3))
