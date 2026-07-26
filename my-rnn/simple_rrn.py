import torch
from torch import nn, Tensor
from typing import Optional


def make_weight(in_dim: int, out_dim: int) -> nn.Parameter:
    weight = nn.Parameter(torch.empty(in_dim, out_dim))
    nn.init.xavier_uniform_(weight)
    return weight


class SimpleRnn(nn.Module):
    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()

        self.hidden_size = hidden_size

        self.w_x = make_weight(input_size, hidden_size)
        self.w_h = make_weight(input_size, hidden_size)
        self.bias = nn.Parameter(torch.zeros(hidden_size))

    def forward(self, x: Tensor, h: Optional[Tensor] = None) -> tuple[Tensor, Tensor]:

        bs, seq_len, _ = x.shape

        if h is None:
            h = x.new_zeros(bs, self.hidden_size)

        outputs = []

        for t in range(seq_len):
            x_t = x[:, t, :]

            h = torch.tanh(x_t @ self.w_x + h @ self.w_h + self.bias)

            outputs.append(h)

        output = torch.stack(outputs, dim=-1)
        return output, h


def make_weight1(in_dim: int, out_dim: int) -> nn.Parameter:
    weight = nn.Parameter(torch.empty(in_dim, out_dim))
    nn.init.xavier_uniform_(weight)
    return weight


class SimpleRNN1(nn.Module):

    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()

        self.hidden_size = hidden_size

        self.w_x = make_weight(input_size, hidden_size)
        self.w_h = make_weight(input_size, hidden_size)

        self.bias = nn.Parameter(torch.zeros(hidden_size))


    def forward(self, x: Tensor, h: Optional[Tensor] = None) -> tuple[Tensor, Tensor]:

        bs, seq_len, _ = x.shape

        outputs = []

        if h is None:
            h = x.new_zeros(bs, self.hidden_size)

        for t in range(seq_len):

            x_t = x[:, t, :]

            h = torch.tanh(
                x_t @ self.w_x
                + h @ self.w_h
                + self.bias
            )

            outputs.append(h)

        output = torch.stack(outputs, dim = 1)
        return output, h
