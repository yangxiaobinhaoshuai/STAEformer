import torch
from torch import nn, Tensor
from typing import Optional


def make_weight(input_size: int, hidden_size: int) -> nn.Parameter:
    weight = nn.Parameter(torch.empty(input_size, hidden_size))
    nn.init.uniform_(weight)
    return weight


class SimpleLSTM(nn.Module):
    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()

        self.hiden_size = hidden_size

        # i f o g
        self.weight = make_weight(input_size + hidden_size, hidden_size * 4)

        self.bias = nn.Parameter(torch.zeros(hidden_size * 4))

    def forward(
        self, x: Tensor, state: tuple[Tensor, Tensor]
    ) -> tuple[Tensor, tuple[Tensor, Tensor]]:

        bs, seq_len, _ = x.shape

        if state is None:
            h = x.new_zeros(bs, self.hiden_size)
            c = x.new_zeors(bs, self.hiden_size)
        else:
            h, c = state

        outputs = []

        for t in range(seq_len):
            x_t = x[:, t, :]

            combined = torch.cat([x_t, h], dim=-1)

            gates = combined @ self.weight + self.bias

            i, f, o, g = gates.split(self.hiden_size, dim=-1)

            i = torch.sigmoid(i)
            f = torch.sigmoid(f)
            o = torch.sigmoid(o)
            g = torch.tanh(g)

            c = f * c + i * g

            h = o * torch.tanh(c)

            outputs.append(h)

        output = torch.stack(outputs, dim=1)

        return output, (h, c)


class SimpleLSTM1(nn.Module):
    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()

        self.hidden_size = hidden_size

        self.weight = make_weight(input_size + hidden_size, hidden_size * 4)

        self.bias = nn.Parameter(torch.zeros(4 * hidden_size))

    def forward(
        self, x: Tensor, state: Optional[tuple[Tensor, Tensor]] = None
    ) -> tuple[Tensor, tuple[Tensor, Tensor]]:

        bs, seq_len, _ = x.shape

        if state is None:
            h = x.new_zeros(bs, self.hidden_size)
            c = x.new_zeros(bs, self.hidden_size)
        else:
            h, c = state

        outputs = []

        for t in range(seq_len):
            x_t = x[:, t, :]

            combined = torch.cat([x_t, h], dim=-1)

            gates = combined @ self.weight + self.bias

            i, f, o, g = gates.split(self.hidden_size, dim=-1)

            i = torch.sigmoid(i)
            f = torch.sigmoid(f)
            o = torch.sigmoid(o)
            g = torch.tanh(g)

            c = f * c + i * g

            h = o * torch.tanh(c)

            outputs.append(h)

        output = torch.stack(outputs, dim=1)

        return output, (h, c)
