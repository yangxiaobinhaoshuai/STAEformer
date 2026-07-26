import torch
from torch import nn, Tensor
from typing import Optional



def make_weight(input_size: int, hidden_size: int) -> nn.Parameter:
    weight = nn.Parameter(torch.empty(input_size, hidden_size))
    nn.init.normal_(weight)
    return weight



class GRU(nn.Module):
    def __init__(self,input_size: int, hidden_size: int):
        super().__init__()

        self.hidden_size = hidden_size

        combined_size = input_size + hidden_size

        self.w_r = make_weight(combined_size, hidden_size)
        self.w_z = make_weight(combined_size, hidden_size)
        self.w_n = make_weight(combined_size, hidden_size)

        self.b_r = nn.Parameter(torch.zeros(hidden_size))
        self.b_z = nn.Parameter(torch.zeros(hidden_size))
        self.b_n = nn.Parameter(torch.zeros(hidden_size))

    def forward(self, x: Tensor, h: Optional[Tensor] = None) -> tuple[Tensor, Tensor]: 

        bs, seq_len, _ = x.shape

        outputs = []

        if h is None:
            h = x.new_zeros(bs, self.hidden_size)

        for t in range(seq_len):
            x_t = x[:, t, :]

            combined = torch.cat([x_t, h], dim = -1)


            r = torch.sigmoid(combined @ self.w_r + self.b_r)
            z = torch.sigmoid(combined @ self.w_z + self.b_z)

            candinate_input = torch.cat([x_t, r * h], dim = -1)

            n = torch.tanh(combined @ self.w_n + self.b_n)

            h = z * h + (1 - z) * n

            outputs.append(h)


        output = torch.stack(outputs, dim = 1)

        return output, h