import torch.nn as nn
from .encoder_block import EncoderBlock

class MHAEncoder(nn.Module):
    def __init__(self, hidden_size=128, num_heads=8, dropout=0.1, layers_count=5, **kwargs):
        super(MHAEncoder, self).__init__()
        self.seq = nn.Sequential(
            *[
                EncoderBlock(input_dim=hidden_size, num_heads=num_heads,
                             dim_feedforward=hidden_size, dropout=dropout)
                for _ in range(layers_count)
            ]
        )

    def forward(self, input, **kwargs):
        # self.seq(input) -> ((q, k, v), mask, weights)
        return self.seq(input)[0][0], self.seq(input)[2][0]