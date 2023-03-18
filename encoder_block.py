# copy pasted & edited
# from https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial6/Transformers_and_MHAttention.html

import torch
import torch.nn as nn
from torch.nn import MultiheadAttention

class EncoderBlock(nn.Module):

    def __init__(self, input_dim, num_heads, dim_feedforward, dropout=0.0):
        """
        Inputs:
            input_dim - Dimensionality of the input
            num_heads - Number of heads to use in the attention block
            dim_feedforward - Dimensionality of the hidden layer in the MLP
            dropout - Dropout probability to use in the dropout layers
        """
        super().__init__()
        self.input_dim = input_dim
        # Attention layer
        self.self_attn = MultiheadAttention(input_dim, num_heads, dropout=dropout, batch_first=True)

        # Two-layer MLP
        self.linear_net = nn.Sequential(
            nn.Linear(input_dim, dim_feedforward),
            nn.Dropout(dropout),
            nn.GELU(),
            nn.Linear(dim_feedforward, input_dim)
        )

        # Layers to apply in between the main layers
        self.norm1 = nn.LayerNorm(input_dim)
        self.norm2 = nn.LayerNorm(input_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, **kwargs):
        # Attention part
        x, mask = x
        q, k, v = x
        attn_out, _ = self.self_attn(q, k, v, attn_mask=mask, need_weights=False)
        q = q + self.dropout(attn_out)
        q = self.norm1(q)

        # MLP part
        linear_out = self.linear_net(q)
        q = q + self.dropout(linear_out)
        q = self.norm2(q)
        return (q, k, v),mask