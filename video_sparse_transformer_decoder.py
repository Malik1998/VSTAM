import torch
import torch.nn as nn
from mha_enncoder import MHAEncoder

class VideoSparseTransformerDecoder(nn.Module):
    def __init__(self,
                 width=8,
                 height=8,
                 hidden_size=128,
                 decoder_layers_count=5,
                 decoder_num_heads=8,
                 decoder_dropout=0.1,
                 decoder_out=128,
                 ):
        super(VideoSparseTransformerDecoder, self).__init__()
        self.width = width
        self.height = height
        self.decoder_mha = MHAEncoder(hidden_size=hidden_size, num_heads=decoder_num_heads,
                                  dropout=decoder_dropout, layers_count=decoder_layers_count)
        self.out = nn.Conv2d(hidden_size, decoder_out, kernel_size=1)

    def forward(self, input, query, **kwargs):
        # input - [B, L * HW, hidden_size]
        decoder_mha = self.decoder_mha(((query, input, input), None))
        # use_somehow to memory saving
        decoder_mha = decoder_mha.view(query.shape[0], -1, self.width, self.height)
        out = self.out(decoder_mha)
        return out



