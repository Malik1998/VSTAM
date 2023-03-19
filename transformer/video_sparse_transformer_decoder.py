import torch.nn as nn
from .modules.mha_enncoder import MHAEncoder
import math

class VideoSparseTransformerDecoder(nn.Module):
    def __init__(self,
                 hidden_size=128,
                 decoder_layers_count=5,
                 decoder_num_heads=8,
                 decoder_dropout=0.1,
                 decoder_out=128,
                 ):
        super(VideoSparseTransformerDecoder, self).__init__()
        self.decoder_mha = MHAEncoder(hidden_size=hidden_size, num_heads=decoder_num_heads,
                                  dropout=decoder_dropout, layers_count=decoder_layers_count)
        self.out = nn.Conv2d(hidden_size, decoder_out, kernel_size=1)

    def forward(self, context, query, **kwargs):
        # input - [B, L * HW, hidden_size]

        # don't send the mask
        decoder_mha, weights = self.decoder_mha(((query, context, context), None))
        # use somehow weights to memory saving (in the original paper we use weights
        # of decoder to choose best frames to save)


        # let's consider image to be square
        h_mult_w_shape = query.shape[1]
        W = int(math.sqrt(h_mult_w_shape))
        H = h_mult_w_shape // W
        decoder_mha = decoder_mha.view(query.shape[0], -1, W, H)
        out = self.out(decoder_mha)
        return out



