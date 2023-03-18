import torch
import torch.nn as nn
from mha_enncoder import MHAEncoder

class spatialViTDecoder(nn.Module):
    def __init__(self, hidden_size=128,
                 decoder_layers_count=5,
                 decoder_num_heads=8,
                 decoder_dropout=0.1,
                 out_size=128):
        super(spatialViTDecoder, self).__init__()
        self.decoder_mha = MHAEncoder(hidden_size=hidden_size, num_heads=decoder_num_heads,
                                  dropout=decoder_dropout, layers_count=decoder_layers_count)

        # self.lin =

    def forward(self, input, query):
        # input - [B, L * patch_size * patch_size, hidden_size]
        decoder_mha = self.decoder_mha(query, input, input)

        return self.encoder(sequence_of_attentioned, mask=attention)



