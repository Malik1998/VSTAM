import torch
import torch.nn as nn
from .modules.mha_enncoder import MHAEncoder

class VideoSparseTransformerEncoder(nn.Module):
    def __init__(self, in_channels=64,
                 hidden_size=128,
                 random_attention_probability=0.1,

                 encoder_layers_count=5,
                 encoder_num_heads=8,
                 encoder_dropout=0.1
                 ):
        super(VideoSparseTransformerEncoder, self).__init__()
        self.in_channels = in_channels
        self.hidden_size = hidden_size
        self.random_attention_probability = random_attention_probability

        self.to_hidden_dim = nn.Linear(in_channels, hidden_size)

        self.encoder = MHAEncoder(hidden_size=hidden_size, num_heads=encoder_num_heads,
                                  dropout=encoder_dropout, layers_count=encoder_layers_count)

    @staticmethod
    def make_random_attention(in_spatial_length, seq_length, random_attention_probability=0.1):
        return torch.where(
            torch.rand((in_spatial_length * seq_length, in_spatial_length * seq_length)) >= random_attention_probability,
            1, 0)

    @staticmethod
    def make_position_attention(in_spatial_length, seq_length):
        one_frame = torch.eye(in_spatial_length, in_spatial_length)

        return one_frame.repeat(seq_length, seq_length)

    @staticmethod
    def make_frame_attention(in_spatial_length, seq_length):
        init_attention = torch.zeros((in_spatial_length * seq_length, in_spatial_length * seq_length))
        for i in range(seq_length):
            init_attention[i * in_spatial_length: (i + 1) * in_spatial_length, i * in_spatial_length: (i + 1) * in_spatial_length] = 1
        return init_attention

    def make_positional_attention(self, in_spatial_length, seq_length,
                                  random_attention_probability):
        return self.make_random_attention(in_spatial_length, seq_length, random_attention_probability) + \
               self.make_position_attention(in_spatial_length, seq_length) + \
               self.make_frame_attention(in_spatial_length, seq_length)


    def forward(self, input, **kwargs):
        # input - [B, L, H * W, C]
        in_spatial_length = input.size(2)
        sequence_of_img_embs = input.view(input.size(0) * input.size(1), -1, self.in_channels)
        sequence_of_img_embs = self.to_hidden_dim(sequence_of_img_embs)
        # input - [B, L * H * W, HiddenSize]
        sequence_of_img_embs = sequence_of_img_embs.view(input.size(0), -1, self.hidden_size)
        attention = self.make_positional_attention(in_spatial_length=in_spatial_length,
                                                   seq_length=input.size(1),
                                                   random_attention_probability=self.random_attention_probability)
        attention = attention.to(input.get_device())
        # returns query and weights
        contexts, _ = self.encoder(((sequence_of_img_embs, sequence_of_img_embs, sequence_of_img_embs), attention))
        return contexts, \
               sequence_of_img_embs[:, :in_spatial_length] # first frame used as query to decoder



