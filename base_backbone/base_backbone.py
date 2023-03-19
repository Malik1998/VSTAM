import torch
import torch.nn as nn
from transformer.video_sparse_transformer_encoder import VideoSparseTransformerEncoder
from transformer.video_sparse_transformer_decoder import VideoSparseTransformerDecoder
from .models.resnet import Resnet50Custom

class BaseBackbone(nn.Module):
    def __init__(self,
                 hidden_size=128,
                 ):
        super(BaseBackbone, self).__init__()
        self.feature_embedder = Resnet50Custom(pretrained=True)
        self.feature_embedder_out = self.feature_embedder.feature_embedder_out
        self.encoder = VideoSparseTransformerEncoder(in_channels=self.feature_embedder_out,
                                                     hidden_size=hidden_size)
        self.decoder = VideoSparseTransformerDecoder(hidden_size=hidden_size,
                                                     decoder_out=self.feature_embedder_out)

    def forward(self, input, **kwargs):
        input_concated = input.view(input.size(0) * input.size(1), input.size(2), input.size(3), -1)
        feature_embeddings = self.feature_embedder(input_concated)
        w,h = feature_embeddings.size(2), feature_embeddings.size(3)
        feature_embeddings = feature_embeddings.view(input.size(0), input.size(1),
                                                     -1,
                                                     self.feature_embedder_out)
        encoder_embeddings, query = self.encoder(feature_embeddings)
        decoder_embeddings = self.decoder(encoder_embeddings, query)

        feature_embeddings = feature_embeddings.view(input.size(0), input.size(1),
                                                     self.feature_embedder_out,
                                                     w,
                                                     h
                                                     )

        summed_embedding = feature_embeddings[:, 0] + decoder_embeddings

        return summed_embedding

