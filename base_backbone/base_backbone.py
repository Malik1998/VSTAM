import torch
import torch.nn as nn
from transformer.video_sparse_transformer_encoder import VideoSparseTransformerEncoder
from transformer.video_sparse_transformer_decoder import VideoSparseTransformerDecoder
from .models.resnet import Resnet50Custom

dict_of_iternal_models = {
    "VideoSparseTransformerEncoder": VideoSparseTransformerEncoder,
    "VideoSparseTransformerDecoder": VideoSparseTransformerDecoder,
    "Resnet50Custom": Resnet50Custom
}

def get_model_by_name(model_name,
                      **kwargs):
    if any([name in model_name for name in ["pytorch", "resnet"]]):
        model = torch.hub.load('pytorch/vision:v0.10.0', model_name, **kwargs)
    else:
        model = dict_of_iternal_models[model_name]
    return model

class BaseBackbone(nn.Module):
    def __init__(self,
                 feature_embedder_name="Resnet50Custom",
                 encoder_name="VideoSparseTransformerEncoder",
                 decoder_name="VideoSparseTransformerDecoder",
                 spatial_encoder_size=64,
                 feature_embedder_out=2048,
                 hidden_size=128,
                 ):
        super(BaseBackbone, self).__init__()
        self.feature_embedder = get_model_by_name(feature_embedder_name)(pretrained=True)
        self.feature_embedder_out = feature_embedder_out
        self.encoder = get_model_by_name(encoder_name)(in_spatial_length=spatial_encoder_size,
                                                                        in_channels=feature_embedder_out,
                                                                        hidden_size=hidden_size
                                                                        )
        self.decoder = get_model_by_name(decoder_name)(hidden_size=hidden_size,
                                                       decoder_out=feature_embedder_out)
        self.out_channels = feature_embedder_out

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

dict_of_iternal_models.update({"BaseBackbone": BaseBackbone})

