import torch
import torch.nn as nn
from mha_enncoder import MHAEncoder
from simple_patch_embedding import simple_cnn_patch_embedding

model_names = {
    "simple_cnn_patch_embedding": simple_cnn_patch_embedding

}

class spatialViT(nn.Module):
    def __init__(self, patch_size=8, img_size=(128, 128), in_channels=64,
                 hidden_size=128, sequence_length=5,
                 random_attention_probability=0.1,


                 # vit_layers_count=5,
                 # vit_num_heads=8,
                 # vit_dropout=0.1,


                 encoder_layers_count=5,
                 encoder_num_heads=8,
                 encoder_dropout=0.1,

                 patch_embedding_name="simple_cnn_patch_embedding"):
        super(spatialViT, self).__init__()
        self.sequence_length = sequence_length
        self.hidden_size = hidden_size
        self.random_attention_probability = random_attention_probability
        self.patch_embedder = model_names[patch_embedding_name](patch_size=patch_size, img_size=img_size,
                                                                in_channels=in_channels,
                                                                out_channels=hidden_size)

        # self.vit = MHAEncoder(hidden_size=hidden_size, num_heads=vit_num_heads,
        #                       dropout=vit_dropout, layers_count=vit_layers_count)
        self.encoder = MHAEncoder(hidden_size=hidden_size, num_heads=encoder_num_heads,
                                  dropout=encoder_dropout, layers_count=encoder_layers_count)

    @staticmethod
    def make_random_attention(patches_on_img, seq_length, random_attention_probability=0.1):
        return torch.where(
            torch.rand((patches_on_img * seq_length, patches_on_img * seq_length)) >= random_attention_probability,
            1, 0)

    @staticmethod
    def make_position_attention(patches_on_img, seq_length):
        one_frame = torch.eye(patches_on_img, patches_on_img)

        return one_frame.repeate(seq_length, seq_length)

    @staticmethod
    def make_frame_attention(patches_on_img, seq_length):
        init_attention = torch.zeros((patches_on_img * seq_length, patches_on_img * seq_length))
        for i in range(seq_length):
            init_attention[i * patches_on_img: (i + 1) * patches_on_img, i * patches_on_img: (i + 1) * patches_on_img] = 1

    def make_positional_attention(self, patches_on_img, seq_length,
                                  random_attention_probability):
        return self.make_random_attention(patches_on_img, seq_length, random_attention_probability) + \
               self.make_position_attention(patches_on_img, seq_length) + \
               self.make_frame_attention(patches_on_img, seq_length)


    def forward(self, input):
        # input - [B, L, H, W, C]
        input_patches = self.patch_embedder(
            input.view(
                input.size(0) * self.sequence_length,
                input.size(2),
                input.size(3),
                input.size(4)
            ) # input_view = [B * L, H, W, C]
        ) # input_patches - [B * L, H//p_size * W//p_size, hidden_size]
        # encoded_patches = self.vit(input_patches)
        # sequence_of_patches - [B, L, H//p_size * W//p_size, hidden_size]
        sequence_of_patches = input_patches.view(input.size(0), input.size(1), -1, self.hidden_size)

        attention = self.make_positional_attention(sequence_of_patches.size(-2),
                                                   sequence_of_patches.size(1),
                                                   self.random_attention_probability)
        # sequence_of_attentioned - [B, L * H//p_size * W//p_size, hidden_size]
        sequence_of_attentioned = sequence_of_patches.view(sequence_of_patches.size(0),
                                                           sequence_of_patches.size(1) * sequence_of_patches.size(2),
                                                           sequence_of_patches.size(3))

        return self.encoder(sequence_of_attentioned, mask=attention)



