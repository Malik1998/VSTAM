import torch
import torch.nn as nn

class simple_cnn_patch_embedding(nn.Module):
    def __init__(self, patch_size=8, in_channels=64, out_channels=128, image_size=(128, 128), **kwargs):
        super(simple_cnn_patch_embedding, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels=in_channels,
                      out_channels=out_channels,
                      kernel_size=patch_size,
                      stride=patch_size,
                      bias=True)
        )
        self.out_channels = out_channels
        self.positional_size = (image_size[0] // patch_size) * (image_size[1] // patch_size)
        self.positional_embedding = torch.nn.Parameter(torch.rand(self.positional_size, self.out_channels))


    def forward(self, input):
        patches = self.cnn(input)
        patches = patches.view(-1, self.positional_size, self.out_channels)
        patches_plus_positions = patches + self.positional_embedding
        return patches_plus_positions