import typing

import torch
from torch import nn

from modules.blocks import ResidualBlock, Downsample


class Discriminator(nn.Module):
    def __init__(self, in_channels: int = 4, n_features_list: typing.List or typing.Tuple = (32, 64, 128, 256)):
        super().__init__()

        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, n_features_list[0], 7, 1, 3),
            nn.BatchNorm2d(n_features_list[0]),
            nn.LeakyReLU(0.1, inplace=True)
        )
        self.model = nn.Sequential(*[
            nn.Sequential(
                ResidualBlock(in_features, out_features),
                Downsample(out_features, out_features),
                ResidualBlock(out_features, out_features)
            )
            for in_features, out_features in zip(n_features_list[:-1], n_features_list[1:])
        ])

        self.to_logits = nn.Conv2d(n_features_list[-1], 1, 1)

    def forward(self, x):
        stem = self.stem(x)
        features = self.model(stem)
        logits = self.to_logits(features)
        return logits


if __name__ == '__main__':
    model = Discriminator(3)
    noise = torch.randn(1, 3, 256, 256)
    print(model(noise).shape)