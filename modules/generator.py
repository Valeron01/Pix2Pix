import typing
import torch
from torch import nn

from modules.blocks import ResidualBlock, Downsample, Upsample


class UNet(nn.Module):
    def __init__(
            self,
            in_channels: int = 3,
            out_channels: int = 3,
            n_features_list: typing.List or typing.Tuple = (64, 128, 256),

    ):
        super().__init__()

        self.n_features_list = n_features_list

        self.depth = len(n_features_list)

        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, n_features_list[0], 7, 1, 3, bias=False),
            nn.BatchNorm2d(n_features_list[0]),
            nn.LeakyReLU(inplace=True)
        )

        self.encoder = nn.ModuleList()
        for in_features, out_features in zip(
                n_features_list[:-1], n_features_list[1:]
        ):
            self.encoder.append(nn.ModuleList([
                ResidualBlock(in_features, out_features),
                ResidualBlock(out_features, out_features),
                Downsample(out_features, out_features)
            ]))

        self.middle_block1 = ResidualBlock(
            in_channels=n_features_list[-1], out_channels=n_features_list[-1],
        )
        self.middle_block2 = ResidualBlock(
            in_channels=n_features_list[-1], out_channels=n_features_list[-1],
        )

        self.decoder = nn.ModuleList()
        for in_features, out_features in zip(
                reversed(n_features_list[1:]), reversed(n_features_list[:-1])
        ):
            self.decoder.append(nn.ModuleList([
                Upsample(in_features, in_features),
                ResidualBlock(in_features * 2, out_features),
                ResidualBlock(out_features, out_features),
            ]))

        self.final_conv = nn.Conv2d(n_features_list[0] * 2, out_channels, 1)

    def forward(self, x):
        stem = self.stem(x)

        downsample_stage = stem
        downsample_stages = [stem]
        for block1, block2, downsample in self.encoder:
            downsample_stage = block1(downsample_stage)
            downsample_stage = block2(downsample_stage)
            downsample_stages.append(downsample_stage)
            downsample_stage = downsample(downsample_stage)

        downsample_stage = self.middle_block1(downsample_stage)
        downsample_stage = self.middle_block2(downsample_stage)

        upsample_stage = downsample_stage
        for previous_stage, (upsample, block1, block2) in zip(reversed(downsample_stages), self.decoder):
            upsample_stage = upsample(upsample_stage)
            upsample_stage = torch.cat([upsample_stage, previous_stage], dim=1)
            upsample_stage = block1(upsample_stage)
            upsample_stage = block2(upsample_stage)

        return self.final_conv(
            torch.cat([upsample_stage, stem], dim=1)
        )


if __name__ == '__main__':
    unet = UNet(
        n_features_list=(32, 64, 128, 256, 512, 512)
    ).cuda()

    total_params = 0
    for param in unet.parameters():
        total_params += param.numel()
    print(f"Total params is: {total_params / 1e6}")

    for i in range(100):
        noise = torch.randn(8, 3, 256, 256).cuda()
        res = unet(noise)
        print(res.mean())
        del res
