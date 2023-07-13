import torch

import modules.generator

gen = modules.generator.UNet(n_features_list=(32, 64, 128, 256, 512, 512)).cuda()
disc = modules.discriminator.Discriminator(3).cuda()

noise = torch.randn(8, 3, 256, 256).cuda()
predicted = gen(noise)
logits = disc(predicted)

logits.sum().backward()
