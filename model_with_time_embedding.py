import math
from inspect import isfunction
from functools import partial

import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from einops import rearrange, reduce
from einops.layers.torch import Rearrange

import torch
from torch import nn, einsum
import torch.nn.functional as F



from utils import Residual, PreNorm, Upsample, Downsample, default, exists
from resnet_block import ResnetBlock
from embeding import SinusoidalPositionEmbeddings
from attention import Attention, LinearAttention



class DenseBlock(nn.Module):
    def __init__(self, num_layers, num_features):
        super(DenseBlock, self).__init__()
        layers = []
        for _ in range(num_layers):
            layers.append(nn.Conv2d(num_features, num_features, 3, 1, 1))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return x + self.block(x)

class RRDB(nn.Module):
    def __init__(self, num_features, num_layers=3, scaling_factor=0.2):
        super(RRDB, self).__init__()
        self.dense_blocks = nn.Sequential(
            DenseBlock(num_layers, num_features),
            DenseBlock(num_layers, num_features),
            DenseBlock(num_layers, num_features),
        )
        self.scaling_factor = scaling_factor

    def forward(self, x):
        out = x
        out = out + self.scaling_factor * self.dense_blocks(out)
        return out



class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, embedding_dim):
        super(SinusoidalPositionEmbeddings, self).__init__()
        self.embedding_dim = embedding_dim

    def forward(self, timesteps):
        half_dim = self.embedding_dim // 2
        emb = torch.exp(-torch.arange(half_dim, device=timesteps.device) * math.log(10000) / (half_dim - 1))
        emb = timesteps[:, None] * emb[None, :]
        return torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)


class Unet(nn.Module):
    """
    Modified UNet with RRDB and timestep embeddings for diffusion models.
    """

    """
    The job of the network is to take in a batch of noisy images and their respective noise levels, and output the noise added to the input.

    The network takes a batch of noisy images of shape
        (batch_size, num_channels, height, width) and a batch of noise levels of shape (batch_size, 1) as input,
    
    And returns a tensor of shape (batch_size, num_channels, height, width)

    Args:
        nn (_type_): _description_
    """
    def __init__(
        self,
        dim,
        init_dim=None,
        out_dim=None,
        dim_mults=(1, 2, 4, 8),
        channels=3,
        self_condition=False,
        resnet_block_groups=4,
    ):
        super().__init__()

        # determine dimensions
        self.channels = channels
        self.self_condition = self_condition
        input_channels = channels * (2 if self_condition else 1)

        init_dim = default(init_dim, dim)
        self.init_conv = nn.Conv2d(input_channels, init_dim, 1, padding=0) # changed to 1 and 0 from 7,3

        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        block_klass = partial(ResnetBlock, groups=resnet_block_groups)

        # time embeddings
        time_dim = dim * 4

        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(dim),
            nn.Linear(dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim),
        )

        # layers
        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            self.downs.append(
                nn.ModuleList(
                    [
                        block_klass(dim_in, dim_in, time_emb_dim=time_dim),
                        block_klass(dim_in, dim_in, time_emb_dim=time_dim),
                        Residual(PreNorm(dim_in, LinearAttention(dim_in))),
                        Downsample(dim_in, dim_out)
                        if not is_last
                        else nn.Conv2d(dim_in, dim_out, 3, padding=1),
                    ]
                )
            )

        mid_dim = dims[-1]
        self.mid_block1 = block_klass(mid_dim, mid_dim, time_emb_dim=time_dim)
        self.mid_attn = Residual(PreNorm(mid_dim, Attention(mid_dim)))
        self.mid_block2 = block_klass(mid_dim, mid_dim, time_emb_dim=time_dim)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out)):
            is_last = ind == (len(in_out) - 1)

            self.ups.append(
                nn.ModuleList(
                    [
                        block_klass(dim_out + dim_in, dim_out, time_emb_dim=time_dim),
                        block_klass(dim_out + dim_in, dim_out, time_emb_dim=time_dim),
                        Residual(PreNorm(dim_out, LinearAttention(dim_out))),
                        Upsample(dim_out, dim_in)
                        if not is_last
                        else nn.Conv2d(dim_out, dim_in, 3, padding=1),
                    ]
                )
            )

        self.out_dim = default(out_dim, channels)

        self.final_res_block = block_klass(dim * 2, dim, time_emb_dim=time_dim)
        self.final_conv = nn.Conv2d(dim, self.out_dim, 1)

    def forward(self, x, time, x_self_cond=None):
        if self.self_condition:
            x_self_cond = default(x_self_cond, lambda: torch.zeros_like(x))
            x = torch.cat((x_self_cond, x), dim=1)

        # first, a convolutional layer is applied on the batch of noisy images
        x = self.init_conv(x)
        r = x.clone()

        # and position embeddings are computed for the noise levels
        t = self.time_mlp(time)

        h = []

        # next, a sequence of downsampling stages are applied.
        # Each downsampling stage consists of 2 ResNet blocks + groupnorm + attention + residual connection + a downsample operation
        for block1, block2, attn, downsample in self.downs:
            x = block1(x, t)
            h.append(x)

            x = block2(x, t)
            x = attn(x)
            h.append(x)

            x = downsample(x)

        # at the middle of the network, again ResNet blocks are applied, interleaved with attention
        x = self.mid_block1(x, t)
        x = self.mid_attn(x)
        x = self.mid_block2(x, t)

        # next, a sequence of upsampling stages are applied.
        # Each upsampling stage consists of 2 ResNet blocks + groupnorm + attention + residual connection + an upsample operation
        for block1, block2, attn, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim=1)
            x = block1(x, t)

            x = torch.cat((x, h.pop()), dim=1)
            x = block2(x, t)
            x = attn(x)

            x = upsample(x)

        x = torch.cat((x, r), dim=1)

        # finally, a ResNet block followed by a convolutional layer is applied.
        x = self.final_res_block(x, t)
        return self.final_conv(x)
