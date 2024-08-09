# adopted from
# https://github.com/openai/improved-diffusion/blob/main/improved_diffusion/gaussian_diffusion.py
# and
# https://github.com/lucidrains/denoising-diffusion-pytorch/blob/7706bdfc6f527f58d33f84b7b522e61e6e3164b3/denoising_diffusion_pytorch/denoising_diffusion_pytorch.py
# and
# https://github.com/openai/guided-diffusion/blob/0ba878e517b276c45d1195eb29f6f5f72659a05b/guided_diffusion/nn.py
#
# thanks!


import os
import math
import torch

import numpy as np
import torch.nn as nn

from einops import repeat, rearrange


#################################################################################
#                                  Unet Utils                                   #
#################################################################################

def checkpoint(func, inputs, params, flag):
    """
    Evaluate a function without caching intermediate activations, allowing for
    reduced memory at the expense of extra compute in the backward pass.
    :param func: the function to evaluate.
    :param inputs: the argument sequence to pass to `func`.
    :param params: a sequence of parameters `func` depends on but does not
                   explicitly take as arguments.
    :param flag: if False, disable gradient checkpointing.
    """
    if flag:
        args = tuple(inputs) + tuple(params)
        return CheckpointFunction.apply(func, len(inputs), *args)
    else:
        return func(*inputs)


class CheckpointFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, run_function, length, *args):
        ctx.run_function = run_function
        ctx.input_tensors = list(args[:length])
        ctx.input_params = list(args[length:])

        with torch.no_grad():
            output_tensors = ctx.run_function(*ctx.input_tensors)
        return output_tensors

    @staticmethod
    def backward(ctx, *output_grads):
        ctx.input_tensors = [x.detach().requires_grad_(True) for x in ctx.input_tensors]
        with torch.enable_grad():
            # Fixes a bug where the first op in run_function modifies the
            # Tensor storage in place, which is not allowed for detach()'d
            # Tensors.
            shallow_copies = [x.view_as(x) for x in ctx.input_tensors]
            output_tensors = ctx.run_function(*shallow_copies)
        input_grads = torch.autograd.grad(
            output_tensors,
            ctx.input_tensors + ctx.input_params,
            output_grads,
            allow_unused=True,
        )
        del ctx.input_tensors
        del ctx.input_params
        del output_tensors
        return (None, None) + input_grads


def timestep_embedding(timesteps, dim, max_period=10000, repeat_only=False):
    """
    Create sinusoidal timestep embeddings.
    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """
    if not repeat_only:
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=timesteps.device)
        args = timesteps[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    else:
        embedding = repeat(timesteps, 'b -> b d', d=dim).contiguous()
    return embedding


def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module


def scale_module(module, scale):
    """
    Scale the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().mul_(scale)
    return module


def mean_flat(tensor):
    """
    Take the mean over all non-batch dimensions.
    """
    return tensor.mean(dim=list(range(1, len(tensor.shape))))


def normalization(channels):
    """
    Make a standard normalization layer.
    :param channels: number of input channels.
    :return: an nn.Module for normalization.
    """
    return GroupNorm32(32, channels)


# PyTorch 1.7 has SiLU, but we support PyTorch 1.5.
class SiLU(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


class GroupNorm32(nn.GroupNorm):
    def forward(self, x):
        return super().forward(x.float()).type(x.dtype)
    
class CameraEmbedder(nn.Module):
    def __init__(self, extrinsics_dim=12, embed_dim=256):
        """
        Initializes the CameraEmbedder with given dimensions for intrinsics and extrinsics.
        
        :param extrinsics_dim: Dimension of the camera extrinsics (default: 12).
        :param embed_dim: Dimension of the output embedding (default: 256).
        """
        super().__init__()
        self.extrinsics_dim = extrinsics_dim
        self.embed_dim = embed_dim

        # Define a single linear layer for combined intrinsics and extrinsics
        self.fc = nn.Linear(extrinsics_dim, embed_dim)
        self.activation = nn.ReLU()

    def forward(self, cam_matrix, num_patches=None):
        """
        Forward pass to embed camera intrinsics and extrinsics.
        
        :param cam_param: Tensor of shape (N, M, extrinsics_dim).
        :return: Tensor of shape (N * num_patches, M, embed_dim) representing the embedded camera parameters.
        """
        # Pass through the linear layer and activation function
        embedding = self.activation(self.fc(cam_matrix))
        if num_patches is not None:
            embedding = embedding.unsqueeze(1).repeat(1, num_patches, 1, 1)
            embedding = rearrange(embedding, 'b t f d->(b t) f d')
        return embedding
    
class PositionalEncoding(nn.Module):
    def __init__(self, num_octaves=8, start_octave=0):
        super().__init__()
        self.num_octaves = num_octaves
        self.start_octave = start_octave

    def forward(self, coords, rays=None):
        embed_fns = []
        batch_size, num_points, dim = coords.shape

        octaves = torch.arange(self.start_octave, self.start_octave + self.num_octaves)
        octaves = octaves.float().to(coords)
        multipliers = 2**octaves * math.pi
        coords = coords.unsqueeze(-1)
        while len(multipliers.shape) < len(coords.shape):
            multipliers = multipliers.unsqueeze(0)

        scaled_coords = coords * multipliers

        sines = torch.sin(scaled_coords).reshape(batch_size, num_points, dim * self.num_octaves)
        cosines = torch.cos(scaled_coords).reshape(batch_size, num_points, dim * self.num_octaves)

        result = torch.cat((sines, cosines), -1)
        return result


class RayEncoder(nn.Module):
    def __init__(self, pos_octaves=8, pos_start_octave=0, ray_octaves=4, ray_start_octave=0, output_channels=256):
        super().__init__()
        self.pos_encoding = PositionalEncoding(num_octaves=pos_octaves, start_octave=pos_start_octave)
        self.ray_encoding = PositionalEncoding(num_octaves=ray_octaves, start_octave=ray_start_octave)
        self.conv_layers = nn.Sequential(
            nn.Conv2d(6 * (pos_octaves + ray_octaves), 256, kernel_size=3, stride=4, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, output_channels, kernel_size=3, stride=4, padding=1),
            nn.ReLU()
        )
        self.embed_dim = output_channels

    def forward(self, pos, rays):
        b, f, h, w, c = rays.shape
        rays = rays.flatten(0, 1)
        pos = pos.flatten(0, 1)

        batchsize, height, width, dims = rays.shape
        pos_enc = self.pos_encoding(pos.unsqueeze(1))
        pos_enc = pos_enc.view(batchsize, pos_enc.shape[-1], 1, 1)
        pos_enc = pos_enc.repeat(1, 1, height, width)
        rays = rays.flatten(1, 2)

        ray_enc = self.ray_encoding(rays)
        ray_enc = ray_enc.view(batchsize, height, width, ray_enc.shape[-1])
        ray_enc = ray_enc.permute((0, 3, 1, 2))
        x = torch.cat((pos_enc, ray_enc), 1)

        x = self.conv_layers(x)
        x = rearrange(x, '(b f) c h w -> (b h w) f c', b=b)

        return x

def conv_nd(dims, *args, **kwargs):
    """
    Create a 1D, 2D, or 3D convolution module.
    """
    if dims == 1:
        return nn.Conv1d(*args, **kwargs)
    elif dims == 2:
        return nn.Conv2d(*args, **kwargs)
    elif dims == 3:
        return nn.Conv3d(*args, **kwargs)
    raise ValueError(f"unsupported dimensions: {dims}")


def linear(*args, **kwargs):
    """
    Create a linear module.
    """
    return nn.Linear(*args, **kwargs)


def avg_pool_nd(dims, *args, **kwargs):
    """
    Create a 1D, 2D, or 3D average pooling module.
    """
    if dims == 1:
        return nn.AvgPool1d(*args, **kwargs)
    elif dims == 2:
        return nn.AvgPool2d(*args, **kwargs)
    elif dims == 3:
        return nn.AvgPool3d(*args, **kwargs)
    raise ValueError(f"unsupported dimensions: {dims}")


# class HybridConditioner(nn.Module):

#     def __init__(self, c_concat_config, c_crossattn_config):
#         super().__init__()
#         self.concat_conditioner = instantiate_from_config(c_concat_config)
#         self.crossattn_conditioner = instantiate_from_config(c_crossattn_config)

#     def forward(self, c_concat, c_crossattn):
#         c_concat = self.concat_conditioner(c_concat)
#         c_crossattn = self.crossattn_conditioner(c_crossattn)
#         return {'c_concat': [c_concat], 'c_crossattn': [c_crossattn]}


def noise_like(shape, device, repeat=False):
    repeat_noise = lambda: torch.randn((1, *shape[1:]), device=device).repeat(shape[0], *((1,) * (len(shape) - 1)))
    noise = lambda: torch.randn(shape, device=device)
    return repeat_noise() if repeat else noise()

def count_flops_attn(model, _x, y):
    """
    A counter for the `thop` package to count the operations in an
    attention operation.
    Meant to be used like:
        macs, params = thop.profile(
            model,
            inputs=(inputs, timestamps),
            custom_ops={QKVAttention: QKVAttention.count_flops},
        )
    """
    b, c, *spatial = y[0].shape
    num_spatial = int(np.prod(spatial))
    # We perform two matmuls with the same number of ops.
    # The first computes the weight matrix, the second computes
    # the combination of the value vectors.
    matmul_ops = 2 * b * (num_spatial ** 2) * c
    model.total_ops += torch.DoubleTensor([matmul_ops])

def count_params(model, verbose=False):
    total_params = sum(p.numel() for p in model.parameters())
    if verbose:
        print(f"{model.__class__.__name__} has {total_params * 1.e-6:.2f} M params.")
    return total_params