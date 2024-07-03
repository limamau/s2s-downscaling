from typing import Tuple
from flax import linen as nn

import jax.numpy as jnp
import math

from diffusers.models.embeddings_flax import get_sinusoidal_embeddings, FlaxTimestepEmbedding
from diffusers.models.unets.unet_2d_blocks_flax import FlaxDownBlock2D, FlaxUpBlock2D, FlaxUNetMidBlock2DCrossAttn


# The following function is ported over from the DDPM codebase:
#  https://github.com/hojonathanho/diffusion/blob/master/diffusion_tf/nn.py
def get_sinusoidal_embeddings_ddpm(timesteps, embedding_dim, max_positions=10000):
    assert len(timesteps.shape) == 1
    half_dim = embedding_dim // 2
    # magic number 10000 is from transformers
    emb = math.log(max_positions) / (half_dim - 1)
    emb = jnp.exp(jnp.arange(half_dim, dtype=jnp.float32) * -emb)
    # emb = tf.range(num_embeddings, dtype=jnp.float32)[:, None] * emb[None, :]
    # emb = tf.cast(timesteps, dtype=jnp.float32)[:, None] * emb[None, :]
    emb = timesteps[:, None] * emb[None, :]
    emb = jnp.concatenate([jnp.sin(emb), jnp.cos(emb)], axis=1)
    if embedding_dim % 2 == 1:  # zero pad
        emb = jnp.pad(emb, [[0, 0], [0, 1]])
    assert emb.shape == (timesteps.shape[0], embedding_dim)
    return emb


# The following network is inspired in:
# https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/unets/unet_2d.py
# but using the blocks unet_2d_blocks_flax.py and some modifications

class Network(nn.Module):
    features: Tuple[int] = (64, 128, 256, 512)
    kernel_size = (3,3)
    layers_per_block: int = 2
    dropout_rate: float = 0.0
    attention_heads: int = 8
    norm_num_groups: int = 32
    iemb_dim: int = 256
    imin: float = 1
    imax: float = 120

    def setup(self):
        # Noise (time) embedding
        # self.iproj = lambda i: get_sinusoidal_embeddings(
        #     i,
        #     embedding_dim=self.features[0],
        #     min_timescale=self.imin,
        #     max_timescale=self.imax,
        #     scale=1.0,
        # )
        self.iproj = lambda i: get_sinusoidal_embeddings_ddpm(
            i,
            embedding_dim=self.features[0],
        )
        self.iemb = FlaxTimestepEmbedding(self.iemb_dim)
        
        # Input
        self.conv_in = nn.Conv(self.features[0], self.kernel_size, padding='SAME')

        # Downsample
        down_blocks = []
        for i in range(4):
            input_channel = self.features[i-1] if i > 0 else self.features[0]
            output_channel = self.features[i]
            is_final_block = i == 3

            down_block = FlaxDownBlock2D(
                in_channels=input_channel,
                out_channels=output_channel,
                dropout=self.dropout_rate,
                num_layers=self.layers_per_block,
                add_downsample=not is_final_block,
            )
            down_blocks.append(down_block)
        self.down_blocks = down_blocks

        # Middle
        self.mid_block = FlaxUNetMidBlock2DCrossAttn(
            in_channels=self.features[-1],
            dropout=self.dropout_rate,
            num_layers=self.layers_per_block,
            num_attention_heads=self.attention_heads,
            use_linear_projection=False, # default
            use_memory_efficient_attention=False, # default
            split_head_dim=True, # to speed up
            transformer_layers_per_block=1, # default
        )

        # Upsample
        up_blocks = []
        for i in range(4):
            input_channel = self.features[3-i]
            output_channel = self.features[2-i] if i < 3 else self.features[0]
            is_final_block = i == 3

            up_block = FlaxUpBlock2D(
                in_channels=input_channel,
                out_channels=output_channel,
                prev_output_channel=self.features[3-i-1],
                dropout=self.dropout_rate,
                num_layers=self.layers_per_block,
                add_upsample=not is_final_block,
            )
            up_blocks.append(up_block)
        self.up_blocks = up_blocks

        # Output
        self.conv_norm_out = nn.GroupNorm(num_groups=self.norm_num_groups)
        self.conv_act = nn.silu
        self.conv_out = nn.Conv(1, self.kernel_size, padding='SAME')

    def __call__(self, x, i, c, is_training=False):
        # Noise (time) embedding
        iemb = self.iproj(i)
        iemb = self.iemb(iemb)

        # Lifting
        x = self.conv_in(x)

        # Downsample
        down_block_residuals = (x,)
        for down_block in self.down_blocks:
            x, residuals = down_block(x, iemb, deterministic=not is_training)
            down_block_residuals += residuals

        # Middle
        # context is being passed as None, so the hidden states will be used instead (self-attention)
        # this can be a nice window to use topography and/or soil moisture as context
        x = self.mid_block(x, iemb, c, deterministic=not is_training)

        # Upsample
        for up_block in self.up_blocks:
            residuals = down_block_residuals[-(self.layers_per_block+1):]
            down_block_residuals = down_block_residuals[:-(self.layers_per_block+1)]
            x = up_block(x, residuals, iemb, deterministic=not is_training)

        # Output
        x = self.conv_norm_out(x)
        x = self.conv_act(x)
        x = self.conv_out(x)
        
        return x
