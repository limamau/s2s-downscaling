from typing import Tuple
import jax
import jax.numpy as jnp
from flax import linen as nn
from functools import partial

# The following network is inspired in:
# https://github.com/CliMA/CliMAgen.jl/blob/main/src/networks.jl
# but using Flax instead of Flux and some modifications


class SinusoidalPositionalEmbedding(nn.Module):
    embedding_dim: int
    scale: float = 16.0
    
    def __call__(self, noise):
        position = jnp.arange(0, self.embedding_dim, 2)
        div_term = jnp.exp(position * -(jnp.log(self.scale) / self.embedding_dim))
        pos = noise[:, None] * div_term[None, :]
        return jnp.concatenate([jnp.sin(pos), jnp.cos(pos)], axis=-1)


class ResNetBlock(nn.Module):
    features: int
    kernel_size: Tuple[int] = (3, 3)
    num_groups: int = 32
    dropout_rate: float = 0.0

    def setup(self):
        self.gnorm1 = nn.Sequential([nn.GroupNorm(num_groups=self.num_groups), nn.swish])
        self.conv1 = nn.Conv(features=self.features, kernel_size=self.kernel_size)
        self.emb_proj = nn.Dense(self.features)
        self.gnorm2 = nn.Sequential([nn.GroupNorm(self.num_groups), nn.swish])
        self.conv2 = nn.Conv(features=self.features, kernel_size=self.kernel_size)
        self.dropout = nn.Dropout(self.dropout_rate)

    @nn.compact
    def __call__(self, x, temb, deterministic=True):
        residual = x
        
        x = self.gnorm1(x)
        x = self.conv1(x)
        
        temb = self.emb_proj(temb)
        x += jnp.expand_dims(temb, axis=(1, 2))
        
        x = self.gnorm2(x)
        x = self.dropout(x, deterministic=deterministic)
        x = self.conv2(x)
        
        return x + residual
    

class Upsampling(nn.Module):
    features: int
    kernel_size: Tuple[int]
    scale: int = 2
    method: str = "nearest"
    
    @nn.compact
    def __call__(self, x):
        new_shape = (x.shape[1] * self.scale, x.shape[2] * self.scale)
        x = jax.image.resize(x, shape=(x.shape[0], *new_shape, x.shape[-1]), method=self.method)
        x = nn.Conv(features=self.features, kernel_size=self.kernel_size)(x)
        return x


class Network(nn.Module):
    features: Tuple[int] = (32, 64, 128, 256)
    num_resnet_blocks: int = 4
    dropout_rate: int = 0.0
    num_heads: int = 8
    num_qkv_features: int = 256
    emb_dim: int = 256
    kernel_size: Tuple[int] = (3, 3)
    strides: Tuple[int] = (2, 2)
    num_groups: int = 32
    out_features: int = 1

    def setup(self):
        # Embedding
        self.sin_pos_encoding = SinusoidalPositionalEmbedding(self.emb_dim)
        self.linear = nn.Dense(self.emb_dim)
        
        # Lifting
        self.down_conv1 = nn.Conv(features=self.features[0], kernel_size=self.kernel_size, strides=self.strides)
        self.down_dense1 = nn.Dense(self.features[0])
        self.down_gnorm1 = nn.Sequential([nn.GroupNorm(num_groups=4), nn.swish])
        
        # Downsample
        self.down_conv2 = nn.Conv(features=self.features[1], kernel_size=self.kernel_size, strides=self.strides)
        self.down_dense2 = nn.Dense(self.features[1])
        self.down_gnorm2 = nn.Sequential([nn.GroupNorm(num_groups=self.num_groups), nn.swish])
        
        self.down_conv3 = nn.Conv(features=self.features[2], kernel_size=self.kernel_size, strides=self.strides)
        self.down_dense3 = nn.Dense(self.features[2])
        self.down_gnorm3 = nn.Sequential([nn.GroupNorm(num_groups=self.num_groups), nn.swish])
        
        self.down_conv4 = nn.Conv(features=self.features[3], kernel_size=self.kernel_size, strides=self.strides)
        self.down_dense4 = nn.Dense(self.features[3])
        self.down_gnorm4 = nn.Sequential([nn.GroupNorm(num_groups=self.num_groups), nn.swish])
        
        # Middle
        self.resnet_blocks = [
            ResNetBlock(
                features=self.features[3], 
                kernel_size=self.kernel_size, 
                num_groups=self.num_groups, 
                dropout_rate=self.dropout_rate,
                )
            for _ in range(self.num_resnet_blocks)
        ]
        
        # Attention
        self.attention = nn.MultiHeadDotProductAttention(num_heads=8, qkv_features=self.num_qkv_features)
        
        # Upsample
        self.up_conv4 = Upsampling(features=self.features[3], kernel_size=self.kernel_size)
        self.up_dense4 = nn.Dense(self.features[3])
        self.up_gnorm4 = nn.Sequential([nn.GroupNorm(num_groups=self.num_groups), nn.swish])
        
        self.up_conv3 = Upsampling(features=self.features[2], kernel_size=self.kernel_size)
        self.up_dense3 = nn.Dense(self.features[2])
        self.up_gnorm3 = nn.Sequential([nn.GroupNorm(num_groups=self.num_groups), nn.swish])
        
        self.up_conv2 = Upsampling(features=self.features[1], kernel_size=self.kernel_size)
        self.up_dense2 = nn.Dense(self.features[1])
        self.up_gnorm2 = nn.Sequential([nn.GroupNorm(num_groups=self.num_groups), nn.swish])
        
        self.up_conv1 = Upsampling(features=self.out_features, kernel_size=self.kernel_size)


    @nn.compact
    def __call__(self, x, t, is_training=False):
        # Embedding
        temb = self.sin_pos_encoding(t)
        temb = self.linear(temb)

        # Downsample
        h1 = x
        
        h1 = self.down_conv1(h1)
        h1 += jnp.expand_dims(self.down_dense1(temb), axis=(1, 2))
        h1 = self.down_gnorm1(h1)
        
        h2 = self.down_conv2(h1)
        h2 += jnp.expand_dims(self.down_dense2(temb), axis=(1, 2))
        h2 = self.down_gnorm2(h2)
        
        h3 = self.down_conv3(h2)
        h3 += jnp.expand_dims(self.down_dense3(temb), axis=(1, 2))
        h3 = self.down_gnorm3(h3)
        
        h4 = self.down_conv4(h3)
        h4 += jnp.expand_dims(self.down_dense4(temb), axis=(1, 2))

        # Middle
        h = h4
        for i in range(self.num_resnet_blocks//2):
            h = self.resnet_blocks[i](h, temb, deterministic=not is_training)
        
        attention_mask = self.attention(h)
        h = h + attention_mask
        
        for i in range(self.num_resnet_blocks//2, self.num_resnet_blocks):
            h = self.resnet_blocks[i](h, temb)
        

        # Upsample
        h = self.down_gnorm4(h)
        h = self.up_conv4(h)
        h += jnp.expand_dims(self.up_dense4(temb), axis=(1, 2))
        h = self.up_gnorm4(h)
        
        h = self.up_conv3(jnp.concatenate([h, h3], axis=-1))
        h += jnp.expand_dims(self.up_dense3(temb), axis=(1, 2))
        h = self.up_gnorm3(h)
        
        h = self.up_conv2(jnp.concatenate([h, h2], axis=-1))
        h += jnp.expand_dims(self.up_dense2(temb), axis=(1, 2))
        h = self.up_gnorm2(h)
        
        # Projection
        h = self.up_conv1(jnp.concatenate([h, h1], axis=-1))
        print("done!")
        return h
