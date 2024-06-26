from typing import Tuple
from jax.image import resize
import jax.numpy as jnp
from flax import linen as nn

# The following network is:
# homemade


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
    num_groups: int = 8

    @nn.compact
    def __call__(self, x, temb):
        residual = x
        x = nn.GroupNorm(num_groups=self.num_groups)(x)
        x = nn.silu(x)
        x = nn.Conv(features=self.features, kernel_size=self.kernel_size)(x)
        
        # Project and add the time step embedding
        temb = nn.silu(temb)
        temb = nn.Dense(features=self.features)(temb)
        temb = jnp.expand_dims(temb, axis=(1, 2))
        x += temb
        
        x = nn.GroupNorm(num_groups=self.num_groups)(x)
        x = nn.silu(x)
        x = nn.Conv(features=self.features, kernel_size=self.kernel_size)(x)
        return x + residual
    
    
class DownSample(nn.Module):
    features: int
    kernel_size: Tuple[int] = (3, 3)
    strides: Tuple[int] = (2, 2)
    window_shape: Tuple[int] = (2, 2)
    num_groups: int = 8
    dropout_rate: float = 0.0

    @nn.compact
    def __call__(self, x, temb, deterministic=True):
        x = nn.Conv(features=self.features, kernel_size=self.kernel_size, strides=self.strides)(x)
        x = ResNetBlock(features=self.features, kernel_size=self.kernel_size, num_groups=self.num_groups)(x, temb)
        x = nn.Dropout(self.dropout_rate, deterministic=deterministic)(x)
        return x


class UpSample(nn.Module):
    features: int
    kernel_size: Tuple[int] = (3, 3)
    strides: Tuple[int] = (2, 2)
    num_groups: int = 8
    scale: int = 2
    method: str = 'nearest'
    dropout_rate: float = 0.0

    @nn.compact
    def __call__(self, x, skip, temb, deterministic=True):
        x = jnp.concatenate([x, skip], axis=-1)
        new_shape = (x.shape[1] * self.scale, x.shape[2] * self.scale)
        x = resize(x, shape=(x.shape[0], *new_shape, x.shape[-1]), method=self.method)
        x = nn.Conv(features=self.features, kernel_size=self.kernel_size)(x)
        x = ResNetBlock(features=self.features, kernel_size=self.kernel_size, num_groups=self.num_groups)(x, temb)
        x = nn.Dropout(self.dropout_rate, deterministic=deterministic)(x)
        return x


class Network(nn.Module):
    kernel_size: Tuple[int] = (3, 3)
    strides: Tuple[int] = (2, 2)
    window_shape: Tuple[int] = (2, 2)
    num_groups: int = 8
    features: Tuple[int] = (128, 128, 256, 256)
    dropout_rate: float = 0.0
    attention_features: int = 256
    num_heads: int = 8
    imin: float = 0.002
    imax: float = 80.0
    embedding_dim: int = 16
    
    def setup(self):
        # Downsample layers
        self.downsample1 = DownSample(features=self.features[0], kernel_size=self.kernel_size, strides=self.strides)
        self.downsample2 = DownSample(features=self.features[1], kernel_size=self.kernel_size, strides=self.strides)
        self.downsample3 = DownSample(features=self.features[2], kernel_size=self.kernel_size, strides=self.strides)
        self.downsample4 = DownSample(features=self.features[3], kernel_size=self.kernel_size, strides=self.strides)
        
        # Attention
        self.attention = nn.MultiHeadDotProductAttention(num_heads=self.num_heads, qkv_features=self.attention_features)
        
        # Upsampling layers
        self.upsample1 = UpSample(features=self.features[-1], kernel_size=self.kernel_size, strides=self.strides)
        self.upsample2 = UpSample(features=self.features[-2], kernel_size=self.kernel_size, strides=self.strides)
        self.upsample3 = UpSample(features=self.features[-3], kernel_size=self.kernel_size, strides=self.strides)
        self.upsample4 = UpSample(features=self.features[-4], kernel_size=self.kernel_size, strides=self.strides)
        
        # Output layer
        self.output_conv = nn.Conv(features=1, kernel_size=self.kernel_size)
        

    @nn.compact
    def __call__(self, x, t, is_training=False):
        # Time embedding
        embedding = SinusoidalPositionalEmbedding(embedding_dim=self.embedding_dim)(t)
        temb = nn.Dense(features=2*self.embedding_dim)(embedding)
        temb = nn.silu(temb)
        temb = nn.Dense(features=2*self.embedding_dim)(temb)

        # Downsampling
        d1 = self.downsample1(x, temb, deterministic=not is_training)
        d2 = self.downsample2(d1, temb, deterministic=not is_training)
        d3 = self.downsample3(d2, temb, deterministic=not is_training)
        d4 = self.downsample4(d3, temb, deterministic=not is_training)
        
        # Attention
        attention_mask = self.attention(d4)
        x = d4 + attention_mask
        
        # Upsampling
        x = self.upsample1(x, d4, temb, deterministic=not is_training)
        x = self.upsample2(x, d3, temb, deterministic=not is_training)
        x = self.upsample3(x, d2, temb, deterministic=not is_training)
        x = self.upsample4(x, d1, temb, deterministic=not is_training)
        
        # Output layer
        x = self.output_conv(x)
        
        return x
    