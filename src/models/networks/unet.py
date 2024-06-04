from typing import Tuple
from flax import linen as nn

from dataclasses import dataclass
from jax.tree_util import register_pytree_node_class


@register_pytree_node_class
@dataclass
class UNet(nn.Module):
    kernel_size: Tuple[int] = (3, 3)
    strides: Tuple[int] = (2, 2)
    window_shape: Tuple[int] = (2, 2)
    num_groups: int = 8

    @nn.compact
    def __call__(self, x):
        # Downsample 1
        x = nn.Conv(features=128, kernel_size=self.kernel_size, padding='SAME')(x)
        x = nn.GroupNorm(num_groups=self.num_groups)(x)
        x = nn.silu(x)
        x = nn.max_pool(x, window_shape=self.window_shape, strides=self.strides)
        
        # Downsample 2
        x = nn.Conv(features=128, kernel_size=self.kernel_size, padding='SAME')(x)
        x = nn.GroupNorm(num_groups=self.num_groups)(x)
        x = nn.silu(x)
        x = nn.max_pool(x, window_shape=self.window_shape, strides=self.strides)
        
        # Downsample 3
        x = nn.Conv(features=256, kernel_size=self.kernel_size, padding='SAME')(x)
        x = nn.GroupNorm(num_groups=self.num_groups)(x)
        x = nn.silu(x)
        x = nn.max_pool(x, window_shape=self.window_shape, strides=self.strides)
        
        # Downsample 4
        x = nn.Conv(features=256, kernel_size=self.kernel_size, padding='SAME')(x)
        x = nn.GroupNorm(num_groups=self.num_groups)(x)
        x = nn.silu(x)
        x = nn.max_pool(x, window_shape=self.window_shape, strides=self.strides)
        
        # Attention is all you need
        x = nn.MultiHeadDotProductAttention(num_heads=8, qkv_features=256)(x)
        
        # Upsample 1
        x = nn.ConvTranspose(features=256, kernel_size=self.kernel_size, strides=self.strides)(x)
        x = nn.GroupNorm(num_groups=self.num_groups)(x)
        x = nn.silu(x)
        
        # Upsample 2
        x = nn.ConvTranspose(features=128, kernel_size=self.kernel_size, strides=self.strides)(x)
        x = nn.GroupNorm(num_groups=self.num_groups)(x)
        x = nn.silu(x)
        
        # Upsample 3
        x = nn.ConvTranspose(features=128, kernel_size=self.kernel_size, strides=self.strides)(x)
        x = nn.GroupNorm(num_groups=self.num_groups)(x)
        x = nn.silu(x)
        
        # Upsample 4
        x = nn.ConvTranspose(features=1, kernel_size=self.kernel_size, strides=self.strides)(x)
        x = nn.GroupNorm(num_groups=1)(x)
        x = nn.silu(x)
        return x
    
    
    def tree_flatten(self):
        children = ()  # no children, since this is a nn.Module
        aux_data = {
            'kernel_size': self.kernel_size,
            'strides': self.strides,
            'window_shape': self.window_shape,
            'num_groups': self.num_groups
        }
        return children, aux_data


    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(**aux_data)
