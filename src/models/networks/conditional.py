import jax.numpy as jnp
from flax import linen as nn
from diffusers.models.unets.unet_2d_condition_flax import FlaxUNet2DConditionModel

class Network(nn.Module):
    sample_size: int = 4
    in_channels: int = 1
    out_channels: int = 1
    dropout_rate: float = 0.0
    block_out_channels = (64, 128, 256, 256)
    imin: int = 0
    imax: int = 120
    
    def setup(self):
        self.net = FlaxUNet2DConditionModel(
            sample_size=self.sample_size,
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            block_out_channels=self.block_out_channels,
            down_block_types=(
                "DownBlock2D",
                "DownBlock2D",
                "DownBlock2D",
                "DownBlock2D",
            ),
            up_block_types=(
                "UpBlock2D",
                "UpBlock2D",
                "UpBlock2D",
                "UpBlock2D",
            ),
            dropout=self.dropout_rate,
        )
    
    def __call__(self, x, i, c, is_training=False):
        B, H, W, C = x.shape
        x = jnp.reshape(x, (B, C, H, W))
        x = self.net(x, i, c, train=is_training)[0]
        x = jnp.reshape(x, (B, H, W, C))
        
        return x
