import jax.numpy as jnp
from utils import batch_mul
from flax import linen as nn


class ConsistencyModel():
    def __init__(self, std_data, min_noise, F):
        self.std_data: float = std_data
        self.min_noise: float = min_noise
        self.F: nn.Module = F
        
    
    def _c_skip(self, noise):
        return self.std_data**2 / ((noise-self.min_noise)**2 + self.std_data**2)


    def _c_out(self, noise):
        # took out the - min_noise following hess
        return self.std_data*(noise) / jnp.sqrt(self.std_data**2 + noise**2)
    
    
    def apply(self, params, x, c, noise, i, is_training=False, rngs=None):
        r = batch_mul(self._c_skip(noise), x) + batch_mul(self._c_out(noise), self.F.apply(params, x, i, c, is_training=is_training, rngs=rngs))
        # r = jnp.clip(r, -self.std_data*2, self.std_data*2)
        return r