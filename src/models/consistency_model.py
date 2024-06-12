import jax
import jax.numpy as jnp

from functools import partial
from utils import batch_mul


class ConsistencyModel():
    def __init__(self, sigma_data, tmin, F):
        self.sigma_data = sigma_data
        self.tmin = tmin
        self.F = F
        
    
    def _c_skip(self, t):
        return self.sigma_data**2 / (self.sigma_data**2 + (t-self.tmin)**2)


    def _c_out(self, t):
        return self.sigma_data*(t-self.tmin) / jnp.sqrt(self.sigma_data**2 + t**2)
    
    
    def apply(self, params, x, t, is_training=False, rngs=None):
        return batch_mul(self._c_skip(t), x) + batch_mul(self._c_out(t), self.F.apply(params, x, t, is_training=is_training, rngs=rngs))