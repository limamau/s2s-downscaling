import jax
import jax.numpy as jnp

from functools import partial
from utils import batch_mul


class ConsistencyModel():
    def __init__(self, sigma_data, tmin, F):
        self.sigma_data = sigma_data
        self.tmin = tmin
        self.F = F
        
    
    @partial(jax.jit, static_argnums=0)
    def _cskip(self, t):
        return self.sigma_data**2 / (self.sigma_data**2 + (t-self.tmin)**2)


    @partial(jax.jit, static_argnums=0)
    def _cout(self, t):
        return self.sigma_data*(t-self.tmin) / jnp.sqrt(self.sigma_data**2 + t**2)
    
    
    @partial(jax.jit, static_argnums=0)
    def apply(self, params, x, t,):
        return batch_mul(self._cskip(t), x) + batch_mul(self._cout(t), self.F.apply(params, x))