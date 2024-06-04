import jax
import jax.numpy as jnp
from functools import partial


@partial(jax.jit, static_argnums=(2,3))
def consistency_loss(
    online_params, 
    target_params,
    d, # static
    cm, # static
    x_online, t_online,
    x_target, t_target,
):
    return jnp.mean(
        d(
            cm.apply(online_params, x_online, t_online), 
            cm.apply(target_params, x_target, t_target)
        )
    )
