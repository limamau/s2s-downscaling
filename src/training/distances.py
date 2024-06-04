import jax.numpy as jnp


def l1(x, y):
    return jnp.mean(jnp.abs(x - y), axis=(1, 2, 3))


def l2(x, y):
    return jnp.sqrt(jnp.mean((x - y)**2, axis=(1, 2, 3)))


# TODO:
def lpips(x, y):
    pass