import jax
import jax.numpy as jnp
import lpips_jax


def l1(x, y):
    return jnp.mean(jnp.abs(x - y))


def l2(x, y):
    return jnp.mean((x - y)**2)


def pseudo_huber(x, y, c=0.1):
    return jnp.sqrt(l2(x,y) + c**2) - c


def lpips(x, y):
    shape = (1, 224, 224, 1)
    x = jax.image.resize(x, shape, method='bilinear')
    y = jax.image.resize(y, shape, method='bilinear')
    return jnp.mean(lpips_jax.LPIPSEvaluator(net='vgg16'))
    