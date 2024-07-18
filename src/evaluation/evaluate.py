import jax
import jax.numpy as jnp
import numpy as np
from .metrics import logpsd_distance, logcdf_distance, perkins_skill_score


def evaluate(model, params, x, c, rng, batch_size=1, steps=50, t_star=1):
    obs = x.__array__()[:,:,:,0]
    sim = np.empty_like(obs)

    for start_idx in range(0, x.shape[0], batch_size):
        # Get batch
        end_idx = start_idx + batch_size
        if c is not None:
            batch = x[start_idx:end_idx], c[start_idx:end_idx]
        else:
            batch = x[start_idx:end_idx], None
        
        # Generate samples using the model
        rng, sample_rng = jax.random.split(rng)
        sim[start_idx:end_idx] = model.sample(
            params, sample_rng, batch, batch_size, steps
        ).__array__()[:,:,:,0]
        
    # Compute validation score
    psd_distance = logpsd_distance(
        "l2",
        obs, obs.shape[2], obs.shape[1], # dummy distances
        sim,
    )
    cdf_distance = logcdf_distance(
        "l2",
        obs,
        sim,
        n_quantiles=100,
    )
    pss = perkins_skill_score(
        obs,
        sim,
        n_quantiles=100,
    )

    return psd_distance, cdf_distance, pss