import jax
import jax.numpy as jnp
import numpy as np
from .metrics import psd_distance, cdf_distance, perkins_skill_score


def evaluate(
    model,
    params,
    x,
    c,
    rng,
    steps,
    batch_size=1,
    t_star=1,
):
    obs = x.__array__()
    sim = np.zeros(obs.shape)

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
        ).__array__()
        print(f"Generated {end_idx} samples")
        
    return obs, sim


def get_metrics(obs, sim, n_quantiles=100):
    # Take channel out
    obs = obs[..., 0]
    sim = sim[..., 0]
    
    # Get metrics
    psd = psd_distance(
        "l2",
        obs, obs.shape[2], obs.shape[1], # dummy distances
        sim,
    )
    cdf = cdf_distance(
        "l2",
        obs,
        sim,
        n_quantiles=n_quantiles,
    )
    pss = perkins_skill_score(
        obs,
        sim,
        n_quantiles=n_quantiles,
    )

    return psd, cdf, pss