import jax
import jax.numpy as jnp
import numpy as np
from .metrics import logpsd_distance, logcdf_distance, perkins_skill_score
from utils import batch_mul

def multi_step_sampling(
    model,
    noise_schedule,
    params,
    rng,
    x,
    c,
    batch_size,
    current_N,
    start_idx,
    end_idx,
    num_steps=1,
):
    # Multi-step sampling
    x_i = x
    for i in range(start_idx, end_idx, -start_idx//num_steps):
        print("i:", i)
        # Current noise level
        idx = jnp.full((batch_size,), i)
        rng, z_rng = jax.random.split(rng)
        z = jax.random.normal(z_rng, x.shape)
        noise = noise_schedule(idx, current_N)**2
        x_i = x_i + batch_mul(noise, z)
        
        # Apply model to denoise
        x_i = model.apply(params, x_i, c, noise, idx)

    return x_i

def evaluate_model(
    model,
    noise_schedule,
    params, 
    val_data, 
    val_conditions, 
    batch_size,
    current_N,
    start_idx,
    end_idx=1,
    x_length = 362., # FIXME: hard coded
    y_length = 232., # FIXME: hard coded
    num_steps=1,
):
    rng = jax.random.PRNGKey(0)
    
    obs = val_data
    sim = np.zeros(obs.shape[:-1])
    
    for data_start_idx in range(0, len(val_data), batch_size):
        data_end_idx = data_start_idx + batch_size
        current_batch_size = min(batch_size, len(val_data) - data_start_idx)
        
        x = val_data[data_start_idx:data_end_idx]
        if val_conditions is not None:
            c = val_conditions[data_start_idx:data_end_idx]
        else:
            c = None
        
        x_generated = multi_step_sampling(
            model,
            noise_schedule,
            params,
            rng,
            x,
            c,
            current_batch_size,
            current_N,
            start_idx,
            end_idx,
            num_steps,
        )
        
        sim[data_start_idx:data_end_idx] = x_generated[:,:,:,0]
        rng, _ = jax.random.split(rng)
    
    # Compute validation score
    psd_distance = logpsd_distance(
        "l2",
        obs.__array__()[:,:,:,0], x_length, y_length,
        sim,
    )
    cdf_distance = logcdf_distance(
        obs.__array__()[:,:,:,0],
        sim,
        "l2",
    )
    pss = perkins_skill_score(
        obs.__array__()[:,:,:,0],
        sim,
    )
    
    return psd_distance, cdf_distance, pss
