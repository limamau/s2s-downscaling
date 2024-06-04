import jax
import optax
import time

import jax.numpy as jnp
from functools import partial

from .losses import consistency_loss
from utils import batch_add, batch_mul, create_folder, log_transform, normalize_data
from .writing import Writer
from .checkpointing import Checkpointer
from models.consistency_model import ConsistencyModel


@partial(jax.jit, static_argnums=(0,1,2,3,4,5))
def train_step(
    cm, # static
    d, # static
    opt, # static
    N, # static
    mu, # static
    tn, # static
    online_params,
    target_params,
    opt_state,
    x,
    n,
    z,
    k,
):
    # Get times (noises) and noised images
    t_online = tn(n+1, N(k))
    x_online = batch_add(x, batch_mul(t_online, z))
    t_target = tn(n, N(k))
    x_target = batch_add(x, batch_mul(t_target, z))
    
    # Get loss and gradients (with respect to online_params)
    loss, grads = jax.value_and_grad(consistency_loss, argnums=0)(
        online_params,
        target_params,
        d,
        cm,
        x_online, t_online,
        x_target, t_target,
    )
    
    # Update parameters and optimizer state
    params_updates, opt_state = opt.update(grads, opt_state)
    online_params = optax.apply_updates(online_params, params_updates)
    target_params = jax.lax.stop_gradient(
        jax.tree.map(
            lambda x,y: x + y,
            jax.tree.map(lambda p: p*mu(k), target_params),
            jax.tree.map(lambda p: p*(1-mu(k)), online_params)
        )
    )
    
    return loss, online_params, target_params, opt_state


def train(experiment):
    # Unpack experiment
    experiment_dir = experiment.experiment_dir
    data = experiment.data
    net = experiment.network
    d = experiment.distance
    opt = experiment.optimizer
    
    batch_size = experiment.batch_size
    tmin = experiment.tmin
    K = experiment.training_iterations
    N = experiment.N
    mu = experiment.mu
    tn = experiment.tn
    sigma_data = experiment.sigma_data
    
    log_each = experiment.log_each
    ckpt_each = experiment.ckpt_each
    
    # Logs and checkpoints
    create_folder(experiment_dir, overwrite=True)
    writer = Writer(experiment_dir)
    writer.log_and_save_config(experiment)
    writer.log_device()
    ckpter = Checkpointer(experiment_dir)
    
    # Adjust dimensions
    Nt, Ny, Nx = experiment.data.shape
    Ny -= Ny % 16
    Nx -= Nx % 16
    data = experiment.data[:Nt, :Ny, :Nx]
    data = jnp.expand_dims(data, axis=(3,))
    
    # Normalization
    data = log_transform(data)
    data, data_mean, data_std = normalize_data(data, sigma_data)
    writer.save_normalizations(data.shape, data_mean, data_std)
    
    # Initialisations
    cm = ConsistencyModel(sigma_data, tmin, net)
    online_params = net.init(jax.random.PRNGKey(42), jnp.ones((batch_size, Ny, Nx, 1)))
    target_params = online_params
    opt_state = opt.init(target_params)
    
    # PRNG for sampling
    rng = jax.random.PRNGKey(37)
    
    # Initialize timers
    timer_samples = 0
    timer_train_step = 0
    timer_log_ckpt = 0

    # Training loop
    for k in range(0, K+1, batch_size):
        # Samples
        start_time = time.time()
        rng, x_rng, n_rng, z_rng = jax.random.split(rng, 4)
        idx = jax.random.randint(x_rng, (batch_size,), 0, Nt)
        x = data[idx, :,:,:]
        n = jax.random.randint(n_rng, (batch_size,), 1, N(k)+1)
        z = jax.random.normal(z_rng, (batch_size, Ny, Nx, 1))
        timer_samples += time.time() - start_time
        
        # Train step with JIT
        start_time = time.time()
        loss, online_params, target_params, opt_state = train_step(
            cm,
            d,
            opt,
            N,
            mu,
            tn,
            online_params,
            target_params,
            opt_state,
            x,
            n,
            z,
            k,
        )
        timer_train_step += time.time() - start_time
        
        # Log, checkpoint, write loss
        start_time = time.time()
        if k == 0:
            continue
        if k % log_each == 0:
            writer.log_loss(k, loss)
        if k % ckpt_each == 0:
            ckpter.save(k, target_params, opt_state)
        timer_log_ckpt += time.time() - start_time
            
    # Save last if it wasn't already
    if K % ckpt_each != 0:
        writer.log_loss(K, loss)
        ckpter.save(K, target_params, opt_state)

    # I intend to take the time control out in the future
    writer.logger.info("Time spent in each block:")
    writer.logger.info("Samples: {:.3f}s".format(timer_samples))
    writer.logger.info("Train step with JIT: {:.3f}s".format(timer_train_step))
    writer.logger.info("Log, checkpoint, write loss: {:.3f}s".format(timer_log_ckpt))
    
    writer.close()
    ckpter.close()