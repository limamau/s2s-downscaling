import jax
import optax

import numpy as np
import jax.numpy as jnp
from flax.training.train_state import TrainState
from functools import partial

from utils import batch_mul, create_folder, process_data
from .writing import Writer
from .checkpointing import Checkpointer
from models.consistency_model import ConsistencyModel


@partial(jax.jit, static_argnums=(0,))
def train_step(
    cm,  # static
    loss_weight,
    mu,
    dropout_rng,
    state,
    target_params,
    x_online, online_noise, i_online,
    x_target, target_noise, i_target,
    c,
):
    # Define the loss function inside the train step
    def loss_fn(params):
        C = 0.3
        x = cm.apply(
            params, 
            x_online, c, online_noise, i_online, 
            is_training=True, 
            rngs={'dropout': dropout_rng}
        )
        y = cm.apply(
            target_params, 
            x_target, c, target_noise, i_target, 
            is_training=True, 
            rngs={'dropout': dropout_rng}
        )
        return jnp.mean(
            batch_mul(
                loss_weight, 
                jnp.sqrt(jnp.mean((x-y)**2, axis=(1,2,3,)) + C**2) - C,
            )
        )
    
    # Get loss and gradients (with respect to online_params)
    loss, grads = jax.value_and_grad(loss_fn)(state.params)
    
    # Update parameters and optimizer state
    state = state.apply_gradients(grads=grads)
    target_params = jax.lax.stop_gradient(
        jax.tree_map(
            lambda x, y: x + y,
            jax.tree_map(lambda p: p * mu, target_params),
            jax.tree_map(lambda p: p * (1 - mu), state.params)
        )
    )
    
    return loss, state, target_params

def train(experiment):
    # Unpack experiment
    experiment_dir = experiment.experiment_dir
    data = experiment.data
    conditions = experiment.conditions
    validation_ratio = experiment.validation_ratio
    net = experiment.network
    optimizer = experiment.optimizer
    learning_rate = experiment.learning_rate
    ema_rate = experiment.ema
    
    batch_size = experiment.batch_size
    min_noise = experiment.min_noise
    epochs = experiment.epochs
    N = experiment.discretization_steps
    mu = experiment.mu
    noise_schedule = experiment.noise_schedule
    loss_weight_fn = experiment.loss_weighting
    norm_mean = experiment.norm_mean
    norm_std = experiment.norm_std
    is_log_transforming = experiment.is_log_transforming
    
    log_each = experiment.log_each
    ckpt_each = experiment.ckpt_each
    
    # Logs and checkpoints
    create_folder(experiment_dir, overwrite=True)
    writer = Writer(experiment_dir, csv_args=["train_loss", "N", "mu"])
    writer.log_and_save_config(experiment)
    writer.log_device()
    ckpter = Checkpointer(experiment_dir)
    del experiment
    
    # Adjust dimensions
    Nt, Ny, Nx, _ = data.shape
    num_blocks = 4  # FIXME: hard coded
    Ny -= Ny % 2**(num_blocks)
    Nx -= Nx % 2**(num_blocks)
    Nt -= Nt % batch_size
    data = data[:Nt, :Ny, :Nx, :]
    
    # Normalization
    data, dataset_mean, dataset_std = process_data(data, None, None, norm_mean, norm_std, is_log_transforming)
    writer.save_normalizations(data.shape, dataset_mean, dataset_std)
    
    # PRNG
    rng = jax.random.PRNGKey(42)
    
    # Divide data into training and validation
    indices = np.arange(Nt)
    rng, idx_rng = jax.random.split(rng, 2)
    jax.random.permutation(idx_rng, indices, independent=True)
    r = 1 - validation_ratio
    training_indices = indices[:int(r * Nt)]
    validation_indices = indices[int(r * Nt):]
    train_data = data[training_indices]
    val_data = data[validation_indices]
    if conditions is not None:
        train_conditions = conditions[training_indices]
        val_conditions = conditions[validation_indices]
    else:
        train_conditions = None
        val_conditions = None
    del data, conditions
    
    # Get rngs for initializations
    rng, params_rng, dropout_rngs = jax.random.split(rng, 3)
    init_rngs = {"params": params_rng, "dropout": dropout_rngs}
    
    # Initializations
    cm = ConsistencyModel(norm_std, min_noise, net)
    if train_conditions is not None:
        online_params = net.init(
            init_rngs,
            jnp.ones((1, *train_data.shape[1:])),
            jnp.ones((1,)),
            jnp.ones((1,*train_conditions.shape[1:])),
        )
    else:
        online_params = net.init(
            init_rngs,
            jnp.ones((1, *train_data.shape[1:])),
            jnp.ones((1,)),
            None,
        )
    
    target_params = online_params
    writer.log_params(online_params)
    
    # Create the optimizer with EMA
    opt = optax.chain(
        optimizer(learning_rate),
        optax.ema(ema_rate, debias=True),
    )
    state = TrainState.create(
        apply_fn=net.apply, 
        params=online_params, 
        tx=opt
    )
    
    # Losses and its index
    loss_idx, k = 0, 0
    losses = np.zeros(log_each, dtype=np.float32)
    
    # Training loop
    for epoch in range(epochs):
        for start_idx in range(0, Nt, batch_size):
            # Iteration index (based on last batch)
            end_idx = start_idx + batch_size
            k = epoch * Nt + end_idx
            idx = indices[start_idx:end_idx]
            
            x = train_data[idx, :, :, :]
            if train_conditions is not None:
                c = train_conditions[idx, :, :, :]
            else:
                c = None
            
            # Sampling
            rng, i_rng, z_rng, dropout_rng = jax.random.split(rng, 4)
            i = jax.random.randint(i_rng, (batch_size,), 1, N(k))
            z = jax.random.normal(z_rng, (batch_size, Ny, Nx, 1))
            
            # Get times (noises) and noised images
            online_noise = noise_schedule(i + 1, N(k))
            x_online = x + batch_mul(online_noise, z)
            target_noise = noise_schedule(i, N(k))
            x_target = x + batch_mul(target_noise, z)
            
            # Loss weight
            loss_weight = loss_weight_fn(target_noise, online_noise)
            
            # Train step with JIT
            loss, state, target_params = train_step(
                cm,
                loss_weight,
                mu(k),
                dropout_rng,
                state,
                target_params,
                x_online, online_noise, i + 1,
                x_target, target_noise, i,
                c,
            )
            losses[loss_idx % log_each] = loss
            loss_idx += 1
            
            
            # TODO: build a proper validation pipeline
            if k % log_each != 0:
                continue
            
            # Train loss
            avg_loss = np.mean(losses)
            losses = np.zeros(log_each, dtype=np.float32)
            writer.log_loss(k, avg_loss)
            writer.add_csv_row(
                k,
                train_loss=avg_loss,
                N=N(k),
                mu=mu(k),
            )
        
        # Log for epoch
        if epoch % ckpt_each == 0:
            ckpter.save(k, state.params, state.opt_state)

    writer.close()
    ckpter.close()
