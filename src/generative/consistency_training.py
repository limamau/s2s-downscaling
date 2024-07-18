import jax
import optax

import numpy as np
import jax.numpy as jnp
from flax.training.train_state import TrainState
from functools import partial

from utils import batch_mul, create_folder, process_data, get_spatial_lengths
from .training.utils import *
from .training.writing import Writer
from .training.checkpointing import Checkpointer
from generative.consistency_model import ConsistencyModel
from evaluation.evaluate import evaluate_model


@partial(jax.jit, static_argnums=(0,))
def train_step(
    loss_weight,
    mu,
    dropout_rng,
    state,
    x_online, online_sigma,
    x_target, target_sigma,
    c,
):
    # Define the loss function inside the train step
    def loss_fn(online_params, target_params):
        C = 0.3
        x = state.apply(
            {'params': online_params}, 
            x_online, online_sigma, c,
            is_training=True, 
            rngs={'dropout': dropout_rng},
        )
        y = state.apply(
            {'params': target_params},
            x_target, target_sigma, c,
            is_training=True, 
            rngs={'dropout': dropout_rng},
        )
        return jnp.mean(
            batch_mul(
                loss_weight, 
                jnp.sqrt(jnp.mean((x-y)**2, axis=(1,2,3,)) + C**2) - C,
            )
        )
    
    # Get loss and gradients (with respect to online_params)
    loss, grads = jax.value_and_grad(loss_fn)(state.params, state.target)
    
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
    writer = Writer(
        experiment_dir, 
        csv_args=[
            "train_loss",
            "N",
            "mu",
            "psd_distance",
            "cdf_distance",
            "pss",
        ]
    )
    writer.log_and_save_config(experiment)
    writer.log_device()
    ckpter = Checkpointer(experiment_dir)
    del experiment
    
    # Adjust dimensions
    data = adjust_dimensions(data)
    Nt, Ny, Nx, _ = data.shape
    
    # Normalization
    data, dataset_mean, dataset_std = process_data(data, None, None, norm_mean, norm_std, is_log_transforming)
    writer.save_normalizations(data.shape, dataset_mean, dataset_std)
    
    # PRNG
    rng = jax.random.PRNGKey(888)
    
    # Divide data into training and validation
    train_data, val_data, train_conditions, val_conditions = split_data(
        data, conditions, validation_ratio, batch_size, rng)
    
    
    # Initializations
    rng, init_rng = jax.random.split(rng, 2)
    model = ConsistencyModel(net)
    variables = initialize_model(model, init_rng, train_data, train_conditions)
    writer.log_params(variables['params'])
    online_params = variables['params']
    target_params = online_params.copy()
    
    # Create the optimizer with EMA
    opt = optax.chain(
        optimizer(learning_rate),
        optax.ema(ema_rate, debias=True),
    )
    state = TrainState.create(
        apply_fn=net.apply, 
        params=online_params,
        target=target_params,
        tx=opt,
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
            
            current_N = N(k)
            current_mu = mu(k)
            
            x = train_data[start_idx:end_idx, :, :, :]
            if train_conditions is not None:
                c = train_conditions[start_idx:end_idx, :, :]
            else:
                c = None
            
            # Sampling
            rng, i_rng, z_rng, dropout_rng = jax.random.split(rng, 4)
            i = jax.random.randint(i_rng, (batch_size,), 1, current_N)
            z = jax.random.normal(z_rng, (batch_size, Ny, Nx, 1))
            
            # Get noises and noised images
            online_sigma = noise_schedule(i + 1, current_N)
            x_online = x + batch_mul(online_sigma, z)
            target_sigma = noise_schedule(i, current_N)
            x_target = x + batch_mul(target_sigma, z)
            
            # Loss weight
            loss_weight = loss_weight_fn(target_sigma, online_sigma)
            
            # Train step with JIT
            loss, state, target_params = train_step(
                loss_weight,
                mu,
                dropout_rng,
                state,
                x_online, online_sigma,
                x_target, target_sigma,
                c,
            )
            losses[loss_idx % log_each] = loss
            loss_idx += 1
            
            if k % log_each != 0:
                continue
            
            # Train loss
            avg_loss = np.mean(losses)
            losses = np.zeros(log_each, dtype=np.float32)
            writer.log_loss(k, avg_loss)
            
            # Validation score
            psd_distance, cdf_distance, pss = evaluate_model(
                model,
                noise_schedule,
                target_params,
                val_data,
                val_conditions,
                batch_size,
                current_N,
                current_N,
            )
            
            # Write to CSV
            writer.add_csv_row(
                k,
                train_loss=avg_loss,
                N=current_N,
                mu=current_mu,
                psd_distance=psd_distance,
                cdf_distance=cdf_distance,
                pss=pss,
            )
        
        # Log for epoch
        if epoch % ckpt_each == 0:
            ckpter.save(k, target_params, state.opt_state)

    writer.close()
    ckpter.close()
