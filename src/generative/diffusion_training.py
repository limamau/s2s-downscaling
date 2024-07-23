import jax
import optax

import numpy as np
import jax.numpy as jnp
from flax.training.train_state import TrainState

from utils import batch_mul, create_folder, process_data
from generative.training.utils import *
from generative.training.writing import Writer
from generative.training.checkpointing import Checkpointer
from generative.diffusion_model import DiffusionModel
from evaluation.evaluate import evaluate, get_metrics

@jax.jit
def train_step(
    state,
    ema_decay,
    dropout_key,
    lambda_sigma,
    c_skip,
    c_out,
    y,
    n,
    sigma,
):
    def loss_fn(params):
        # Compute the network output
        network_output = state.apply_fn(
            {'params': params},
            y + n, sigma, None,
            is_training=True,
            rngs={'dropout': dropout_key}
        )
        
        # Compute the effective training target
        target = (y - c_skip * (y + n)) / c_out
        
        # Compute the l2_error
        l2_error = jnp.mean((network_output - target)**2, axis=(1, 2, 3))
        
        # Compute the weighted squared error
        weighted_error = batch_mul(lambda_sigma, c_out**2 * l2_error)
        
        # Compute the mean loss
        loss = jnp.mean(weighted_error)
        
        return loss
    
    # Compute loss and gradients
    loss, grads = jax.value_and_grad(loss_fn)(state.params)
    
    # Calculate EMA parameters
    updates, new_opt_state = state.tx.update(
        grads, state.opt_state, state.params
    )
    new_params = optax.apply_updates(state.params, updates)
    ema_params = jax.tree_map(
        lambda ema_p, new_p: ema_p*ema_decay + (1-ema_decay)*new_p,
        state.params, new_params
    )
    
    # Update state
    state.replace(params=ema_params, opt_state=new_opt_state)
    
    return loss, state


def train(experiment):
    # Unpack parameters from experiment
    experiment_dir = experiment.experiment_dir
    data = experiment.data
    conditions = experiment.conditions
    validation_ratio = experiment.validation_ratio
    net = experiment.network
    # optimizer = experiment.optimizer
    # learning_rate = experiment.learning_rate
    # ema_rate = experiment.ema
    
    batch_size = experiment.batch_size
    # min_noise = experiment.min_noise
    epochs = experiment.epochs
    norm_mean = experiment.norm_mean
    norm_std = experiment.norm_std
    is_log_transforming = experiment.is_log_transforming
    
    log_each = experiment.log_each
    ckpt_each = experiment.ckpt_each
    rng = jax.random.PRNGKey(888)
    
    # Logs and checkpoints
    create_folder(experiment_dir, overwrite=True)
    writer = Writer(
        experiment_dir, 
        csv_args=[
            "train_loss",
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
    
    # Normalization
    data, dataset_mean, dataset_std = process_data(data, None, None, norm_mean, norm_std, is_log_transforming)
    writer.save_normalizations(data.shape, dataset_mean, dataset_std)
    
    # Divide data into training and validation
    train_data, val_data, train_conditions, val_conditions = split_data(
        data, conditions, validation_ratio, batch_size, rng
    )
    Nt = train_data.shape[0]
    
    # Initializations
    rng, init_key = jax.random.split(rng, 2)
    model = DiffusionModel(net)
    variables = initialize_model(model, init_key, train_data, train_conditions)
    writer.log_params(variables['params'])
    
    # Create the optimizer and the ema
    tx = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adam(
            learning_rate=optax.linear_schedule(
                init_value=1e-1,
                end_value=1e-4,
                transition_steps=5000,
            ),
        ),
    )
    
    # Train state
    state = TrainState.create(
        apply_fn=model.apply,
        params=variables['params'],
        tx=tx,
    )
    ema_decay = 0.999
    
    # Losses and its index
    loss_idx, k = 0, 0
    losses = np.zeros(log_each, dtype=np.float32)
    
    # Training loop
    for epoch in range(epochs):
        for start_idx in range(0, Nt, batch_size):
            # Iteration data
            end_idx = start_idx + batch_size
            k += 1
            y = train_data[start_idx:end_idx, :, :, :]
            if train_conditions is not None:
                c = train_conditions[start_idx:end_idx, :, :]
            else:
                c = None
                
            # Sample noise level and noise
            rng, sigma_key, normal_key, dropout_key = jax.random.split(rng, 4)
            sigma = model.sample_noise(sigma_key, batch_size)
            n = batch_mul(sigma, jax.random.normal(normal_key, y.shape))
            
            # and preconditionnings
            c_skip = model.skip_scaling(sigma)
            c_out = model.output_scaling(sigma)
            lambda_sigma = model.loss_weighting(sigma)
            
            # Train step with JIT
            loss, state = train_step(
                state,
                ema_decay,
                dropout_key,
                lambda_sigma,
                c_skip,
                c_out,
                y,
                n,
                sigma,
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
            obs, sim = evaluate(
                model, state.params, val_data, val_conditions, rng, batch_size=1, steps=5, t_star=1,
            )
            psd_distance, cdf_distance, pss = get_metrics(obs, sim)
            
            # Write to CSV
            writer.add_csv_row(
                k,
                train_loss=avg_loss,
                psd_distance=psd_distance,
                cdf_distance=cdf_distance,
                pss=pss,
            )
        
        # Log for epoch
        if epoch % ckpt_each == 0:
            ckpter.save(k, state.params, state.opt_state)

    writer.close()
    ckpter.close()
