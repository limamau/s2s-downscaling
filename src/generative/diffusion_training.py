import jax
import optax

import numpy as np
import jax.numpy as jnp
from flax.training.train_state import TrainState

from utils import batch_mul, create_folder, process_data, get_spatial_lengths
from .training.utils import *
from .training.writing import Writer
from .training.checkpointing import Checkpointer
from generative.diffusion_model import DiffusionModel
from evaluation.evaluate import evaluate


# @jax.jit
def train_step(
    c_skip,
    c_out,
    lambda_sigma,
    dropout_key,
    state,
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
    
    loss, grads = jax.value_and_grad(loss_fn)(state.params)
    
    state = state.apply_gradients(grads=grads)
    
    return loss, state


def train(experiment):
    # Unpack parameters from experiment
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
    Nt = data.shape[0]
    
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
    model = DiffusionModel(net)
    variables = initialize_model(model, init_rng, train_data, train_conditions)
    writer.log_params(variables['params'])
    
    # Create the optimizer with EMA
    opt = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adam(1e-6),
    )
    
    # Train state
    state = TrainState.create(
        apply_fn=model.apply,
        params=variables['params'],
        tx=opt,
    )
    
    # Losses and its index
    loss_idx, k = 0, 0
    losses = np.zeros(log_each, dtype=np.float32)
    
    # Training loop
    k = 0
    for epoch in range(epochs):
        for start_idx in range(0, Nt, batch_size):
            # Iteration data
            end_idx = start_idx + batch_size
            k +=1
            y = train_data[start_idx:end_idx, :, :, :]
            if train_conditions is not None:
                c = train_conditions[start_idx:end_idx, :, :]
            else:
                c = None
                
            # Sample noise level and noise
            rng, sigma_rng, normal_rng, dropout_rng = jax.random.split(rng, 4)
            sigma = model.sample_noise(sigma_rng, batch_size)
            n = batch_mul(sigma, jax.random.normal(normal_rng, y.shape))
            
            # and preconditionnings
            c_skip = model.skip_scaling(sigma)
            c_out = model.output_scaling(sigma)
            lambda_sigma = model.loss_weighting(sigma)
            
            # Train step with JIT
            loss, state = train_step(
                c_skip,
                c_out,
                lambda_sigma,
                dropout_rng,
                state,
                y,
                n,
                sigma,
            )
            print("loss: ", loss)
            losses[loss_idx % log_each] = loss
            loss_idx += 1
            
            if k % log_each != 0:
                continue
            
            # Train loss
            avg_loss = np.mean(losses)
            losses = np.zeros(log_each, dtype=np.float32)
            writer.log_loss(k, avg_loss)
            
            # Validation score
            psd_distance, cdf_distance, pss = evaluate(
                model, state.params, val_data, val_conditions, rng, batch_size=1, steps=5, t_star=1
            )
            
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
