from clu import metric_writers
import jax
import jax.numpy as jnp
import optax
import orbax.checkpoint as ocp
import tensorflow as tf

from swirl_dynamics import templates
from swirl_dynamics.lib import diffusion as dfn_lib
from swirl_dynamics.projects import probabilistic_diffusion as dfn
from swirl_dynamics.data.hdf5_utils import read_single_array

import configs


def get_dataset(file_path: str, key: str, split: float, batch_size: int):
    # Read the dataset from the .hdf5 file.
    images = read_single_array(file_path, key)

    # Determine the split indices.
    num_images = images.shape[0]
    if split > 0:
        end_idx = int(num_images * split)
        images = images[:end_idx]
    elif split < 0:
        start_idx = int(num_images * (1 + split))
        images = images[start_idx:]

    # Normalize the images
    mu = jnp.mean(images)
    sigma = jnp.std(images)
    images = (images - mu) / sigma

    # Expand dims
    images = jnp.expand_dims(images, axis=-1)

    # Create a TensorFlow dataset from the images.
    ds = tf.data.Dataset.from_tensor_slices({"x": images})

    # Repeat, batch, and prefetch the dataset.
    ds = ds.repeat()
    ds = ds.batch(batch_size)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    ds = ds.as_numpy_iterator()

    return ds


def main(config):
    # ************
    # Architecture 
    # ************
    denoiser_model = dfn_lib.PreconditionedDenoiserUNet(
        out_channels=1,
        resize_to_shape=(224, 336),
        num_channels=config.num_channels,
        downsample_ratio=config.downsample_ratio,
        num_blocks=config.num_blocks,
        noise_embed_dim=128,
        padding="SAME",
        use_attention=True,
        use_position_encoding=True,
        num_heads=8,
        sigma_data=config.data_std,
    )
    
    # **************
    # Training setup
    # **************
    diffusion_scheme = dfn_lib.Diffusion.create_variance_exploding(
        sigma=dfn_lib.tangent_noise_schedule(),
        data_std=config.data_std,
    )
    
    dummy_ds = get_dataset(
        file_path=config.file_path,
        key=config.key,
        split=0.001, # asssuming the dataset has a size a little over 1000 in number of samples
        batch_size=1,
    ) #TODO: create a function to get the shape of the dataset

    model = dfn.DenoisingModel(
        # `input_shape` must agree with the expected sample shape (without the batch
        # dimension), which in this case is simply the dimensions of a single MNIST
        # sample.
        input_shape=next(iter(dummy_ds))["x"].shape[1:],
        denoiser=denoiser_model,
        noise_sampling=dfn_lib.log_uniform_sampling(
            diffusion_scheme, clip_min=1e-4, uniform_grid=True,
        ),
        noise_weighting=dfn_lib.edm_weighting(data_std=config.data_std),
    )
    
    # *****
    # Train
    # *****
    trainer = dfn.DenoisingTrainer(
        model=model,
        rng=jax.random.PRNGKey(888),
        optimizer=optax.adam(
            learning_rate=optax.warmup_cosine_decay_schedule(
                init_value=config.initial_lr,
                peak_value=config.peak_lr,
                warmup_steps=config.warmup_steps,
                decay_steps=config.num_train_steps,
                end_value=config.end_lr,
            ),
        ),
        # We keep track of an exponential moving average of the model parameters
        # over training steps. This alleviates the "color-shift" problems known to
        # exist in the diffusion models.
        ema_decay=config.ema_decay,
    )
    
    templates.run_train(
        train_dataloader=get_dataset(
            file_path=config.file_path,
            key=config.key,
            split=0.75,
            batch_size=config.train_batch_size,
        ),
        trainer=trainer,
        workdir=config.workdir,
        total_train_steps=config.num_train_steps,
        metric_writer=metric_writers.create_default_writer(
            config.workdir, asynchronous=False
        ),
        metric_aggregation_steps=100,
        eval_dataloader=get_dataset(
            file_path=config.file_path,
            key=config.key,
            split=-0.25,
            batch_size=config.eval_batch_size,
        ),
        eval_every_steps = 1000,
        num_batches_per_eval = 2,
        callbacks=(
            # This callback displays the training progress in a tqdm bar
            templates.TqdmProgressBar(
                total_train_steps=config.num_train_steps,
                train_monitors=("train_loss",),
            ),
            # This callback saves model checkpoint periodically
            templates.TrainStateCheckpoint(
                base_dir=config.workdir,
                options=ocp.CheckpointManagerOptions(
                    save_interval_steps=config.ckpt_interval, max_to_keep=config.max_ckpt_to_keep
                ),
            ),
        ),
    )
    

if __name__ == "__main__":
    config = configs.heavy.get_config()
    main(config)
