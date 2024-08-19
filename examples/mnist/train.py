import os

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


def get_mnist_dataset(file_path: str, key: str, split:float, batch_size: int):
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

    # Normalize the images to [0, 1].
    images = images.astype(jnp.float32) / 255.0

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

def main(mnist_folder: str, workidir: str):
    # *******
    # Dataset
    # *******
    # The standard deviation of the normalized dataset.
    # This is useful for determining the diffusion scheme and preconditioning
    # of the neural network parametrization.
    DATA_STD = 0.31
    
    # ************
    # Architecture 
    # ************
    denoiser_model = dfn_lib.PreconditionedDenoiserUNet(
        out_channels=1,
        num_channels=(64, 128),
        downsample_ratio=(2, 2),
        num_blocks=4,
        noise_embed_dim=128,
        padding="SAME",
        use_attention=True,
        use_position_encoding=True,
        num_heads=8,
        sigma_data=DATA_STD,
    )
    
    # ********
    # Training
    # ********
    diffusion_scheme = dfn_lib.Diffusion.create_variance_exploding(
        sigma=dfn_lib.tangent_noise_schedule(),
        data_std=DATA_STD,
    )

    model = dfn.DenoisingModel(
        # `input_shape` must agree with the expected sample shape (without the batch
        # dimension), which in this case is simply the dimensions of a single MNIST
        # sample.
        input_shape=(28, 28, 1),
        denoiser=denoiser_model,
        noise_sampling=dfn_lib.log_uniform_sampling(
            diffusion_scheme, clip_min=1e-4, uniform_grid=True,
        ),
        noise_weighting=dfn_lib.edm_weighting(data_std=DATA_STD),
    )
    
    # **********
    # Parameters
    # **********
    num_train_steps = 100_000  #@param
    train_batch_size = 32  #@param
    eval_batch_size = 32  #@param
    initial_lr = 0.0  #@param
    peak_lr = 1e-4  #@param
    warmup_steps = 1000  #@param
    end_lr = 1e-6  #@param
    ema_decay = 0.999  #@param
    ckpt_interval = 1000  #@param
    max_ckpt_to_keep = 5  #@param
    
    # *****
    # Train
    # *****
    trainer = dfn.DenoisingTrainer(
        model=model,
        rng=jax.random.PRNGKey(888),
        optimizer=optax.adam(
            learning_rate=optax.warmup_cosine_decay_schedule(
                init_value=initial_lr,
                peak_value=peak_lr,
                warmup_steps=warmup_steps,
                decay_steps=num_train_steps,
                end_value=end_lr,
            ),
        ),
        # We keep track of an exponential moving average of the model parameters
        # over training steps. This alleviates the "color-shift" problems known to
        # exist in the diffusion models.
        ema_decay=ema_decay,
    )
    
    templates.run_train(
        train_dataloader=get_mnist_dataset(
            file_path=os.path.join(mnist_folder,'train.hdf5'),
            key="image",
            split=0.75,
            batch_size=train_batch_size,
        ),
        trainer=trainer,
        workdir=workdir,
        total_train_steps=num_train_steps,
        metric_writer=metric_writers.create_default_writer(
            workdir, asynchronous=False
        ),
        metric_aggregation_steps=100,
        eval_dataloader=get_mnist_dataset(
            file_path=os.path.join(mnist_folder,'train.hdf5'),
            key="image",
            split=-0.25,
            batch_size=eval_batch_size,
        ),
        eval_every_steps = 1000,
        num_batches_per_eval = 2,
        callbacks=(
            # This callback displays the training progress in a tqdm bar
            templates.TqdmProgressBar(
                total_train_steps=num_train_steps,
                train_monitors=("train_loss",),
            ),
            # This callback saves model checkpoint periodically
            templates.TrainStateCheckpoint(
                base_dir=workdir,
                options=ocp.CheckpointManagerOptions(
                    save_interval_steps=ckpt_interval, max_to_keep=max_ckpt_to_keep
                ),
            ),
        ),
    )
    

if __name__ == "__main__":
    # Folder where the downloaded dataset is stored
    mnist_folder = "/work/FAC/FGSE/IDYST/tbeucler/downscaling/mlima/data/mnist"
    
    # Directory to store the training checkpoints
    workdir = "/work/FAC/FGSE/IDYST/tbeucler/downscaling/mlima/s2s-downscaling/examples/mnist"
    
    main(mnist_folder, workdir)
