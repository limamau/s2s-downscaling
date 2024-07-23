from clu import metric_writers
import h5py
import jax
import numpy as np
import jax.numpy as jnp
import optax
import orbax.checkpoint as ocp
import os
import tensorflow as tf
from sklearn.model_selection import train_test_split


from swirl_dynamics import templates
from swirl_dynamics.lib import diffusion as dfn_lib
from swirl_dynamics.projects import probabilistic_diffusion as dfn

# ## Dataset
# def batch_iterator(data, batch_size):
#     """Create an iterator that yields batches of data."""
#     data_len = len(data)
#     indices = np.arange(data_len)
#     np.random.shuffle(indices)
    
#     for start_idx in range(0, data_len, batch_size):
#         end_idx = min(start_idx + batch_size, data_len)
#         batch_indices = indices[start_idx:end_idx]
#         yield {"x": data[batch_indices]}

# def get_custom_dataset(file_path: str, key: str, batch_size: int, split_ratio: float, seed: int = 42):
#     # Load the dataset from the .h5 file
#     with h5py.File(file_path, 'r') as f:
#         data = f[key][:]
    
#     # Find the maximum value for normalization
#     max_value = np.max(data)
    
#     # Normalize the dataset
#     data = data / max_value
    
#     # Adapt shape for the model
#     num_blocks = 2
#     _, Ny, Nx = data.shape
#     Ny -= Ny % 2**(num_blocks)
#     Nx -= Nx % 2**(num_blocks)
#     data = data[:, :Ny, :Nx]
#     data = jnp.expand_dims(data, axis=-1)
    
#     # Split the dataset into training and validation sets
#     train_data, val_data = train_test_split(data, test_size=split_ratio, random_state=seed)
    
#     # Convert numpy arrays to JAX arrays
#     train_data = jnp.array(train_data, dtype=jnp.float32)
#     print("train data:", train_data.shape)
#     val_data = jnp.array(val_data, dtype=jnp.float32)
#     print("val data:", val_data.shape)
    
#     # Create iterators for batching
#     train_iterator = batch_iterator(train_data, batch_size)
#     val_iterator = batch_iterator(val_data, batch_size)
    
#     return train_iterator, val_iterator, Ny, Nx

def get_custom_dataset(split: str, batch_size: int, num_blocks: int = 2):
    # Load the dataset from the .h5 file
    file_path = '/work/FAC/FGSE/IDYST/tbeucler/downscaling/mlima/data/train_data/cpc_june-july-dry-filter.h5'
    key = 'precip'
    with h5py.File(file_path, 'r') as f:
        data = f[key][:]
    
    # Find the maximum value for normalization
    max_value = np.max(data)
    
    # Normalize the dataset
    data = data / max_value
    
    # Adapt shape for the model
    _, Ny, Nx = data.shape
    Ny -= Ny % 2**(num_blocks)
    Nx -= Nx % 2**(num_blocks)
    data = data[:, :Ny, :Nx]
    data = jnp.expand_dims(data, axis=-1)
    
    # Convert to TensorFlow dataset
    ds = tf.data.Dataset.from_tensor_slices(data)
    
    # Apply split
    total_size = ds.cardinality().numpy()
    if split == "train[:75%]":
        ds = ds.take(int(0.75 * total_size))
    elif split == "train[75%:]":
        ds = ds.skip(int(0.75 * total_size))
    
    # Prepare dataset
    ds = ds.map(lambda x: {"x": tf.cast(x, tf.float32)})
    ds = ds.repeat()
    ds = ds.batch(batch_size)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    ds = ds.as_numpy_iterator()
    
    return ds

# train_iterator, val_iterator, Ny, Nx = get_custom_dataset(
#     '/work/FAC/FGSE/IDYST/tbeucler/downscaling/mlima/data/train_data/cpc_june-july-dry-filter.h5',
#     'precip',
#     batch_size=2,
#     split_ratio=0.2,
# )

# The standard deviation of the normalized dataset.
# This is useful for determining the diffusion scheme and preconditioning
# of the neural network parametrization.
DATA_STD = 0.31

## Architecture
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

## Training
diffusion_scheme = dfn_lib.Diffusion.create_variance_exploding(
    sigma=dfn_lib.tangent_noise_schedule(),
    data_std=DATA_STD,
)


## Architecture
file_path = '/work/FAC/FGSE/IDYST/tbeucler/downscaling/mlima/data/train_data/cpc_june-july-dry-filter.h5'
key = 'precip'
with h5py.File(file_path, 'r') as f:
    data = f[key][0,:,:]

# Adapt shape for the model
Ny, Nx = data.shape
num_blocks = 2
Ny -= Ny % 2**(num_blocks)
Nx -= Nx % 2**(num_blocks)
model = dfn.DenoisingModel(
    # `input_shape` must agree with the expected sample shape (without the batch
    # dimension), which in this case is simply the dimensions of a single MNIST
    # sample.
    input_shape=(Ny, Nx, 1),
    denoiser=denoiser_model,
    noise_sampling=dfn_lib.log_uniform_sampling(
        diffusion_scheme, clip_min=1e-4, uniform_grid=True,
    ),
    noise_weighting=dfn_lib.edm_weighting(data_std=DATA_STD),
)

script_dir = os.path.dirname(os.path.realpath(__file__))
## Learning parameters
num_train_steps = 100_000  #@param
workdir = os.path.join(script_dir, "/tmp/diffusion_demo_mnist")  #@param
print("workdir:", workdir)
train_batch_size = 2  #@param
eval_batch_size = 2  #@param
initial_lr = 0.0  #@param
peak_lr = 1e-4  #@param
warmup_steps = 1000  #@param
end_lr = 1e-6  #@param
ema_decay = 0.999  #@param
ckpt_interval = 1000  #@param
max_ckpt_to_keep = 5  #@param


## Setup trainer
# NOTE: use `trainers.DistributedDenoisingTrainer` for multi-device
# training with data parallelism.
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

## Train
templates.run_train(
    train_dataloader= get_custom_dataset(
        split="train[:75%]",
        batch_size=train_batch_size,
    ),
    trainer=trainer,
    workdir=workdir,
    total_train_steps=num_train_steps,
    metric_writer=metric_writers.create_default_writer(
        workdir, asynchronous=False,
    ),
    metric_aggregation_steps=100,
    eval_dataloader=get_custom_dataset(
        split="train[75%:]",
        batch_size=eval_batch_size,
    ),
    eval_every_steps=1000,
    num_batches_per_eval=5,
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
