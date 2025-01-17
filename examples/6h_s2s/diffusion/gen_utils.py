import jax.numpy as jnp
import tensorflow as tf
import jax.numpy as jnp

from swirl_dynamics.data.hdf5_utils import read_single_array


def get_dataset(
    file_path: str,
    key: str,
    batch_size: int,
    apply_log: bool=False,
    epsilon: float=1e-6,
):
    # Read the dataset from the .hdf5 file.
    images = read_single_array(file_path, key)
    
    # Apply log
    if apply_log:
        images = jnp.log(images + epsilon)

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


def normalize(images, mu=None, sigma=None, apply_log=False, epsilon=1e-6):
    if mu is None:
        mu = jnp.mean(images)
        
    if sigma is None:
        sigma = jnp.std(images)
        
    images = (images - mu) / sigma

    if apply_log:
        images = jnp.log(images + epsilon)

    return images


def denormalize(images, mu, sigma, apply_log=False, epsilon=1e-6):
    # Denormalize the images
    images = images * sigma + mu

    # Apply exp if log was applied
    if apply_log:
        images = jnp.exp(images) - epsilon

    return images
