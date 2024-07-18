import h5py
import numpy as np
import jax.numpy as jnp
from sklearn.model_selection import train_test_split

def batch_iterator(data, batch_size):
    """Create an iterator that yields batches of data."""
    data_len = len(data)
    indices = np.arange(data_len)
    np.random.shuffle(indices)
    
    for start_idx in range(0, data_len, batch_size):
        end_idx = min(start_idx + batch_size, data_len)
        batch_indices = indices[start_idx:end_idx]
        yield {"x": data[batch_indices]}

def get_custom_dataset(file_path: str, key: str, batch_size: int, split_ratio: float, seed: int = 42):
    # Load the dataset from the .h5 file
    with h5py.File(file_path, 'r') as f:
        data = f[key][:]
    
    # Find the maximum value for normalization
    max_value = np.max(data)
    
    # Normalize the dataset
    data = data / max_value
    
    # Adapt shape for the model
    num_blocks = 2
    _, Ny, Nx = data.shape
    Ny -= Ny % 2**(num_blocks)
    Nx -= Nx % 2**(num_blocks)
    data = data[:, :Ny, :Nx]
    data = jnp.expand_dims(data, axis=-1)
    
    # Split the dataset into training and validation sets
    train_data, val_data = train_test_split(data, test_size=split_ratio, random_state=seed)
    
    # Convert numpy arrays to JAX arrays
    train_data = jnp.array(train_data, dtype=jnp.float32)
    print("train data:", train_data.shape)
    val_data = jnp.array(val_data, dtype=jnp.float32)
    print("val data:", val_data.shape)
    
    # Create iterators for batching
    train_iterator = batch_iterator(train_data, batch_size)
    val_iterator = batch_iterator(val_data, batch_size)
    
    return train_iterator, val_iterator, Ny, Nx

train_iterator, val_iterator, Ny, Nx = get_custom_dataset(
    '/work/FAC/FGSE/IDYST/tbeucler/downscaling/mlima/data/train_data/cpc_june-july-dry-filter.h5',
    'precip',
    batch_size=2,
    split_ratio=0.2,
)