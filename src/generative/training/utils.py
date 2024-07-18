import jax
import jax.numpy as jnp

def adjust_dimensions(data, num_blocks=4):
    """
    Adjust the dimensions of the input data to be compatible with the given number of blocks.

    Args:
        data (numpy array): Input data with shape (Nt, Ny, Nx, ...)
        num_blocks (int, optional): Number of blocks. Defaults to 4.

    Returns:
        numpy array: Adjusted data with shape (Nt, Ny', Nx', ...)
    """
    _, Ny, Nx, _ = data.shape
    Ny -= Ny % 2**num_blocks
    Nx -= Nx % 2**num_blocks
    return data[:, :Ny, :Nx, :]


def initialize_model(model, init_rng, train_data, train_conditions):
    """
    Initializes the model with the given data and conditions.

    Args:
    - model: The model to be initialized.
    - init_rng: The random number generator for initialization.
    - train_data: The training data.
    - train_conditions: The conditions corresponding to the training data. If None, it is ignored.

    Returns:
    - variables: The initialized model variables.
    """
    if train_conditions is not None:
        variables = model.init(
            init_rng,
            jnp.ones((1, *train_data.shape[1:])),
            jnp.ones((1,)),
            jnp.ones((1,*train_conditions.shape[1:])),
        )
    else:
        variables = model.init(
            init_rng,
            jnp.ones((1, *train_data.shape[1:])),
            jnp.ones((1,)),
            None,
        )
    return variables


def split_data(data, conditions, validation_ratio, batch_size, rng):
    """
    Splits the data into training and validation sets.

    Args:
    - data: The input data to be split.
    - conditions: The conditions corresponding to the data. If None, it is ignored.
    - validation_ratio: The ratio of data to be used for validation.
    - batch_size: The batch size for training.
    - rng: The random number generator.

    Returns:
    - train_data: The training data.
    - val_data: The validation data.
    - train_conditions: The conditions corresponding to the training data. If conditions is None, it is None.
    - val_conditions: The conditions corresponding to the validation data. If conditions is None, it is None.
    """
    Nt = len(data)
    indices = jnp.arange(Nt)
    rng, idx_rng = jax.random.split(rng, 2)
    indices = jax.random.permutation(idx_rng, indices, independent=True)
    r = 1 - validation_ratio
    training_indices = indices[:int(r*Nt)]
    validation_indices = indices[int(r*Nt):]
    Nt = int(r*Nt)
    Nt -= Nt % batch_size
    train_data = data[training_indices]
    val_data = data[validation_indices]
    if conditions is not None:
        train_conditions = conditions[training_indices]
        val_conditions = conditions[validation_indices]
    else:
        train_conditions = None
        val_conditions = None
    
    del data, conditions
    
    return train_data, val_data, train_conditions, val_conditions