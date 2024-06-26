import jax.numpy as jnp

from utils import *

def test_utils():
    # Create some test data
    data = jnp.array([1.0, 2.0, 3.0, 4.0])

    # Test process_data without log transforming
    data, mean, std = process_data(data, mean=None, std=None, is_log_transforming=False)
    assert jnp.allclose(mean, 2.5)
    assert jnp.allclose(std, 1.118033988749895)
    data = deprocess_data(data, mean, std, is_log_transforming=False)
    assert jnp.allclose(data, jnp.array([1.0, 2.0, 3.0, 4.0]))
    
    # Test process_data with log transforming
    data, mean, std = process_data(data, mean=None, std=None, is_log_transforming=True)
    data = deprocess_data(data, mean, std, is_log_transforming=True)
    assert jnp.allclose(data, jnp.array([1.0, 2.0, 3.0, 4.0]))
    
    print("utils.py passed!")
