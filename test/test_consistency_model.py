import jax.numpy as jnp

from models.generative_models.consistency_model import ConsistencyModel

def test_consistency_model():
    std_data = 0.5
    min_noise = 0.002
    
    class F:
        @staticmethod
        def apply(params, x, i, c, is_training, rngs):
            return jnp.zeros_like(x)
    cm = ConsistencyModel(std_data, min_noise, F)

    # Create some dummy input data
    params = jnp.zeros((1,))
    x = jnp.ones((1,))
    c = None
    i = jnp.ones((1,))

    # Call the apply method with min_noise as the input
    output = output = cm.apply(
        params, 
        x, c, jnp.array([min_noise]), i, 
        is_training=True, rngs=None,
    )

    # Check if the output is the same as the input
    assert jnp.allclose(output, x)
    
    print("consistency_model.py passed!")
