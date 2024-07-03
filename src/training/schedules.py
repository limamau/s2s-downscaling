import jax.numpy as jnp

def N_schedule(s0, s1, K):
    """
    Schedule N(k) following https://arxiv.org/abs/2303.01469.

    Args:
        s0 (int): initial discretization steps.
        s1 (int): target discretization steps.
        K (int): training iterations.

    Returns:
        function: a function that takes an iteration k and returns the corresponding N(k).
    """
    return lambda k: jnp.ceil(jnp.sqrt(k/K * ((s1+1)**2 - s0**2) + s0**2) - 1) + 1


def N_schedule_improved(s0, s1, K):
    """
    Improved schedule N(k) following https://arxiv.org/abs/2310.14189.

    Args:
        s0 (int): initial discretization steps.
        s1 (int): target discretization steps.
        K (int): training iterations.

    Returns:
        function: a function that takes an iteration k and returns the corresponding N(k).
    """
    return lambda k: jnp.min(s0*2**(k/jnp.floor(K / (jnp.log2(s1/s0)+1))), s1) + 1


def mu_schedule(s0, mu0, N):
    """
    Schedule mu(N(k)).

    Args:
        s0 (int): initial discretization steps.
        mu0 (float): EMA decay rate at the beginning of model training.
        N (function): a function that takes k and returns the corresponding N(k).

    Returns:
        function: a function that takes k and returns the corresponding mu(N(k)).
    """
    return lambda k: jnp.exp(s0*jnp.log(mu0) / N(k))


def noise_schedule(min_noise, max_noise, rho=7):
    """
    Returns a schedule noise(n, N).

    Args:
        min_noise (float): minimum noise.
        max_noise (float): maximum noise.
        rho (int, optional): default is 7.

    Returns:
        function: a function that takes n and N and returns the corresponding noise(n, N).
    """
    return lambda n, N: (min_noise**(1/rho) + (n-1)/(N-1)*(max_noise**(1/rho) - min_noise**(1/rho)))**rho


def loss_weight(target_noise, online_noise):
    """
    Returns the loss weight.
    """
    return 1 / (online_noise - target_noise)