import jax.numpy as jnp

def N_schedule(s0, s1, K):
    """Returns a schedule N(k).

    Args:
        s0 (int): initial discretization steps.
        s1 (int): target discretization steps.
        K (int): training iterations.

    Returns:
        function: a function that takes an iteration k and returns the corresponding N(k).
    """
    return lambda k: jnp.ceil(jnp.sqrt(k/K * ((s1+1)**2 - s0**2) + s0**2) - 1) + 1


def mu_schedule(s0, mu0):
    """Returns a schedule mu(N).

    Args:
        s0 (int): initial discretization steps.
        mu0 (float): EMA decay rate at the beginning of model training.

    Returns:
        function: a function that takes N and returns the corresponding mu(N).
    """
    return lambda N: jnp.exp(s0*jnp.log(mu0) / N)


def tn_schedule(tmin, tmax, rho=7):
    """Returns a schedule t(n, N).

    Args:
        tmin (float): minimum noise.
        tmax (float): maximum noise.
        rho (int, optional): default is 7.

    Returns:
        function: a function that takes n and N and returns the corresponding t(n, N).
    """
    return lambda n, N: (tmin**(1/rho) + (n-1)/(N-1)*(tmax**(1/rho) - tmin**(1/rho)))**rho