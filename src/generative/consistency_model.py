import jax
import jax.numpy as jnp
from flax import linen as nn
from utils import batch_mul


class ConsistencyModel(nn.Module):
    """Consistency Model"""
    F: nn.Module
    sigma_data: float = 0.5
    sigma_min: float = 0.002
    sigma_max: float = 80.0
    rho: float = 7

    def setup(self):
        # Initialize any necessary submodules or parameters here
        pass

    def __call__(self, x, sigma, c, is_training=False):
        """fθ(x; σ) = cskip(σ)x + cout(σ)Fθ(x; σ)"""
        # Scaling factors
        c_skip = self.skip_scaling(sigma)
        c_out = self.output_scaling(sigma)
        network_output = self.F.apply(x, sigma, c, is_training=is_training)
        denoised = batch_mul(c_skip, x) + batch_mul(c_out, network_output)
        return denoised
    
    
    # ***********************
    # Network preconditioning
    # ***********************
    def skip_scaling(self, sigma):
        """Skip scaling c_skip(σ) = σ_data^2 / ((σ - σ_min)^2 + σ_data^2)"""
        return self.sigma_data**2 / ((sigma - self.sigma_min)**2 + self.sigma_data**2)

    def output_scaling(self, sigma):
        """Output scaling c_out(σ) = σ_data * σ / sqrt(σ_data^2 + σ^2)"""
        return self.sigma_data * sigma / jnp.sqrt(self.sigma_data**2 + sigma**2)


    # ********
    # Training
    # ********
    def N_schedule(self, k, s0, s1, K):
        """
        Schedule N(k) following https://arxiv.org/abs/2303.01469.

        Args:
            k (int): current iteration.
            s0 (int): initial discretization steps.
            s1 (int): target discretization steps.
            K (int): training iterations.

        Returns:
            int: the corresponding N(k).
        """
        return jnp.ceil(jnp.sqrt(k / K * ((s1 + 1)**2 - s0**2) + s0**2) - 1) + 1

    def N_schedule_improved(self, k, s0, s1, K):
        """
        Improved schedule N(k) following https://arxiv.org/abs/2310.14189.

        Args:
            k (int): current iteration.
            s0 (int): initial discretization steps.
            s1 (int): target discretization steps.
            K (int): training iterations.

        Returns:
            int: the corresponding N(k).
        """
        return jnp.min(s0 * 2**(k / jnp.floor(K / (jnp.log2(s1 / s0) + 1))), s1) + 1

    def mu_schedule(self, k, s0, mu0, N):
        """
        Schedule mu(N(k)).

        Args:
            k (int): current iteration.
            s0 (int): initial discretization steps.
            mu0 (float): EMA decay rate at the beginning of model training.
            N (int): corresponding N(k).

        Returns:
            float: the corresponding mu(N(k)).
        """
        return jnp.exp(s0 * jnp.log(mu0) / N)

    def sigma_schedule(self, n, N):
        """
        Returns a schedule σ(n, N).

        Args:
            n (int): current iteration step.
            N (int): total discretization steps.

        Returns:
            float: the corresponding noise(n, N).
        """
        return (self.sigma_min**(1 / self.rho) + (n - 1) / (N - 1) * (self.sigma_max**(1 / self.rho) - self.sigma_min**(1 / self.rho)))**self.rho


    # ********
    # Training
    # ********
    @staticmethod
    def loss_weight(target_noise, online_noise):
        """
        Returns the loss weight.

        Args:
            target_noise (float): target noise level.
            online_noise (float): online noise level.

        Returns:
            float: the corresponding loss weight.
        """
        return 1 / (online_noise - target_noise)
    
    # ********
    # Sampling
    # ********
    def sample(self, params, rng, batch, batch_size, steps=1, sigma_star=None, sigma_decay=2):
        """
        Sampling function of the consistency moodel.

        Args:
        params: model parameters.
        rng: PRNG key.
        batch: batch.
        batch_size: number of samples to generate.
        steps (optional): number of sampling steps. Standard is 1.

        Returns:
        x: generated samples
        """
        x, c = batch
        if sigma_star is None:
            sigma = self.sigma_max
        else:
            sigma = sigma_star
        for _ in range(steps):
            rng, step_rng = jax.random.split(rng)
            x = x + self.sigma_max*jax.random.normal(step_rng, x.shape)
            x = self.apply({'params': params}, x, c, sigma)
            sigma /= sigma_decay

        return x
