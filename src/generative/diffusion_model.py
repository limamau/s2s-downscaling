import flax.linen as nn
import jax
import jax.numpy as jnp
from utils import batch_mul


class DiffusionModel(nn.Module):
    """Diffusion Model"""
    F: nn.Module
    sigma_min: float = 0.002
    sigma_max: float = 80.0
    sigma_data: float = 0.5
    rho: float = 7
    P_mean: float = -1.2
    P_std: float = 1.2

    def setup(self):
        # Initialize any necessary submodules or parameters here
        pass

    def __call__(self, x, sigma, c, is_training=False):
        """Dθ(x; σ) = cskip(σ)x + cout(σ)Fθ(cin(σ)x; cnoise(σ))"""
        # Scaling factors
        c_skip = self.skip_scaling(sigma)
        c_out = self.output_scaling(sigma)
        c_in = self.input_scaling(sigma)
        c_noise = self.noise_conditioning(sigma)

        # Scale the input
        scaled_x = batch_mul(c_in, x)

        # Call the network Fθ
        network_output = self.F(scaled_x, c_noise, c, is_training=is_training)

        # Combine the skip connection and the network output
        denoised = batch_mul(c_skip, x) + batch_mul(c_out, network_output)

        return denoised
    
    
    # ***********************
    # Network preconditioning
    # ***********************
    def sigma_of_t(self, t):
        """Compute σ(t) for the EDM schedule."""
        return t
    
    def t_of_simga(self, sigma):
        """Compute σ⁻¹(σ) for the EDM schedule."""
        return sigma

    def skip_scaling(self, sigma):
        """Skip scaling c_skip(σ) = σ_data^2 / (σ^2 + σ_data^2)"""
        return self.sigma_data**2 / (sigma**2 + self.sigma_data**2)

    def output_scaling(self, sigma):
        """Output scaling c_out(σ) = σ · σ_data / sqrt(σ^2 + σ_data^2)"""
        return sigma * self.sigma_data / jnp.sqrt(sigma**2 + self.sigma_data**2)

    def input_scaling(self, sigma):
        """Input scaling c_in(σ) = 1 / sqrt(σ^2 + σ_data^2)"""
        return 1 / jnp.sqrt(sigma**2 + self.sigma_data**2)

    def noise_conditioning(self, sigma):
        """Noise conditioning c_noise(σ) = 1/4 * ln(σ)"""
        return 0.25 * jnp.log(sigma)
    
    
    # ********
    # Training
    # ********
    def sample_noise(self, rng, batch_size):
        """Noise sampling based on ln(σ) ~ N(P_mean, P_std^2)"""
        return jnp.exp(self.P_mean + self.P_std * jax.random.normal(rng, (batch_size,)))
    
    def loss_weighting(self, sigma):
        """Loss weighting λ(σ) = (σ^2 + σ_data^2) / (σ · σ_data)^2"""
        return (sigma**2 + self.sigma_data**2) / (sigma * self.sigma_data)**2
    
    
    # ********
    # Sampling
    # ********
    def sigma_schedule(self, i, N):
        """Compute sigma(t) for the EDM schedule during sampling."""
        if i < N:
            return (self.sigma_max**(1/self.rho) + i/(N-1)*(self.sigma_min**(1/self.rho) - self.sigma_max**(1/self.rho)))**self.rho
        else:
            return 0
    
    def sample(self, params, rng, batch, batch_size=1, steps=50, sigma_star=None):
        """
        Sampling function using Euler-Maruyama method.

        Args:
        params: model parameters
        rng: PRNG key
        batch: batch
        batch_size: number of samples to generate
        steps: number of sampling steps

        Returns:
        x: generated samples
        """
        # Generate initial noise
        x, c = batch
        rng, step_rng = jax.random.split(rng)
        x = x + self.sigma_max*jax.random.normal(step_rng, x.shape)

        # Time steps based on the EDM formula
        t = jnp.linspace(0, 1, steps + 1)
        dt = t[1] - t[0]

        # Euler-Maruyama sampling loop
        for i in range(steps):
            rng, step_rng = jax.random.split(rng, 2)
            batched_i = jnp.full((batch_size,), i)
            t_i = t[batched_i]
            sigma_i = self.sigma_schedule(batched_i, steps)
            # Check dimensions
            denoise = batch_mul(1/t_i, x) - batch_mul(2/t_i, self.apply({'params': params}, x, sigma_i, None)) * dt
            noise_injection = batch_mul(jnp.sqrt(2 * t_i), jax.random.normal(step_rng, x.shape)) * jnp.sqrt(dt)
            x = denoise + noise_injection

        return x
