import os, functools

from clu import metric_writers
import jax
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import optax
import orbax.checkpoint as ocp
import tensorflow as tf
import tensorflow_datasets as tfds

from swirl_dynamics import templates
from swirl_dynamics.lib import diffusion as dfn_lib
from swirl_dynamics.lib import solvers as solver_lib
from swirl_dynamics.projects import probabilistic_diffusion as dfn


## Get back information from train.py
script_dir = os.path.dirname(os.path.abspath(__file__))
workdir = os.path.join("", "/tmp/diffusion_demo_mnist")
DATA_STD = 0.31
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
diffusion_scheme = dfn_lib.Diffusion.create_variance_exploding(
    sigma=dfn_lib.tangent_noise_schedule(),
    data_std=DATA_STD,
)

## Restore model
# Restore train state from checkpoint. By default, the move recently saved
# checkpoint is restored. Alternatively, one can directly use
# `trainer.train_state` if continuing from the training section above.
trained_state = dfn.DenoisingModelTrainState.restore_from_orbax_ckpt(
    f"{workdir}/checkpoints", step=None
)
# Construct the inference function
denoise_fn = dfn.DenoisingTrainer.inference_fn_from_state_dict(
    trained_state, use_ema=True, denoiser=denoiser_model
)

## Sampler
sampler = dfn_lib.SdeSampler(
    input_shape=(28, 28, 1),
    integrator=solver_lib.EulerMaruyama(),
    tspan=dfn_lib.edm_noise_decay(
        diffusion_scheme, rho=7, num_steps=256, end_sigma=1e-3,
    ),
    scheme=diffusion_scheme,
    denoise_fn=denoise_fn,
    guidance_transforms=(),
    apply_denoise_at_end=True,
    return_full_paths=False,  # Set to `True` if the full sampling paths are needed
)

## Generate samples with JIT
generate = jax.jit(sampler.generate, static_argnames=('num_samples',))
samples = generate(
    rng=jax.random.PRNGKey(8888), num_samples=4
)
# Plot generated samples
fig, ax = plt.subplots(1, 4, figsize=(8, 2))
for i in range(4):
  im = ax[i].imshow(samples[i, :, :, 0] * 255, cmap="gray", vmin=0, vmax=255)

plt.tight_layout()
figs_dir = os.path.join(workdir, "figures")
fig.savefig("samples.png")