import jax, time
import jax.numpy as jnp
import xarray as xr

from swirl_dynamics.lib import diffusion as dfn_lib
from swirl_dynamics.lib import solvers as solver_lib
from swirl_dynamics.projects import probabilistic_diffusion as dfn
from swirl_dynamics.data.hdf5_utils import read_single_array
from utils import write_dataset

import configs

def get_dataset_info(file_path: str, key: str, split:float):
    # Read the dataset from the .hdf5 file.
    images = read_single_array(file_path, key)

    # Determine the split indices.
    num_images = images.shape[0]
    if split > 0:
        end_idx = int(num_images * split)
        images = images[:end_idx]
    elif split < 0:
        start_idx = int(num_images * (1 + split))
        images = images[start_idx:]

    # Normalize the images
    mu = jnp.mean(images)
    sigma = jnp.std(images)
    images = (images - mu) / sigma

    return images.shape, mu, sigma


def get_test_dataset_info(file_path: str, key: str):
    # Read the dataset from the .hdf5 file.
    images = read_single_array(file_path, key)
    lons = read_single_array(file_path, "longitude")
    lats = read_single_array(file_path, "latitude")
    times = xr.open_dataset(file_path).time.values

    # Normalize the images
    mu = jnp.mean(images)
    sigma = jnp.std(images)
    images = (images - mu) / sigma

    # Expand dims
    ds = jnp.expand_dims(images, axis=-1)

    return ds, lons, lats, times


def main(config, test_file_path, save_path, clip_max, num_samples):
    # Get denoiser model back
    denoiser_model = dfn_lib.PreconditionedDenoiserUNet(
        out_channels=1,
        resize_to_shape=(224, 336),
        num_channels=config.num_channels,
        downsample_ratio=config.downsample_ratio,
        num_blocks=config.num_blocks,
        noise_embed_dim=128,
        padding="SAME",
        use_attention=True,
        use_position_encoding=True,
        num_heads=8,
        sigma_data=config.data_std,
    )
    
    # Restore train state from checkpoint. By default, the move recently saved
    # checkpoint is restored. Alternatively, one can directly use
    # `trainer.train_state` if continuing from the training section above.
    trained_state = dfn.DenoisingModelTrainState.restore_from_orbax_ckpt(
        f"{config.workdir}/checkpoints", step=None
    )
    # Construct the inference function
    denoise_fn = dfn.DenoisingTrainer.inference_fn_from_state_dict(
        trained_state, use_ema=True, denoiser=denoiser_model
    )
    
    # Schemes
    diffusion_scheme = dfn_lib.Diffusion.create_variance_exploding(
        sigma=dfn_lib.tangent_noise_schedule(),
        data_std=config.data_std,
    )
    new_diffusion_scheme = dfn_lib.Diffusion.create_variance_exploding(
        sigma=dfn_lib.tangent_noise_schedule(clip_max=clip_max),
        data_std=config.data_std,
    )
    
    # Get train dataset info
    train_shape, train_mean, train_std = get_dataset_info(
        file_path=config.file_path,
        key="precip",
        split=0.75,
    )

    # Sampler
    sampler = dfn_lib.SdeCustomSampler(
        input_shape=train_shape[1:],
        integrator=solver_lib.EulerMaruyama(),
        tspan=dfn_lib.edm_noise_decay(
            new_diffusion_scheme, rho=7, num_steps=256, end_sigma=1e-3,
        ),
        scheme=new_diffusion_scheme,
        denoise_fn=denoise_fn,
        guidance_transforms=(),
        apply_denoise_at_end=True,
        return_full_paths=False,  # Set to `True` if the full sampling paths are needed
    )
    
    # JIT sampler and sample
    generate = jax.jit(sampler.generate, static_argnames=('num_samples',))
    
    # Test dataset
    test_ds, lons, lats, times = get_test_dataset_info(
        file_path=test_file_path,
        key="precip",
    )
    
    # Forecast samples
    samples_list = []

    # Iterate over the test dataset and generate samples
    rng = jax.random.PRNGKey(0)
    for i in range(test_ds.shape[0]):
        rng, rng_step = jax.random.split(rng)
        # Generate samples from the test dataset
        samples = generate(
            init_sample=test_ds[i],
            rng=rng_step,
            num_samples=num_samples,
        ) * train_std + train_mean
        
        # Append samples to the list
        samples_list.append(samples)
    
    # Convert list to a single numpy array
    forecasts = jnp.array(samples_list)[:,:,:,:,0]
    
    # Clip zeros
    forecasts = jnp.clip(forecasts, min=0, max=None)

    # Save all samples in a single HDF5 file
    write_dataset(
        times, lats, lons, forecasts, save_path,
    )
    

if __name__ == "__main__":
    # Config
    config = configs.light.get_config()
    
    # Additional file paths
    test_file_path = "/work/FAC/FGSE/IDYST/tbeucler/downscaling/mlima/data/test_data/cpc.h5"
    
    # Directory to store the training checkpoints
    workdir = f"/work/FAC/FGSE/IDYST/tbeucler/downscaling/mlima/s2s-downscaling/examples/cpc_era5_wrf/diffusion/{config.experiment_name}"
    
    # Noise up which to noise-denoise
    clip_max = 50
    
    # Number of samples
    num_samples = 4
    
    # Path to save the generated forecasts
    save_path = f"/work/FAC/FGSE/IDYST/tbeucler/downscaling/mlima/data/generated_forecasts/{config.experiment_name}_{clip_max}.h5"
    
    start_time = time.time()
    main(config, test_file_path, save_path, clip_max, num_samples)
    final_time = time.time() - start_time
    print(f"Total time taken: {final_time} seconds.")
