import jax
import jax.numpy as jnp
import tensorflow as tf
import jax.numpy as jnp
from tqdm import tqdm

from swirl_dynamics.data.hdf5_utils import read_single_array
from swirl_dynamics.lib import diffusion as dfn_lib
from swirl_dynamics.lib import solvers as solver_lib
from swirl_dynamics.projects import probabilistic_diffusion as dfn
from data.surface_data import SurfaceData, ForecastEnsembleSurfaceData

# TODO: create functionality to apply log during training (and take it out in test)

def get_dataset(file_path: str, key: str, batch_size: int, apply_log: bool=False):
    # Read the dataset from the .hdf5 file.
    images = read_single_array(file_path, key)

    # Normalize the images
    mu = jnp.mean(images)
    sigma = jnp.std(images)
    images = (images - mu) / sigma

    # Expand dims
    images = jnp.expand_dims(images, axis=-1)

    # Create a TensorFlow dataset from the images.
    ds = tf.data.Dataset.from_tensor_slices({"x": images})

    # Repeat, batch, and prefetch the dataset.
    ds = ds.repeat()
    ds = ds.batch(batch_size)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    ds = ds.as_numpy_iterator()

    return ds


def generate(
    config,
    train_file_path: str,
    prior_sfc_data: ForecastEnsembleSurfaceData,
    clip_max: int,
    num_samples: int,
    num_chunks: int=1,
):
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
    
    # read training surface data
    train_sfc_data = SurfaceData.load_from_h5(train_file_path, ["precip"])
    
    # get train dataset info
    train_shape = train_sfc_data.get_shape()
    train_mean = train_sfc_data.get_means()[0]
    train_std = train_sfc_data.get_stds()[0]
    
    # delete train dataset as it may be too large
    del train_sfc_data

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
        return_full_paths=False, # Set to `True` if the full sampling paths are needed
    )
    
    # JIT sampler and sample
    generate = jax.jit(sampler.generate, static_argnames=('num_samples',))
    
    # Get test dataset
    prior_sfc_data.normalize() # use train mean and std to normalize instead???
    test_ds = prior_sfc_data.precip
    test_ds = jnp.expand_dims(test_ds, axis=-1) # channels
    
    # Calculate the new shape for the samples array
    num_lead_times, num_ensembles, num_times, num_lats, num_lons, num_channels = test_ds.shape
    new_shape = (
        num_lead_times,
        num_ensembles*num_samples,
        num_times,
        num_lats,
        num_lons,
        num_channels,
    )

    # Preallocate the array for the samples
    samples_array = jnp.zeros(new_shape)

    # Forecast samples with tqdm tracker
    rng = jax.random.PRNGKey(0)

    # Samples generation loop
    total_iterations = num_lead_times * num_ensembles * num_times * num_chunks
    with tqdm(total=total_iterations, desc="Generating samples") as pbar:
        for lead_time_idx in range(num_lead_times):
            for ensemble_idx in range(num_ensembles):
                for time_idx in range(num_times):
                    for chunk_idx in range(num_chunks):
                        rng, rng_step = jax.random.split(rng)

                        # Generate samples
                        samples = generate(
                            init_sample=test_ds[lead_time_idx, ensemble_idx, time_idx],
                            rng=rng_step,
                            num_samples=num_samples//num_chunks,
                        ) * train_std + train_mean

                        # Save the samples into the preallocated array
                        start_idx = chunk_idx * (num_samples // num_chunks)
                        end_idx = start_idx + (num_samples // num_chunks)
                        samples_array = samples_array.at[
                            lead_time_idx, start_idx:end_idx, time_idx,
                        ].set(samples)

                        pbar.update(1)
    
    # Clip zeros
    samples = jnp.clip(samples_array[...,0], min=0, max=None)
    
    return ForecastEnsembleSurfaceData(
        lead_time=prior_sfc_data.lead_time,
        number=range(num_samples*prior_sfc_data.number.size),
        time=prior_sfc_data.time,
        latitude=prior_sfc_data.latitude,
        longitude=prior_sfc_data.longitude,
        precip=samples,
    )
