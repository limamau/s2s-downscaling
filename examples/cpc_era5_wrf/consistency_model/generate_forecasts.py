import os, h5py
import numpy as np
import jax
import jax.numpy as jnp
import xarray as xr
from time import time

from models.generative_models.consistency_model import ConsistencyModel
from training.experiment import Experiment
from training.checkpointing import Checkpointer
from evaluation.evaluate import multi_step_sampling
from utils import *

def main():
    start_time = time()
    # Experiment folder and name
    experiment_name = "diffusers_jj"
    script_dir = os.path.dirname(os.path.realpath(__file__))
    experiment_dir = os.path.join(script_dir, "experiments", experiment_name)
    figs_dir = os.path.join(script_dir, "figs")
    create_folder(figs_dir)
    
    # Get experiment back using the .yml file...
    experiment = Experiment(
        experiment_file=os.path.join(experiment_dir, "experiment.yml"),
        #TODO: take the dataset out of the experiment
        dataset_file= "/work/FAC/FGSE/IDYST/tbeucler/downscaling/mlima/data/test_data/cpc.h5",
    )
    net = experiment.network
    imin = experiment.imin
    imax = 1280 
    noise_schedule = experiment.noise_schedule
    Nt, Ny, Nx = experiment.dimensions
    log_transform = experiment.is_log_transforming
    norm_mean = experiment.norm_mean
    norm_std = experiment.norm_std
    dataset_mean = experiment.dataset_mean
    dataset_std = experiment.dataset_std
    batch_size = experiment.batch_size
    cm = ConsistencyModel(norm_std, imin, net)
    
    # ... and the checkpoints
    ckpter = Checkpointer(experiment_dir)
    latest_step = ckpter.manager.latest_step()
    print("Latest step:", latest_step)
    restored = ckpter.manager.restore(latest_step)
    params = restored['params']
    
    
    # Load data to test
    test_data_dir = "/work/FAC/FGSE/IDYST/tbeucler/downscaling/mlima/data/test_data"
    
    era5_files = {
        'qm_all': os.path.join(test_data_dir, "era5_qm_all.h5"),
        'qm_point': os.path.join(test_data_dir, "era5_qm_point.h5")
    }
    
    with h5py.File(era5_files['qm_all'], "r") as f:
        lons = f['longitude'][:Nx]
        lats = f['latitude'][:Ny]
    times = xr.open_dataset(era5_files['qm_all'], engine='h5netcdf')['time'].values[:48]
    Nt = len(times)
    print("Nt:", Nt)
        
    # PRNG to the Gaussian noise
    rng = jax.random.PRNGKey(37)

    for key, era5_file in era5_files.items():
        with h5py.File(era5_file, 'r') as f:
            era5_data = f['precip'][:,:Ny,:Nx]
        
        # Initialize arrays for concatenated results
        paired_outputs = np.zeros((Nt, Ny, Nx))
        unpaired_outputs = np.zeros((Nt, Ny, Nx))
        
        for start_idx in range(0, Nt, batch_size):
            print("Processing:", start_idx)
            end_idx = start_idx + batch_size
            
            # Adjust input data
            era5_slice = era5_data[start_idx:end_idx,:,:]
            input_data = jnp.expand_dims(era5_slice, axis=(3))
            x, _, _ = process_data(
                input_data,
                dataset_mean, dataset_std,
                norm_mean, norm_std,
                log_transform,
            )
            
    
            # Add noises
            paired_i = 830
            paired_noise = noise_schedule(paired_i, imax)
            print("Paired noise:", paired_noise)
            unpaired_i = 830
            unpaired_noise = noise_schedule(unpaired_i, imax)
            print("Unpaired noise:", unpaired_noise)
        
            # Output time
            num_steps = 4
            rng, rng_paired, rng_unpaired = jax.random.split(rng, 3)
            paired_output = multi_step_sampling(
                cm,
                noise_schedule,
                params,
                rng_paired,
                x,
                None,
                1,
                imax,
                paired_i,
                1,
                num_steps=2,
            )
            unpaired_output = multi_step_sampling(
                cm,
                noise_schedule,
                params,
                rng_unpaired,
                x,
                None,
                1,
                imax,
                unpaired_i,
                1,
                num_steps=4,
            )
            
            # Restore normalizations
            paired_imgs = deprocess_data(
                    paired_output,
                    dataset_mean,
                    dataset_std,
                    norm_mean,
                    norm_std,
                    log_transform,
                    clip_zero=True,
            ).__array__()[:, :, :, 0]
            print("paired out max:", np.max(paired_output))
            unpaired_imgs = deprocess_data(
                    unpaired_output,
                    dataset_mean,
                    dataset_std,
                    norm_mean,
                    norm_std,
                    log_transform,
                    clip_zero=True,
            ).__array__()[:, :, :, 0]
            print("unpaired out max:", np.max(unpaired_output))
            
            # Collect results for this time step
            paired_outputs[start_idx:end_idx,:,:] = paired_imgs
            unpaired_outputs[start_idx:end_idx,:,:] = unpaired_imgs
        
        # Save datasets
        write_dataset(times, lats, lons, paired_outputs, os.path.join(test_data_dir, f"cm_{key}_{2}.h5"))
        write_dataset(times, lats, lons, unpaired_outputs, os.path.join(test_data_dir, f"cm_{key}_{4}.h5"))

    print("Done!")
    print("Time elapsed:", time() - start_time)

if __name__ == "__main__":
    main()
