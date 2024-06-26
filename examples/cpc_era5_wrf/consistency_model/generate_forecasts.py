import os, h5py
import numpy as np
import jax
import jax.numpy as jnp

from models.consistency_model import ConsistencyModel
from training.experiment import Experiment
from training.checkpointing import Checkpointer
from utils import *

def main():
    # Experiment folder and name
    experiment_name = "debug"
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
    imax = 182 # experiment.imax
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
    latest_step = 30000 # ckpter.manager.latest_step()
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
        era5_lons = f['longitude'][:Nx]
        era5_lats = f['latitude'][:Ny]
        era5_times = f['time'][:]
        
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
            paired_i = jnp.full((batch_size,), 35)
            paired_noise = noise_schedule(paired_i, imax)
            unpaired_i = jnp.full((batch_size,), 120)
            unpaired_noise = noise_schedule(unpaired_i, imax)
            rng, z_rng = jax.random.split(rng, 2)
            z = jax.random.normal(z_rng, (end_idx-start_idx, Ny, Nx, 1))
            x_paired = x + batch_mul(paired_noise, z)
            x_unpaired = x + batch_mul(unpaired_noise, z)
            
            # Output time
            paired_output = cm.apply(params, x_paired, None, paired_noise, paired_i)
            unpaired_output = cm.apply(params, x_unpaired, None, unpaired_noise, unpaired_i)
            
            # Restore normalizations
            paired_imgs = deprocess_data(
                    paired_output,
                    dataset_mean,
                    dataset_std,
                    norm_mean,
                    norm_std,
                    log_transform,
            ).__array__()[:, :, :, 0]
            print("paired out max:", np.max(paired_output))
            unpaired_imgs = deprocess_data(
                    unpaired_output,
                    dataset_mean,
                    dataset_std,
                    norm_mean,
                    norm_std,
                    log_transform,
            ).__array__()[:, :, :, 0]
            print("unpaired out max:", np.max(unpaired_output))
            
            # Collect results for this time step
            paired_outputs[start_idx:end_idx,:,:] = paired_imgs
            unpaired_outputs[start_idx:end_idx,:,:] = unpaired_imgs
        
        # Save datasets
        write_dataset(era5_times, era5_lats, era5_lons, paired_outputs, os.path.join(test_data_dir, f"paired_{key}.h5"))
        write_dataset(era5_times, era5_lats, era5_lons, unpaired_outputs, os.path.join(test_data_dir, f"unpaired_{key}.h5"))

    print("Done!")

if __name__ == "__main__":
    main()
