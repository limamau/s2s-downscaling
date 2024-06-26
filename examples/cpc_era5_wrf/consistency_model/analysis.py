import os, h5py
import pandas as pd
import xarray as xr

import numpy as np
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

from models.consistency_model import ConsistencyModel
from training.experiment import Experiment
from training.checkpointing import Checkpointer
from evaluation.plots import plot_maps
from utils import *


def main():
    # Experiment folder and name
    experiment_name = "diff_nolambda"
    script_dir = os.path.dirname(os.path.realpath(__file__))
    experiment_dir = os.path.join(script_dir, "experiments", experiment_name)
    figs_dir = os.path.join(script_dir, "figs")
    create_folder(figs_dir)
    
    # Get experiment back using the .yml file...
    experiment = Experiment(
        experiment_file=os.path.join(experiment_dir, "experiment.yml"),
        dataset_file="/work/FAC/FGSE/IDYST/tbeucler/downscaling/mlima/data/test_data/cpc.h5",
    )
    net = experiment.network
    imin = experiment.imin
    imax = experiment.imax
    min_noise = experiment.min_noise
    max_noise = experiment.max_noise
    noise_schedule = experiment.noise_schedule
    _, Ny, Nx = experiment.dimensions
    log_transform = experiment.is_log_transforming
    norm_mean = experiment.norm_mean
    norm_std = experiment.norm_std
    cm = ConsistencyModel(norm_std, min_noise, net)
    
    # ... and the checkpoints
    ckpter = Checkpointer(experiment_dir)
    latest_step = 180000 # ckpter.manager.latest_step()
    print("Latest step:", latest_step)
    restored = ckpter.manager.restore(latest_step)
    params = restored['params']
    
    # Load data to test
    data_dir = "/work/FAC/FGSE/IDYST/tbeucler/downscaling/mlima/data/train_data"
    time = 5
    cpc_file = os.path.join(data_dir, "cpc.h5")
    with h5py.File(cpc_file, 'r') as f:
        cpc_data = f['precip'][[time],:,:]
    era5_file = os.path.join(data_dir, "era5_qm_all.h5")
    with h5py.File(era5_file, 'r') as f:
        era5_data = f['precip'][[time],:,:]
        era5_lons = f['longitude'][:]
        era5_lats = f['latitude'][:]
    
    # Check times
    cpc_times = xr.open_dataset(cpc_file).time.values
    era5_times = xr.open_dataset(era5_file).time.values
    print("ERA5 time:", era5_times[time])
    print("CPC time:", cpc_times[time])
    
    # Adjust input data
    era5_data = era5_data[[0],:Ny,:Nx]
    input_data = jnp.expand_dims(era5_data, axis=(3))
    dataset_mean = experiment.dataset_mean
    dataset_std = experiment.dataset_std
    input_data, _, _ = process_data(
        input_data,
        dataset_mean, dataset_std,
        norm_mean, norm_std,
        log_transform,
    )
    
    # Add noises
    x = input_data[[0],:Ny,:Nx,:]
    paired_i = jnp.array([50])
    imax_at_the_step = 182 # FIXME: hard coded
    paired_noise = noise_schedule(paired_i, imax_at_the_step)
    print("Paired noise:", paired_noise)
    unpaired_i = jnp.array([182])
    unpaired_noise = noise_schedule(unpaired_i, imax_at_the_step)
    print("Unpaired noise:", unpaired_noise)
    z = jax.random.normal(jax.random.PRNGKey(0), (1, Ny, Nx, 1))
    x_paired = x + batch_mul(paired_noise, z)
    x_unpaired = x + batch_mul(unpaired_noise, z)
    
    # Output time
    paired_output = cm.apply(params, x_paired, None, paired_noise, paired_i)
    unpaired_output = cm.apply(params, x_unpaired, None, unpaired_noise, unpaired_i)
    
    # Restore normalizations
    paired_img = deprocess_data(
        paired_output,
        dataset_mean,
        dataset_std,
        norm_mean,
        norm_std,
        log_transform,
    ).__array__()
    unpaired_img = deprocess_data(
        unpaired_output,
        dataset_mean,
        dataset_std,
        norm_mean,
        norm_std,
        log_transform,
    ).__array__()
    
    extent = (
        np.min(era5_lons[:Nx]),
        np.max(era5_lons[:Nx]),
        np.min(era5_lats[:Ny]),
        np.max(era5_lats[:Ny]),
    )
    
    # Output plots
    arrays = (
        era5_data[0,:Ny,:Nx],
        paired_img[0,:,:,0],
        unpaired_img[0,:,:,0],
        cpc_data[0,:Ny,:Nx],
    )
    titles = ("input", "paired", "unpaired", "ground truth")
    extents = (extent, extent, extent, extent)
    fig, _ = plot_maps(arrays, titles, extents, vmax=5)
    fig.savefig(os.path.join(figs_dir, "output_maps.png"))
    
    print("c_skip:", cm._c_skip(paired_noise))
    print("c_out:", cm._c_out(paired_noise))
    
    # Training plots
    arrays = (
        x_paired[0,:,:,0], paired_output[0,:,:,0],
        x_unpaired[0,:,:,0], unpaired_output[0,:,:,0],
    )
    titles = ("paired input", "paired output", "unpaired input", "unpaired output")
    extents = (extent, extent, extent, extent)
    fig, _ = plot_maps(
        arrays, titles, extents,
        axis_labels=((None, None), (None, None), (None, None), (None, None),),
        vmin=norm_mean - 2 * norm_std,
        vmax=norm_mean + 2 * norm_std,
        cbar_label=' ',
    )
    fig.savefig(os.path.join(figs_dir, "training_maps.png"))

    print("Done!")


if __name__ == "__main__":
    main()
