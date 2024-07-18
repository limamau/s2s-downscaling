import os, h5py
import xarray as xr

import numpy as np
import jax
import jax.numpy as jnp

from models.generative_models.consistency_model import ConsistencyModel
from training.experiment import Experiment
from training.checkpointing import Checkpointer
from evaluation.plots import plot_maps
from evaluation.evaluate import multi_step_sampling
from utils import *


def main():
    # TODO: calculate of the simulation I'm seeing
    # is that normal that the loss is going down but simulations seem
    # so bad?
    # download more common data and use more common validations
    # Experiment folder and name
    experiment_name = "min-noise=0.002_1e6_80M_sigma-data=2"
    script_dir = os.path.dirname(os.path.realpath(__file__))
    experiment_dir = os.path.join(script_dir, "experiments", experiment_name)
    figs_dir = os.path.join(script_dir, "figs")
    create_folder(figs_dir)
    
    # Get experiment back using the .yml file...
    experiment = Experiment(
        experiment_file=os.path.join(experiment_dir, "experiment.yml"),
        dataset_file="/work/FAC/FGSE/IDYST/tbeucler/downscaling/mlima/data/train_data/cpc_june-july-dry-filter.h5",
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
    cm = ConsistencyModel(norm_std, 1, net)
    
    # ... and the checkpoints
    ckpter = Checkpointer(experiment_dir)
    latest_step = ckpter.manager.latest_step()
    print("Latest step:", latest_step)
    restored = ckpter.manager.restore(latest_step)
    params = restored['params']
    
    # Load data to test
    data_dir = "/work/FAC/FGSE/IDYST/tbeucler/downscaling/mlima/data/test_data"
    time = 38
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
    imax_at_the_step = 1280
    paired_i = 500
    paired_noise = noise_schedule(paired_i, imax_at_the_step)
    print("Paired noise:", paired_noise)
    unpaired_i = 500
    unpaired_noise = noise_schedule(unpaired_i, imax_at_the_step)
    print("Unpaired noise:", unpaired_noise)
    
    # Output time
    paired_output = multi_step_sampling(
        cm,
        noise_schedule,
        params,
        jax.random.PRNGKey(37),
        x,
        None,
        1,
        imax_at_the_step,
        paired_i,
        1,
        num_steps=1,
    )
    unpaired_output = multi_step_sampling(
        cm,
        noise_schedule,
        params,
        jax.random.PRNGKey(37),
        x,
        None,
        1,
        imax_at_the_step,
        unpaired_i,
        1,
        num_steps=4,
    )
    
    # Restore normalizations
    paired_img = deprocess_data(
        paired_output,
        dataset_mean,
        dataset_std,
        norm_mean,
        norm_std,
        log_transform,
        clip_zero=True,
    ).__array__()
    unpaired_img = deprocess_data(
        unpaired_output,
        dataset_mean,
        dataset_std,
        norm_mean,
        norm_std,
        log_transform,
        clip_zero=True,
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
    titles = (
        "input",
        "num steps={}".format(1),
        "num steps={}".format(2),
        "ground truth"
    )
    extents = (extent, extent, extent, extent)
    fig, _ = plot_maps(arrays, titles, extents)
    fig.savefig(os.path.join(figs_dir, "output_maps.png"))
    
    print("paired c_skip:", cm._c_skip(paired_noise))
    print("paired c_out:", cm._c_out(paired_noise))
    
    # # Training plots
    # arrays = (
    #     x_paired[0,:,:,0], paired_output[0,:,:,0],
    #     x_unpaired[0,:,:,0], unpaired_output[0,:,:,0],
    # )
    # titles = ("paired input", "paired output", "unpaired input", "unpaired output")
    # extents = (extent, extent, extent, extent)
    # fig, _ = plot_maps(
    #     arrays, titles, extents,
    #     axis_labels=((None, None), (None, None), (None, None), (None, None),),
    #     cmap="viridis",
    #     norm=None,
    #     vmin=norm_mean - 2 * norm_std,
    #     vmax=norm_mean + 2 * norm_std,
    #     cbar_label=' ',
    # )
    # fig.savefig(os.path.join(figs_dir, "training_maps.png"))

    print("Done!")


if __name__ == "__main__":
    main()
