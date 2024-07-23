import os, h5py
import xarray as xr
import numpy as np
import jax
import jax.numpy as jnp

from generative.diffusion_model import DiffusionModel
from generative.training.experiment import Experiment
from generative.training.checkpointing import Checkpointer
from evaluation.evaluate import evaluate
from evaluation.plots import plot_maps
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
        dataset_file="/work/FAC/FGSE/IDYST/tbeucler/downscaling/mlima/data/train_data/cpc_june-july-dry-filter.h5",
    )
    net = experiment.network
    _, Ny, Nx = experiment.dimensions
    log_transform = experiment.is_log_transforming
    norm_mean = experiment.norm_mean
    norm_std = experiment.norm_std
    model = DiffusionModel(net)
    
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
    era5_data = cpc_data[[0],:Ny,:Nx]
    input_data = jnp.expand_dims(era5_data, axis=(3))
    dataset_mean = experiment.dataset_mean
    dataset_std = experiment.dataset_std
    input_data, _, _ = process_data(
        input_data,
        dataset_mean, dataset_std,
        norm_mean, norm_std,
        log_transform,
    )
    
    # Generate samples
    rng = jax.random.PRNGKey(888)
    _, sim = evaluate(
        model, params, input_data, None, rng, batch_size=1, steps=5, t_star=1,
    )
    print("max sim value:", np.max(sim))
    print("min sim value:", np.min(sim))
    
    # Restore normalizations
    output_img = deprocess_data(
        sim,
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
        output_img[0,:,:,0],
        cpc_data[0,:Ny,:Nx],
    )
    titles = (
        "output",
        "ground truth",
    )
    extents = (extent, extent, extent, extent)
    fig, _ = plot_maps(arrays, titles, extents)
    fig.savefig(os.path.join(figs_dir, "output_maps.png"))

    print("Done!")


if __name__ == "__main__":
    main()
