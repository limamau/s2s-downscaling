import os
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


def plot_trainig_curve(figs_dir, df):
    plt.figure(figsize=(10,5))
    plt.plot(df['iteration'], df['loss'], color='red')
    plt.xlabel("Training iterations")
    plt.ylabel("Loss")
    plt.savefig(os.path.join(figs_dir, "training_losses.png"))


def main():
    # Experiment folder and name
    experiment_name = "diffusers_drop04"
    script_dir = os.path.dirname(os.path.realpath(__file__))
    experiment_dir = os.path.join(script_dir, "experiments", experiment_name)
    
    # Training losses
    training_df = pd.read_csv(os.path.join(experiment_dir, "losses.csv"))
    figs_dir = os.path.join(script_dir, "figs/training")
    create_folder(figs_dir)
    plot_trainig_curve(figs_dir, training_df)
    
    # Get experiment back using the .yml file...
    experiment = Experiment(os.path.join(experiment_dir, "experiment.yml"))
    net = experiment.network
    tmin = experiment.tmin
    Nt, Ny, Nx = experiment.dimensions
    sigma_data = experiment.sigma_data
    sigma_star = 2 # experiment.sigma_star
    cm = ConsistencyModel(sigma_data, tmin, net)
    
    # ... and the checkpoints
    ckpter = Checkpointer(experiment_dir)
    latest_step = ckpter.manager.latest_step()
    restored = ckpter.manager.restore(latest_step)
    params = restored['params']
    
    # Load data to test
    data_dir = "/work/FAC/FGSE/IDYST/tbeucler/downscaling/mlima/data/train_data"
    cpc_file = os.path.join(data_dir, "cpc.h5")
    cpc_ds = xr.open_dataset(cpc_file, engine='h5netcdf')
    cpc_data = cpc_ds['precip'].values
    era5_file = os.path.join(data_dir, "era5.h5")
    era5_ds = xr.open_dataset(era5_file, engine='h5netcdf')
    era5_data = era5_ds['precip'].values
    
    # Adjust input data
    time = 70
    era5_data = era5_data[[time],:Ny,:Nx]
    input_data = jnp.expand_dims(era5_data, axis=(3))
    input_data = log_transform(input_data)
    input_data, input_data_mean, input_data_std = normalize_data(input_data, sigma_data)
    
    # Add noise
    x = input_data[[0],:Ny,:Nx,:]
    t_star = jnp.array([sigma_star])
    z = jax.random.normal(jax.random.PRNGKey(37), (1, Ny, Nx, 1))
    x_noised = x + batch_mul(t_star, z)
    
    # Output time
    output_data = cm.apply(params, x_noised, t_star)
    
    # Restore normalizations
    input_img = unlog_transform(
        unnormalize_data(
            x_noised,
            input_data_mean,
            input_data_std,
            sigma_data,
        )
    ).__array__()
    output_img = unlog_transform(
        unnormalize_data(
            output_data,
            input_data_mean,
            input_data_std,
            sigma_data,
        )
    ).__array__()
    
    # Original ERA5 dataset
    raw_data_dir = "/work/FAC/FGSE/IDYST/tbeucler/downscaling/raw_data"
    raw_era5_file = os.path.join(raw_data_dir, "era5/precip/era5_tp_2024_01.nc")
    raw_era5_ds = xr.open_dataset(raw_era5_file)
    
    # Extents
    raw_era5_lons = raw_era5_ds.longitude.values
    raw_era5_lats = raw_era5_ds.latitude.values
    raw_era5_extent = (
        np.min(raw_era5_lons[:Nx]),
        np.max(raw_era5_lons[:Nx]),
        np.min(raw_era5_lats[:Ny]),
        np.max(raw_era5_lats[:Ny]),
    )
    era5_lons = era5_ds.longitude.values
    era5_lats = era5_ds.latitude.values
    extent = (
        np.min(era5_lons[:Nx]),
        np.max(era5_lons[:Nx]),
        np.min(era5_lats[:Ny]),
        np.max(era5_lats[:Ny]),
    )
    
    # Check plot
    arrays = (
        raw_era5_ds.tp.values[time,Ny::-1,:Nx]*1000,
        input_img[0,:,:,0],
        output_img[0,:,:,0],
        cpc_data[time,:Ny,:Nx],
    )
    titles = ("ERA5", "input", "CM", "CPC")
    extents = (raw_era5_extent, extent, extent, extent)
    fig, _ = plot_maps(arrays, titles, extents, vmax=15)
    fig.savefig(os.path.join(figs_dir, "training_maps.png"))
    
    print("Done!")
    
    
if __name__ == "__main__":
    main()



