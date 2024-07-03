import os, h5py
import numpy as np
from tqdm import tqdm
import xarray as xr
from datetime import datetime

from engineering.spectrum import get_psd, apply_low_pass_filter
from evaluation.plots import plot_maps, plot_psds
from utils import create_folder, write_dataset, get_spatial_lengths

from engineering_utils import concat_era5, regrid_era5

    

def main():
    # Train set
    ##################
    
    # File directories
    raw_data_dir = "/work/FAC/FGSE/IDYST/tbeucler/downscaling/raw_data/era5/precip/"
    test_data_dir = "/work/FAC/FGSE/IDYST/tbeucler/downscaling/mlima/data/test_data"
        
    # Scale
    M_TO_MM = 1000
    
    # Read info from already preprocessed CPC data
    cpc_file = os.path.join(test_data_dir, "cpc.h5")
    with h5py.File(cpc_file, "r") as f:
        cpc_lons = f["longitude"][:]
        cpc_lats = f["latitude"][:]
    titles = ("ERA5", "Interpolated ERA5", "Low-pass filtered ERA5", "CPC")
    script_dir = os.path.dirname(os.path.abspath(__file__))
    figs_dir = os.path.join(script_dir, "figs")
    
    # Read file
    file_path = os.path.join(raw_data_dir, "era5_tp_2021_06_28-30.nc")
    ds = xr.open_dataset(file_path)
    
    times = ds.time.values
    raw_era5_lats = ds.latitude.values
    raw_era5_lons = ds.longitude.values
    raw_era5_data = ds.tp.values
    
    # Sort data
    if raw_era5_lats[0] > raw_era5_lats[-1]:
        raw_era5_lats = raw_era5_lats[::-1]
        raw_era5_data = raw_era5_data[:, ::-1, :]
    if raw_era5_lons[0] > raw_era5_lons[-1]:
        raw_era5_lons = raw_era5_lons[::-1]
        raw_era5_data = raw_era5_data[:, :, ::-1]
        
    # Scale
    raw_era5_data = raw_era5_data * M_TO_MM
    era5_extent = (raw_era5_lons[0], raw_era5_lons[-1], raw_era5_lats[0], raw_era5_lats[-1])
    cpc_extent = (cpc_lons[0], cpc_lons[-1], cpc_lats[0], cpc_lats[-1])
    extents = (era5_extent, cpc_extent, cpc_extent, cpc_extent)
    
    # Interpolate ERA5 to the same resolution as CombiPrecip
    interpolated_era5_data = regrid_era5(raw_era5_lats, raw_era5_lons, raw_era5_data, cpc_lats, cpc_lons, method='nearest')
    
    # Read info from already processed CPC data just to plot
    with h5py.File(os.path.join(test_data_dir, "cpc.h5"), "r") as f:
        t = -10 # just to plot again
        cpc_data_t = f["precip"][t,:,:]
    
    # Plot maps
    arrays = (
        interpolated_era5_data[t,:,:],
        cpc_data_t,
    )
    fig, _ = plot_maps(arrays, titles, extents)
    fig.savefig(os.path.join(figs_dir, "maps/era5_nearest.png"))
    
    # Create and save new dataset
    create_folder(test_data_dir)
    write_dataset(times, cpc_lats, cpc_lons, interpolated_era5_data, os.path.join(test_data_dir, "era5_nearest.h5"))
    
    print("Done!")
    
    
if __name__ == "__main__":
    main()
