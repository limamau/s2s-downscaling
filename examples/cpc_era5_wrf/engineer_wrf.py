import os
import numpy as np

from engineering.regridding import *
from evaluation.plots import plot_maps
from utils import *

def main():
    # Data paths
    wrf_data_dir = "/work/FAC/FGSE/IDYST/tbeucler/downscaling/mlima/data/output_wrf/2021062800_analysis"
    cpc_data_dir = "/work/FAC/FGSE/IDYST/tbeucler/downscaling/mlima/data/test_data"
    
    # Read and plot original datasets
    wrf_ds = concat_wrf(wrf_data_dir)
    time = 12
    print("Plot time:", wrf_ds.Time.values[time])
    
    # To plot
    script_dir = os.path.dirname(os.path.realpath(__file__))
    figs_dir = os.path.join(script_dir, "figs/engineering")
    os.makedirs(figs_dir, exist_ok=True)
    os.makedirs(os.path.join(figs_dir, "maps"), exist_ok=True)
    lons_2d = wrf_ds.XLONG.values
    lats_2d = wrf_ds.XLAT.values
    times = wrf_ds.Time.values
    original_data = wrf_ds.PREC_ACC_NC.values
    original_extent = (np.min(lons_2d), np.max(lons_2d), np.min(lats_2d), np.max(lats_2d))
    
    # Get reference grid from CPC
    cpc_file = os.path.join(cpc_data_dir, "cpc.h5")
    cpc_ds = xr.open_dataset(cpc_file)
    new_lons = cpc_ds.longitude.values
    new_lats = cpc_ds.latitude.values
    
    # Postprocessed WRF data
    times, wrf_data = regrid_wrf(wrf_ds, new_lons, new_lats)
    post_extent = (np.min(new_lons), np.max(new_lons), np.min(new_lats), np.max(new_lats))
    
    # Plot
    arrays = (original_data[time,:,:], wrf_data[time,:,:])
    titles = ("WRF (original)", "WRF (post-processed)")
    extents = (original_extent, post_extent)
    axis_labels = (("lon", "lat"), ("lon", "lat"))
    fig, _ = plot_maps(arrays, titles, extents, axis_labels=axis_labels)
    fig.savefig(os.path.join(figs_dir, "maps/wrf_postprocessed.png"))
    
    # Clip negative values to 0
    wrf_data[wrf_data < 0] = 0
    
    # Create and save new datasets
    wrf_data_dir = "/work/FAC/FGSE/IDYST/tbeucler/downscaling/mlima/data/test_data"
    create_folder(wrf_data_dir)
    write_dataset(times, new_lats, new_lons, wrf_data, os.path.join(wrf_data_dir, "wrf.h5"))


if __name__ == '__main__':
    main()
