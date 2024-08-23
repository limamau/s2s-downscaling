import os, h5py
from evaluation.plots import *
from utils import get_spatial_lengths
from config import get_config

def main():
    config = get_config()
    test_data_dir = config.test_data_dir
    sim_data_dir = config.sim_data_dir
    
    # era5
    with h5py.File(os.path.join(test_data_dir, "era5.h5"), "r") as f:
        era5 = f["precip"][:,:,:]
        era5_lons = f["longitude"][:]
        era5_lats = f["latitude"][:]
    
    # era5 + nearest neighbors + low-pass
    with h5py.File(os.path.join(test_data_dir, "era5_nearest_low-pass.h5"), "r") as f:
        era5_nearest_lowpass = f["precip"][:,:,:]
    
    # point-to-point
    with h5py.File(os.path.join(sim_data_dir, "qm_nearest_low-pass_point.h5"), "r") as f:
        qm = f["precip"][:,:,:]
        
    # cpc
    with h5py.File(os.path.join(test_data_dir, "cpc.h5"), "r") as f:
        cpc = f["precip"][:,:,:]
        cpc_lats = f["latitude"][:]
        cpc_lons = f["longitude"][:]
        
    # Plots maps
    script_dir = os.path.dirname(os.path.realpath(__file__))
    figs_dir = os.path.join(script_dir, "figs")
    time = -10
    arrays = (era5[time], era5_nearest_lowpass[time], qm[time], cpc[time])
    titles = ("ERA5", "ERA5 + nearest neighbors + low-pass", "Quantile Mapping", "CombiPrecip")
    era5_extent = (era5_lons[0], era5_lons[-1], era5_lats[0], era5_lats[-1])
    cpc_extent = (cpc_lons[0], cpc_lons[-1], cpc_lats[0], cpc_lats[-1])
    extents = (era5_extent, cpc_extent, cpc_extent, cpc_extent)
    fig, _ = plot_maps(arrays, titles, extents)
    fig.savefig(os.path.join(figs_dir, "maps.png"))

    
if __name__ == "__main__":
    main()
