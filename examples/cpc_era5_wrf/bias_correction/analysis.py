import os, h5py
from matplotlib import pyplot as plt
from evaluation.plots import plot_psds, plot_maps
from utils import get_spatial_lengths

def main():
    test_data_dir = "/work/FAC/FGSE/IDYST/tbeucler/downscaling/mlima/data/test_data"
    
    # all
    with h5py.File(os.path.join(test_data_dir, "era5_qm_all.h5"), "r") as f:
        era5_qm_all = f["precip"][:,:,:]
        
    # point-to-point
    with h5py.File(os.path.join(test_data_dir, "era5_qm_point.h5"), "r") as f:
        era5_qm_point = f["precip"][:,:,:]
        
    # era5
    with h5py.File(os.path.join(test_data_dir, "era5_low-pass.h5"), "r") as f:
        era5 = f["precip"][:,:,:]
        
    # cpc
    with h5py.File(os.path.join(test_data_dir, "cpc.h5"), "r") as f:
        cpc = f["precip"][:,:,:]
        lats = f["latitude"][:]
        lons = f["longitude"][:]
        
    # Plots PSDs
    script_dir = os.path.dirname(os.path.realpath(__file__))
    figs_dir = os.path.join(script_dir, "figs")
    x_length, y_length = get_spatial_lengths(lons, lats)
    fig, _ = plot_psds(
        (era5_qm_all, era5_qm_point, cpc),
        ("QM (all)", "QM (point-to-point)", "CombiPrecip"),
        ((x_length, y_length), (x_length, y_length), (x_length, y_length)),
        min_threshold=1e-10,
        lambda_star=680,
    )
    fig.savefig(os.path.join(figs_dir, "psd.png"))
    
    # Plot maps comparison
    time = -10
    arrays = (era5_qm_all[time], era5_qm_point[time])
    # titles = ("QM (all)", "QM (point-to-point)")
    titles = (None, None)
    extent = (lons[0], lons[-1], lats[0], lats[-1])
    extents = (extent, extent)
    fig, _ = plot_maps(arrays, titles, extents)
    fig.savefig(os.path.join(figs_dir, "maps.png"))
        

    
if __name__ == "__main__":
    main()
