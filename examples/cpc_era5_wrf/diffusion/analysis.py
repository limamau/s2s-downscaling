import os, h5py
from swirl_dynamics.data.hdf5_utils import read_single_array
from evaluation.plots import plot_maps, plot_cdfs, plot_psds, plot_pp
from utils import get_spatial_lengths


def main(file_path: str, extent_file_path: str, time_idx: int):
    # Read extent from file
    with h5py.File(extent_file_path, "r") as f:
        lons = f['longitude'][:]
        lats = f['latitude'][:]
    extent = (lons.min(), lons.max(), lats.min(), lats.max())
    
    # Read the dataset from the .hdf5 file
    images = read_single_array(file_path, "precip")

    # Get the image at the specified time index
    samples = images[time_idx, :, :, :, 0]

    # Plot maps
    arrays = (samples[0], samples[1], samples[2], samples[3])
    titles = ("a) Sample 1", "b) Sample 2", "c) Sample 3", "d) Sample 4")
    extents = (extent,) * 4
    fig, _ = plot_maps(arrays, titles, extents)
    script_dir = os.path.dirname(os.path.realpath(__file__))
    figs_dir = os.path.join(script_dir, "figs")
    os.makedirs(figs_dir, exist_ok=True)
    fig.savefig("figs/map_t{}.png".format(time_idx))
    
    # Plot CDFs
    samples = images[:, :, :, :, 0]
    arrays = (samples[:,0], samples[:,1], samples[:,2], samples[:,3])
    titles = ("Sample 1", "Sample 2", "Sample 3", "Sample 4")
    fig, _ = plot_cdfs(arrays, titles)
    fig.savefig("figs/cdf.png")
    
    # Plot PSDs
    spatial_lengths = get_spatial_lengths(lons, lats)
    spatial_lengths = (spatial_lengths,) * 4
    fig, _ = plot_psds(arrays, titles, spatial_lengths)
    fig.savefig("figs/psd.png")
    
    # Plot PP
    fig, _ = plot_pp(arrays, titles)
    fig.savefig("figs/pp.png")
    
    
if __name__ == "__main__":
    experiment_name = "light"
    file_path = f"/work/FAC/FGSE/IDYST/tbeucler/downscaling/mlima/data/generated_forecasts/light_50.h5"
    test_file_path = "/work/FAC/FGSE/IDYST/tbeucler/downscaling/mlima/data/test_data/cpc.h5"
    time_idx = 16
    main(file_path, test_file_path, time_idx)
