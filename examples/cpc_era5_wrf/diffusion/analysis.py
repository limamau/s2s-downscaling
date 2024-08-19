import os, h5py
import matplotlib.pyplot as plt
from swirl_dynamics.data.hdf5_utils import read_single_array
from evaluation.plots import plot_maps


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

    # Save image
    arrays = (samples[0], samples[1], samples[2], samples[3])
    titles = (None, None, None, None)
    extents = (extent, extent, extent, extent)
    fig, _ = plot_maps(arrays, titles, extents)
    script_dir = os.path.dirname(os.path.realpath(__file__))
    figs_dir = os.path.join(script_dir, "figs")
    os.makedirs(figs_dir, exist_ok=True)
    fig.savefig("figs/image_4PM.png")
    
    
if __name__ == "__main__":
    experiment_name = "light"
    file_path = f"/work/FAC/FGSE/IDYST/tbeucler/downscaling/mlima/data/generated_forecasts/light_50.h5"
    extent_file_path = "/work/FAC/FGSE/IDYST/tbeucler/downscaling/mlima/data/test_data/cpc.h5"
    time_idx = 16
    main(file_path, extent_file_path, time_idx)
    