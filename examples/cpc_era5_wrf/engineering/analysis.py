import os, h5py
import xarray as xr

from evaluation.plots import plot_maps

from configs.analysis import get_config

def plot(test_data_dir, time):
    # Combi Precip
    with h5py.File(os.path.join(test_data_dir, "cpc.h5"), "r") as f:
        cpc_data = f["precip"][time,:,:]
        lons = f["longitude"][:]
        lats = f["latitude"][:]
        
    # Raw ERA5
    with h5py.File(os.path.join(test_data_dir, "era5_raw.h5"), "r") as f:
        raw_data = f["precip"][time,:,:]
    
    # Nearest neighbor interpolation + low-pass filter
    with h5py.File(os.path.join(test_data_dir, "era5_nearest_low-pass.h5"), "r") as f:
        nearest_data = f["precip"][time,:,:]
        
    # Linear interpolation + low-pass filter
    with h5py.File(os.path.join(test_data_dir, "era5_linear_low-pass.h5"), "r") as f:
        linear_data = f["precip"][time,:,:]
    
    times = xr.open_dataset(os.path.join(test_data_dir, "cpc.h5"), engine='h5netcdf').time.values
    print("Times: ", times[time])
        
    arrays = (raw_data, nearest_data, linear_data, cpc_data)
    titles = ("ERA5 (original)", "ERA5 (nearest + low-pass)", "ERA5 (linear + low-pass)", "CombiPrecip")
    extent = (lons[0], lons[-1], lats[0], lats[-1])
    extents = (extent, extent, extent, extent)
    fig, _ = plot_maps(arrays, titles, extents)
    script_dir = os.path.dirname(os.path.realpath(__file__))
    maps_dir = os.path.join(script_dir, "figs/maps")
    fig.savefig(os.path.join(maps_dir, "era5_comparison.png"))
    
    print("Done!")
    
    
def main():
    config = get_config()
    test_data_dir = config.test_data_dir
    time = -10
    plot(test_data_dir, time)
    
if __name__ == "__main__":
    main()