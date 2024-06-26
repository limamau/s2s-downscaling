import os, h5py
import xarray as xr

from evaluation.plots import plot_maps

def main():
    test_dir = "/work/FAC/FGSE/IDYST/tbeucler/downscaling/mlima/data/test_data"
    
    time = -10
    
    # Combi Precip
    with h5py.File(os.path.join(test_dir, "cpc.h5"), "r") as f:
        cpc_data = f["precip"][time,:,:]
        
    # Raw ERA5
    with h5py.File(os.path.join(test_dir, "era5_raw.h5"), "r") as f:
        raw_data = f["precip"][time,:,:]
    
    # Interpolated
    with h5py.File(os.path.join(test_dir, "era5_interpolated.h5"), "r") as f:
        interpolated_data = f["precip"][time,:,:]
        
    # Low-pass filtered
    with h5py.File(os.path.join(test_dir, "era5_low-pass.h5"), "r") as f:
        lowpassed_data = f["precip"][time,:,:]
        lons = f["longitude"][:]
        lats = f["latitude"][:]
    
    times = xr.open_dataset(os.path.join(test_dir, "cpc.h5"), engine='h5netcdf').time.values
    print("Times: ", times[time])
        
    arrays = (raw_data, interpolated_data, lowpassed_data, cpc_data)
    titles = ("ERA5 (original)", "ERA5 (x25)", "ERA5 (low-passed)", "CombiPrecip")
    titles = (None, None, None, None)
    extent = (lons[0], lons[-1], lats[0], lats[-1])
    extents = (extent, extent, extent, extent)
    fig, _ = plot_maps(arrays, titles, extents)
    script_dir = os.path.dirname(os.path.realpath(__file__))
    maps_dir = os.path.join(script_dir, "figs/maps")
    fig.savefig(os.path.join(maps_dir, "era5_steps_test.png"))
    
    print("Done!")
    
    
if __name__ == "__main__":
    main()