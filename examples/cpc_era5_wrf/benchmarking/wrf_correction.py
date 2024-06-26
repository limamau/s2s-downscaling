import os, h5py
import xarray as xr
from utils import write_dataset

def main():
    test_dir = "/work/FAC/FGSE/IDYST/tbeucler/downscaling/mlima/data/test_data"
    with h5py.File(os.path.join(test_dir, "wrf.h5"), 'r') as file:
        data = file['precip'][12:60,:,:]
        lons = file['longitude'][:]
        lats = file['latitude'][:]
        
    times = xr.open_dataset(os.path.join(test_dir, "wrf.h5"))['time'].values[12:60]
        
    print("First time:", times[0])
    print("Last time:", times[-1])
    
    write_dataset(times, lats, lons, data, os.path.join(test_dir, "wrf.h5"))
    
if __name__ == '__main__':
    main()
