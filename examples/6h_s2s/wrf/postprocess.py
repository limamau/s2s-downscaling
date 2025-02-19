import os, tomllib
import numpy as np
import xarray as xr
from tqdm import tqdm

from data.surface_data import SurfaceData
from engineering.regridding import interpolate_data

from configs.config import get_config

HOURLY_RESOLUTION = 6
EVENT_DAYS = 2


### auxiliary functions ###
def concat_wrf(data_dir, dates):
    # function to write np.datetime64 in the format of WRF output files from the scripts
    def wrf_format_time(time):
        return "wrfout_d03_" + time.astype(str).replace("T", "_")[:19]
    
    # iteration over the dates
    datasets = []
    print("Concatenating WRF files...")
    for date in tqdm(dates):
        file = wrf_format_time(date)
        ds = xr.open_dataset(os.path.join(data_dir, file), engine='h5netcdf')[['PREC_ACC_NC']]
        datasets.append(ds)
    
    return xr.concat(datasets, dim="Time")


def regrid_wrf(ds, new_lons, new_lats):
    # get arrays from the datasets
    lons_2d = ds.XLONG.values[[0],:,:]
    lats_2d = ds.XLAT.values[[0],:,:]
    data = ds.PREC_ACC_NC.values
    
    # get grid from reference latitude and longitudes
    new_lon_2d, new_lat_2d = np.meshgrid(new_lons, new_lats)
    
    # interpolation of the irregular latlon to the regular latlon
    new_data = interpolate_data(
        data,
        lons_2d,
        lats_2d,
        new_lon_2d,
        new_lat_2d,
    )
    
    return ds.XTIME.values, new_data


### main call ###
def run_processing(wrf_output_dir, cpc_file):
    # reference
    cpc = SurfaceData.load_from_h5(cpc_file, ["precip"])
    latitudes, longitudes = cpc.latitude, cpc.longitude
    
    # dataset to be processed
    num_event_idxs = EVENT_DAYS * 24 // HOURLY_RESOLUTION
    # TODO: take out hard coding
    initial_idx = 0
    wrf_ds = concat_wrf(wrf_output_dir, cpc.time[initial_idx:initial_idx+num_event_idxs])
    
    # processing
    times, wrf_data = regrid_wrf(wrf_ds, longitudes, latitudes)
    wrf_data[wrf_data < 0] = 0
    return SurfaceData(
        times, latitudes, longitudes, precip=wrf_data,
    )


def main():
    # directory paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(script_dir, "../dirs.toml"), "rb") as f:
        dirs = tomllib.load(f)
    raw_wrf_dir = dirs["raw"]["wrf"]
    base = dirs["main"]["base"]
    simulations_dir = os.path.join(base, dirs["subs"]["simulations"])
    test_data_dir = os.path.join(base, dirs["subs"]["test"])
    
    # extra configurations
    config = get_config()
    wrf_output_dir = os.path.join(raw_wrf_dir, config.output_dir)
    cpc_file = os.path.join(test_data_dir, config.cpc_file)
    wrf_simulations_dir = os.path.join(simulations_dir, config.wrf_simulations_dir)
    os.makedirs(wrf_simulations_dir, exist_ok=True)

    # main calls
    wrf = run_processing(wrf_output_dir, cpc_file)
    wrf.save_to_h5(os.path.join(wrf_simulations_dir, "wrf.h5"))

if __name__ == '__main__':
    main()
