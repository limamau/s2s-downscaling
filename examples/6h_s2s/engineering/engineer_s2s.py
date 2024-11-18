import os, h5py
import numpy as np
import xarray as xr

from engineering.regridding import interpolate_data
from engineering.spectrum import get_1dpsd, radial_low_pass_filter
from utils import write_dataset, get_spatial_lengths

from configs.s2s import get_config


def cut_s2s_data_and_coords(raw_s2s_data, raw_s2s_lats, raw_s2s_lons, cpc_lats, cpc_lons):
    min_lat, max_lat = cpc_lats[0], cpc_lats[-1]
    min_lon, max_lon = cpc_lons[0], cpc_lons[-1]
    lat_idxs = np.where((raw_s2s_lats >= min_lat) & (raw_s2s_lats <= max_lat))[0]
    lon_idxs = np.where((raw_s2s_lons >= min_lon) & (raw_s2s_lons <= max_lon))[0]
    cut_s2s_data = raw_s2s_data[:, lat_idxs[0]:lat_idxs[-1]+1, lon_idxs[0]:lon_idxs[-1]+1]
    s2s_lons = raw_s2s_lons[lon_idxs]
    s2s_lats = raw_s2s_lats[lat_idxs]
    
    return cut_s2s_data, s2s_lats, s2s_lons


def regrid_s2s(raw_era5_lats, raw_era5_lons, raw_era5_data, new_lats, new_lons, method="linear"):
    old_lon_2d, old_lat_2d = np.meshgrid(raw_era5_lons, raw_era5_lats)
    new_lon_2d, new_lat_2d = np.meshgrid(new_lons, new_lats)
    
    new_data = interpolate_data(
        raw_era5_data,
        old_lon_2d,
        old_lat_2d,
        new_lon_2d,
        new_lat_2d,
        method=method,
    )
    
    return new_data


def get_idxs_to_keep(times, bound_dates, hourly_resolution):
    idxs_to_keep = []
    start, end = bound_dates
    start = np.datetime64(start) + np.timedelta64(6, 'h')
    end = np.datetime64(end) + np.timedelta64(24, 'h')
    idxs_to_keep = [i for i, t in enumerate(times) if start <= t <= end]
    return idxs_to_keep
    
    
def process_test_data(test_data_dir, storm_dates, storm_files, hourly_resolution):
    # Read info from already preprocessed CPC data
    cpc_file = os.path.join(test_data_dir, f"cpc_{hourly_resolution}h.h5")
    with h5py.File(cpc_file, "r") as f:
        cpc_lons = f["longitude"][:]
        cpc_lats = f["latitude"][:]
    
    # Read and concatenate S2S data
    first = True
    for file in storm_files:
        ds = xr.open_dataset(file)
        idxs_to_keep = get_idxs_to_keep(ds.time.values, storm_dates[0], get_idxs_to_keep)
        if first:
            first = False
            times = ds.time.values[idxs_to_keep]
            raw_s2s_lats = ds.latitude.values
            raw_s2s_lons = ds.longitude.values
            raw_s2s_data = ds.tp.values[idxs_to_keep]
        else:
            raw_s2s_data = np.concatenate([raw_s2s_data, ds.tp.values[idxs_to_keep]], axis=0)
            times = np.concatenate([times, ds.time.values[idxs_to_keep]], axis=0)
    
    # Sort data
    if raw_s2s_lats[0] > raw_s2s_lats[-1]:
        raw_s2s_lats = raw_s2s_lats[::-1]
        raw_s2s_data = raw_s2s_data[:, ::-1, :]
    if raw_s2s_lons[0] > raw_s2s_lons[-1]:
        raw_s2s_lons = raw_s2s_lons[::-1]
        raw_s2s_data = raw_s2s_data[:, :, ::-1]
        
    # Scale
    raw_s2s_data = raw_s2s_data / hourly_resolution
    
    # Cut according to lat/lon bounds
    cut_s2s_data, s2s_lats, s2s_lons = cut_s2s_data_and_coords(raw_s2s_data, raw_s2s_lats, raw_s2s_lons, cpc_lats, cpc_lons)
    
    # Interpolate s2s to the same resolution as CombiPrecip
    nearest_s2s_data = regrid_s2s(
        raw_s2s_lats, raw_s2s_lons, raw_s2s_data, cpc_lats, cpc_lons, method='nearest'
    )
    
    # (Radial) low-pass filter
    x_length, y_length = get_spatial_lengths(cpc_lons, cpc_lats)
    s2s_x_length, s2s_y_length = get_spatial_lengths(s2s_lons, s2s_lats)
    k, _ = get_1dpsd(cut_s2s_data, s2s_x_length, s2s_y_length)
    cutoff = np.max(k)
    nearest_lowpassed_s2s_data = radial_low_pass_filter(
        nearest_s2s_data, cutoff, x_length, y_length,
    )
    
    # Create and save new datasets
    write_dataset(
        times, s2s_lats, s2s_lons, cut_s2s_data,
        os.path.join(test_data_dir, "s2s.h5"),
    )
    write_dataset(
        times, cpc_lats, cpc_lons, nearest_s2s_data,
        os.path.join(test_data_dir, "s2s_nearest.h5"),
    )
    write_dataset(
        times, cpc_lats, cpc_lons, nearest_lowpassed_s2s_data,
        os.path.join(test_data_dir, "s2s_nearest_low-pass.h5"),
    )
    
    
def main():
    config = get_config()
    test_data_dir = config.test_data_dir
    storm_dates = config.storm_dates
    storm_files = config.storm_files
    hourly_resolution = 6
    
    process_test_data(
        test_data_dir, storm_dates, storm_files, hourly_resolution,
    )


if __name__ == "__main__":
    main()
