import os, glob
import h5py
import numpy as np
import xarray as xr
from datetime import timedelta
from scipy import interpolate
from pyproj import Transformer
from tqdm import tqdm

def transform_cpc_coordinates_old(ds):
    # Transform coordinates from swiss coordinates to lat,lon
    transformer = Transformer.from_proj(2056, 4326, always_xy=True)
    xx, yy = np.meshgrid(ds.x.values, ds.y.values)
    lon_2d, lat_2d = transformer.transform(xx, yy)

    return lon_2d, lat_2d


def transform_cpc_coordinates(xs, ys):
    # Transform coordinates from swiss coordinates to lat,lon
    transformer = Transformer.from_proj(2056, 4326, always_xy=True)
    xx, yy = np.meshgrid(xs, ys)
    lon_2d, lat_2d = transformer.transform(xx, yy)

    return lon_2d, lat_2d


def regularize_grid(lon_2d, lat_2d, nx, ny):
    lon_min, lon_max = lon_2d.min(), lon_2d.max()
    lat_min, lat_max = lat_2d.min(), lat_2d.max()
    new_lon = np.linspace(lon_min, lon_max, nx)
    new_lat = np.linspace(lat_min, lat_max, ny)
    new_lon_2d, new_lat_2d = np.meshgrid(new_lon, new_lat)
    
    return new_lon, new_lat, new_lon_2d, new_lat_2d


def interpolate_data(data, lon_2d, lat_2d, new_lon_2d, new_lat_2d, final_shape):
    # Create 2D arrays to interpolate
    lonlat = np.stack([lon_2d.flatten(), lat_2d.flatten()], axis=1)
    new_lonlat = np.stack([new_lon_2d.flatten(), new_lat_2d.flatten()], axis=1)
    
    # Interpolation 
    reshaped_data = data.reshape(data.shape[0], -1).T
    new_data = interpolate.griddata(lonlat, reshaped_data, new_lonlat, method='linear')
    new_data = new_data.T.reshape(final_shape)
    
    return new_data


def cut_precip(lon, lat, precip, extent):
    lon_min, lon_max, lat_min, lat_max = extent
    lon_idx = np.where((lon >= lon_min) & (lon <= lon_max))[0]
    lat_idx = np.where((lat >= lat_min) & (lat <= lat_max))[0]
    lon = lon[lon_idx]
    lat = lat[lat_idx]
    precip = precip[:, lat_idx, :][:, :, lon_idx]
    return lon, lat, precip


def regrid_cpc_old(ds, extent):
    # Transform cooridinates from CH1903 to WGS84
    lon_2d, lat_2d = transform_cpc_coordinates_old(ds)
    
    # Create new regular lon,lat vectors and grid
    new_lon, new_lat, new_lon_2d, new_lat_2d = regularize_grid(lon_2d, lat_2d, ds.x.size, ds.y.size)
    
    # Interpolation of the irregular latlon to the regular latlon
    new_data = interpolate_data(ds.CPC.values, lon_2d, lat_2d, new_lon_2d, new_lat_2d, ds.CPC.shape)
    
    # Cut data to the swiss region
    new_lon, new_lat, new_data = cut_precip(new_lon, new_lat, new_data, extent)
    
    return ds.REFERENCE_TS.values, new_lat, new_lon, new_data


def regrid_cpc(data, xs, ys, extent):
    # Transform cooridinates from CH1903 to WGS84
    lon_2d, lat_2d = transform_cpc_coordinates(xs, ys)
    
    # Create new regular lon,lat vectors and grid
    new_lon, new_lat, new_lon_2d, new_lat_2d = regularize_grid(lon_2d, lat_2d, xs.size, ys.size)
    
    # Interpolation of the irregular latlon to the regular latlon
    new_data = interpolate_data(data, lon_2d, lat_2d, new_lon_2d, new_lat_2d, data.shape)
    
    # Cut data to the swiss region
    new_lon, new_lat, new_data = cut_precip(new_lon, new_lat, new_data, extent)
    
    return new_lat, new_lon, new_data


def regrid_era5(raw_era5_ds, times, new_lats, new_lons):
    # Get arrays from the datasets
    era5_lon = raw_era5_ds.longitude.values
    era5_lat = raw_era5_ds.latitude.values
    Nt = len(times)
    data = raw_era5_ds.tp.values[:Nt,:,:]
    
    # Create grid
    old_lon_2d, old_lat_2d = np.meshgrid(era5_lon, era5_lat)
    new_lon_2d, new_lat_2d = np.meshgrid(new_lons, new_lats)
    
    # Create 2D arrays to interpolate
    new_data = interpolate_data(
        data,
        old_lon_2d,
        old_lat_2d,
        new_lon_2d,
        new_lat_2d,
        (data.shape[0], new_lats.size, new_lons.size)
    )
    
    return new_data*1000


def regrid_wrf(ds, new_lons, new_lats):
    # Get arrays from the datasets
    lons_2d = ds.XLONG.values[[0],:,:]
    lats_2d = ds.XLAT.values[[0],:,:]
    data = ds.PREC_ACC_NC.values
    
    # Get grid from reference latitude and longitudes
    new_lon_2d, new_lat_2d = np.meshgrid(new_lons, new_lats)
    
    # Interpolation of the irregular latlon to the regular latlon
    new_data = interpolate_data(
        data,
        lons_2d,
        lats_2d,
        new_lon_2d,
        new_lat_2d,
        (data.shape[0], new_lats.size, new_lons.size),
    )
    
    return ds.XTIME.values, new_data


def concat_cpc_old(data_dir):
    datasets = []
    for file in sorted(os.listdir(data_dir)):
        if file.startswith("CPC_00060_H_2024") & file.endswith(".nc"):
            ds = xr.open_dataset(os.path.join(data_dir, file))
            datasets.append(ds)
    return xr.concat(datasets, dim="REFERENCE_TS")


def concat_cpc(data_dir, initial_date, final_date):
    # Get associated times array
    times = [initial_date + timedelta(days=n, hours=h) for n in range((final_date - initial_date).days + 1) for h in range(24)]
    
    # Get first dataset
    pattern = f"CPC{initial_date.strftime('%y%j')}"
    file_path = glob.glob(os.path.join(data_dir, pattern) + '*')[0]
    h5_file = h5py.File(file_path, 'r')
    data = h5_file['dataset1/data1/data'][...]
    h5_file.close()
    
    # Memory allocation for concatenated array
    data = np.empty((len(times), *data.shape), dtype=data.dtype)
    
    # Loop over all the files in the directory matching the given dates
    for d in tqdm(range(int((final_date - initial_date).days)+1), desc="Concatenating CPC files"):
        date = initial_date + timedelta(d)
        pattern = f"CPC{date.strftime('%y%j')}"
        for (h, file_path) in enumerate(sorted(glob.glob(os.path.join(data_dir, pattern) + '*'))):
            with h5py.File(file_path, 'r') as h5_file:
                data[24*d+h,:,:] = h5_file['dataset1/data1/data'][...]
    
    return times, data


def concat_wrf(data_dir):
    datasets = []
    for file in sorted(os.listdir(data_dir)):
        if file.startswith("wrfout_d03_"):
            ds = xr.open_dataset(os.path.join(data_dir, file))[['PREC_ACC_NC']]
            datasets.append(ds)
    return xr.concat(datasets, dim="Time")