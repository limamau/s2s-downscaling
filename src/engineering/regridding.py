import numpy as np
from scipy import interpolate


def regularize_grid(lon_2d, lat_2d, nx, ny):
    lon_min, lon_max = lon_2d.min(), lon_2d.max()
    lat_min, lat_max = lat_2d.min(), lat_2d.max()
    new_lon = np.linspace(lon_min, lon_max, nx)
    new_lat = np.linspace(lat_min, lat_max, ny)
    new_lon_2d, new_lat_2d = np.meshgrid(new_lon, new_lat)
    
    return new_lon, new_lat, new_lon_2d, new_lat_2d


def interpolate_data(data, lon_2d, lat_2d, new_lon_2d, new_lat_2d, method='linear'):
    # Create 2D arrays to interpolate
    lonlat = np.stack([lon_2d.flatten(), lat_2d.flatten()], axis=1)
    new_lonlat = np.stack([new_lon_2d.flatten(), new_lat_2d.flatten()], axis=1)
    
    # Interpolation
    reshaped_data = data.reshape(data.shape[0], -1).T
    new_data = interpolate.griddata(lonlat, reshaped_data, new_lonlat, method=method, fill_value=0.0)
    new_data = new_data.T.reshape(data.shape[0], new_lon_2d.shape[0], new_lon_2d.shape[1])
    
    return new_data


def cut_data(lon, lat, data, extent):
    lon_min, lon_max, lat_min, lat_max = extent
    lon_idx = np.where((lon >= lon_min) & (lon <= lon_max))[0]
    lat_idx = np.where((lat >= lat_min) & (lat <= lat_max))[0]
    lon = lon[lon_idx]
    lat = lat[lat_idx]
    data = data[:, lat_idx, :][:, :, lon_idx]
    return lon, lat, data
