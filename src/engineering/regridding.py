import numpy as np
from scipy import interpolate


def regularize_grid(lon_2d, lat_2d, nx, ny):
    lon_min, lon_max = lon_2d.min(), lon_2d.max()
    lat_min, lat_max = lat_2d.min(), lat_2d.max()
    new_lon = np.linspace(lon_min, lon_max, nx)
    new_lat = np.linspace(lat_min, lat_max, ny)
    new_lon_2d, new_lat_2d = np.meshgrid(new_lon, new_lat)
    
    return new_lon, new_lat, new_lon_2d, new_lat_2d


def interpolate_data(
    data, 
    lon_2d, 
    lat_2d,
    new_lon_2d,
    new_lat_2d,
    method='linear'
):
    """
    Interpolate a dataset on a 2D grid to a new 2D grid, dynamically handling extra dimensions.
    
    Parameters:
        data: np.ndarray
            Input data to interpolate. Should have dimensions (..., time, latitude, longitude).
        lon_2d: np.ndarray
            2D array of longitudes for the original grid.
        lat_2d: np.ndarray
            2D array of latitudes for the original grid.
        new_lon_2d: np.ndarray
            2D array of longitudes for the new grid.
        new_lat_2d: np.ndarray
            2D array of latitudes for the new grid.
        method: str, optional
            Interpolation method ('linear', 'nearest', etc.). Default is 'linear'.
    
    Returns:
        new_data: np.ndarray
            Interpolated data with the same extra dimensions and shape as the input, but with new grid.
    """
    # Validate input dimensions
    if lon_2d.shape != lat_2d.shape:
        raise ValueError("lon_2d and lat_2d must have the same shape.")
    if new_lon_2d.shape != new_lat_2d.shape:
        raise ValueError("new_lon_2d and new_lat_2d must have the same shape.")

    # Extract the shape of the grid and data
    original_shape = data.shape
    extra_dims = original_shape[:-3]  # dimensions before time, lat, lon
    time_dim = original_shape[-3]

    # Prepare the interpolation grid
    lonlat = np.stack([lon_2d.flatten(), lat_2d.flatten()], axis=1)
    new_lonlat = np.stack([new_lon_2d.flatten(), new_lat_2d.flatten()], axis=1)
    
    # Prepare the output array
    new_shape = extra_dims + (time_dim, new_lon_2d.shape[0], new_lon_2d.shape[1])
    new_data = np.zeros(new_shape, dtype=data.dtype)

    # Interpolation for each slice along extra dimensions
    for idx in np.ndindex(*extra_dims):  # Iterate over all extra dimensions
        sliced_data = data[idx]  # Extract the slice with shape (time, lat, lon)
        reshaped_data = sliced_data.reshape(time_dim, -1).T  # Flatten time-lat-lon into (points, time)
        interpolated = interpolate.griddata(lonlat, reshaped_data, new_lonlat, method=method, fill_value=0.0)
        new_data[idx] = interpolated.T.reshape(time_dim, new_lon_2d.shape[0], new_lon_2d.shape[1])

    return new_data


def cut_data(lons, lats, data, extent):
    lon_min, lon_max, lat_min, lat_max = extent
    lon_idx = np.where((lons >= lon_min) & (lons <= lon_max))[0]
    lat_idx = np.where((lats >= lat_min) & (lats <= lat_max))[0]
    lons = lons[lon_idx]
    lats = lats[lat_idx]
    data = data[..., lat_idx, :][..., lon_idx]
    return lats, lons, data
