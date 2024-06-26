import numpy as np
from scipy import interpolate
from tqdm import tqdm
import pyinterp


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


# this is slower because I probably don't know how to use pyinterp...
# but I'm leaving it here still
def pyinterp_data(data, lon_2d, lat_2d, new_lon_2d, new_lat_2d):
    # Allocate memory for the interpolated data
    interp_data = np.zeros((data.shape[0], new_lon_2d.shape[0], new_lon_2d.shape[1]))
    
    # Flatten the input longitude and latitude arrays
    lon_flat = lon_2d.flatten()
    lat_flat = lat_2d.flatten()
    points = np.vstack([lon_flat, lat_flat]).T
    
    # Pre-compute the new points for interpolation
    new_points = np.vstack([new_lon_2d.flatten(), new_lat_2d.flatten()]).T
    
    # Pre-compute the RTree
    mesh = pyinterp.RTree()
    mesh.packing(points, np.zeros_like(lon_flat))
    
    # Interpolate
    for t in tqdm(range(data.shape[0]), desc="Interpolating"):
        mesh.packing(points, data[t].flatten())
        idw, _ = mesh.inverse_distance_weighting(
            new_points,
            within=False,  # extrapolation is forbidden
            k=5,           # number of neighbors
            num_threads=0, # using all available threads
        )
        interp_data[t] = idw.reshape(new_lon_2d.shape)
    
    return interp_data


def cut_data(lon, lat, data, extent):
    lon_min, lon_max, lat_min, lat_max = extent
    lon_idx = np.where((lon >= lon_min) & (lon <= lon_max))[0]
    lat_idx = np.where((lat >= lat_min) & (lat <= lat_max))[0]
    lon = lon[lon_idx]
    lat = lat[lat_idx]
    data = data[:, lat_idx, :][:, :, lon_idx]
    return lon, lat, data
