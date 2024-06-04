import os, shutil
import jax
import jax
import numpy as np
import jax.numpy as jnp
import xarray as xr
from math import radians, sin, cos, sqrt, atan2

# Utils function in alfabetical order

def batch_add(a, b):
    """
    Sum for each batch.
    """
    return jax.vmap(lambda a, b: a + b)(a, b)


def batch_mul(a, b):
    """
    Multiply for each batch.
    """
    return jax.vmap(lambda a, b: a * b)(a, b)


def create_folder(path, overwrite=False):
    """
    Creates a folder at the specified path, deleting the existing folder if overwrite is True.
    """
    if os.path.exists(path) & overwrite:
        shutil.rmtree(path)
    os.makedirs(path, exist_ok=True)


def find_intersection(wavelengths, era5_psd, cpc_psd, threshold=1):
    """
    Finds the intersection point of two curves (era5_psd and cpc_psd) based on the given wavelengths.
    """
    # Find the differences between the two curves
    y1 = np.log(era5_psd)[1:] # there's a problem with the first value?
    y2 = np.log(cpc_psd)[1:]
    diff = y2 - y1

    # Find the index of the minimum absolute difference (closest to zero)
    idx = np.where(diff < threshold)[0][0]

    # Return the wavelength and y-value at the intersection point
    lambda_star = wavelengths[idx+1 +1]
    psd_star = cpc_psd[idx+1 +1]

    return lambda_star, psd_star


def gaussian_noise(data, sigma):
    """
    Adds Gaussian noise to the input data with the specified standard deviation (sigma).
    """
    noise = np.random.normal(0, sigma, data.shape)
    return data + noise


def get_spatial_lengths(lons, lats):
    """
    Calculates the spatial lengths (x_length and y_length) based on the given longitude and latitude values.
    """
    # Get spatial lengths
    x_length = haversine(lons.min(), lats.min(), lons.max(), lats.min())
    y_length = haversine(lons.min(), lats.min(), lons.min(), lats.max())
    
    return x_length, y_length

# Utils function in alfabetical order

def batch_add(a, b):
    """
    Sum for each batch.
    """
    return jax.vmap(lambda a, b: a + b)(a, b)


def batch_mul(a, b):
    """
    Multiply for each batch.
    """
    return jax.vmap(lambda a, b: a * b)(a, b)


def create_folder(path, overwrite=False):
    """
    Creates a folder at the specified path, deleting the existing folder if overwrite is True.
    """
    if os.path.exists(path) & overwrite:
        shutil.rmtree(path)
    os.makedirs(path, exist_ok=True)


def find_intersection(wavelengths, era5_psd, cpc_psd, threshold=1):
    """
    Finds the intersection point of two curves (era5_psd and cpc_psd) based on the given wavelengths.
    """
    # Find the differences between the two curves
    y1 = np.log(era5_psd + 10e-20)[1:] # there's a problem with the first value?
    y2 = np.log(cpc_psd + 10e-20)[1:]
    diff = y2 - y1

    # Find the index of the minimum absolute difference (closest to zero)
    idx = np.where(diff < threshold)[0][0]

    # Return the wavelength and y-value at the intersection point
    lambda_star = wavelengths[idx+1]
    psd_star = cpc_psd[idx+1]

    return lambda_star, psd_star


def get_pdf(x, bins):
    """
    Calculates the Probability Density Function (PDF) of the input data based on the given bins.
    """
    pdf, _ = np.histogram(x, bins)
    return pdf / pdf.sum()


def get_cdf(x, bins):
    """
    Calculates the Cumulative Distribution Function (CDF) of the input data based on the given bins.
    """
    pdf = get_pdf(x, bins)
    cdf = np.cumsum(pdf)
    cdf = np.insert(cdf, 0, 0.0)
    return cdf / cdf.max()


def get_spatial_lengths(lons, lats):
    """
    Calculates the spatial lengths (x_length and y_length) based on the given longitude and latitude values.
    """
    # Get spatial lengths
    x_length = haversine(lons.min(), lats.min(), lons.max(), lats.min())
    y_length = haversine(lons.min(), lats.min(), lons.min(), lats.max())
    
    return x_length, y_length


def haversine(lon1, lat1, lon2, lat2):
    """
    Calculates the distance between two points on the Earth's surface using the Haversine formula.
    """
    # Convert latitude and longitude from degrees to radians
    lat1_rad = radians(lat1)
    lon1_rad = radians(lon1)
    lat2_rad = radians(lat2)
    lon2_rad = radians(lon2)
    
    # Haversine formula
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad
    a = sin(dlat / 2)**2 + cos(lat1_rad) * cos(lat2_rad) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    R = 6371  # Radius of the Earth in kilometers
    distance = R * c
    
    return distance


def log_transform(data, epsilon=0.00001):
    """
    Applies a log transformation to the input data using the formula x̃ = log(x + ϵ) − log(ϵ).
def log_transform(data, epsilon=0.005):
    """
    return jnp.log(data + epsilon) - jnp.log(epsilon)


def normalize_data(data, sigma_data=1.0):
    """
    Normalizes the input data to have zero mean and the specified standard deviation (sigma_data).
    """
    mean = jnp.mean(data)
    std = jnp.std(data)
    return (data - mean) * sigma_data / std , mean, std


def unlog_transform(data, epsilon=0.00001):
    """
    Inverse of log transformation.
    """
    return jnp.exp(data + jnp.log(epsilon)) - epsilon


def unnormalize_data(data, mean, std, sigma_data=1.0):
    """
    Inverse of normalization.
    """
    return data * std / sigma_data + mean


def write_dataset(times, lats, lons, data, filename):
    """
    Creates an xarray dataset from the input data and writes it to a NetCDF file at the specified filename.
    """
    # Create dataset
    ds = xr.Dataset(
        {
            "precip": (["time", "latitude", "longitude"], data),
        },
        coords = {
            "time": (["time"], times),
            "latitude": (["latitude"], lats),
            "longitude": (["longitude"], lons),
        },
    )
    
    # Write dataset
    ds.to_netcdf(filename, engine="h5netcdf")
    
    
# I'm leaving this here...
# # Interpolate functions to find approximate intersection
# x  = np.linspace(min(x1[0], x2[0]), max(x1[-1], x2[-1]), 100)
# f1 = np.interp(x, x1, y1)
# f2 = np.interp(x, x2, y2)

# interp1 = interpolate.InterpolatedUnivariateSpline(x, f1)
# interp2 = interpolate.InterpolatedUnivariateSpline(x, f2)

# def difference(x):
#     return np.abs(interp1(x) - interp2(x))

# return optimize.fsolve(difference, x0=x0)