import numpy as np
from utils import get_cdf


# Inspired by https://github.com/btschwertfeger/python-cmethods/


def get_inverse_of_cdf(base_cdf, insert_cdf, xbins):
    # Interpolate to find the inverse CDF values
    return np.interp(insert_cdf, base_cdf, xbins)


def quantile_mapping(obs, simh, simp, n_quantiles=2000):
    # Find global min and max values across observed and historical simulated data
    global_max = max(np.nanmax(obs), np.nanmax(simh))
    global_min = min(np.nanmin(obs), np.nanmin(simh))
    
    # Define bins for the histogram
    wide = abs(global_max - global_min) / n_quantiles
    xbins = np.arange(global_min, global_max + wide, wide)

    # Calculate the CDFs for the observed and historical simulated data
    cdf_obs = get_cdf(obs, xbins)
    cdf_simh = get_cdf(simh, xbins)

    # Normalize the historical simulated CDF to match the range of the observed CDF
    cdf_simh = np.interp(cdf_simh, (cdf_simh.min(), cdf_simh.max()), (cdf_obs.min(), cdf_obs.max()))

    # Interpolate to find where `simp` values fall in the normalized historical simulated CDF
    epsilon = np.interp(simp, xbins, cdf_simh)
    
    corrected_simp = get_inverse_of_cdf(cdf_obs, epsilon, xbins)
    
    return corrected_simp