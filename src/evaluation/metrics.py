import numpy as np
import properscoring as ps
from scipy.stats import wasserstein_distance as wass_dist
from scipy.stats import rankdata

from utils import get_cdf, get_pdf
from engineering.spectrum import get_2dpsd


# Metric functions in alphabetical order:

def cdf_distance(obs, sim, n_quantiles=200, distance="l2"):
    """
    Calculate the CDF distance between two arrays.

    ## Parameters:
    distance (string): Distance metric to use. Options are "l1", "l2", and "max". 
    obs (array): Observed values.
    sim (array): Simulated values.
    The latest corresponds to the Kolmogorov-Smirnov distance.
    

    ## Returns:
    cdf_distance (float): CDF distance.
    """
    # Define bins for the histogram
    global_max = max(np.nanmax(obs), np.nanmax(sim))
    global_min = min(np.nanmin(obs), np.nanmin(sim))
    wide = abs(global_max - global_min) / n_quantiles
    bins = np.arange(global_min, global_max + wide, wide)
    
    # Get CDF
    cdf_obs = get_cdf(obs, bins)
    cdf_sim = get_cdf(sim, bins)
    
    match distance:
        case "l1":
            cdf_distance = np.mean(np.abs(cdf_obs - cdf_sim))
        case "l2":
            cdf_distance = np.sqrt(np.mean((cdf_obs - cdf_sim) ** 2))
        case "max":
            cdf_distance = np.max(np.abs(cdf_obs - cdf_sim))
        case _:
            raise ValueError(f"Invalid distance metric: {distance}")
            
    return cdf_distance


def crps(obs, sim):
    """
    Calculate the Continuous Ranked Probability Score (CRPS) between two arrays.
    Currently using the properscoring library.
    
    ## Parameters:
    obs (array): Observed values.
    sim (array): Simulated values.
    
    ## Returns:
    crps (float): Continuous Ranked Probability Score.
    """
    if len(sim.shape) > len(obs.shape):
        # this is done because I use the ensemble dimension as the first one
        # (actually the second after lead time, but we calculate that for each lead time)
        # and properscoring expects the ensemble dimension as the last one
        sim = np.transpose(sim)
        obs = np.transpose(obs)
    return np.mean(ps.crps_ensemble(obs, sim))


def mean_absolute_error(obs, sim, axis=(0,1,2)):
    """
    Calculate the Mean Absolute Error (MAE) between two arrays.

    ## Parameters:
    obs (array): Observed values.
    sim (array): Simulated values.
    axis (tuple): Axes over which to compute the MAE. Default is (0,1,2).

    ## Returns:
    mae (float): Mean Absolute Error.
    """
    return np.mean(np.abs(obs - sim))


def perkins_skill_score(obs, sim, n_quantiles=2000):
    """
    Calculate the Perkins Skill Score (PSS) between two arrays.

    ## Parameters:
    obs (array): Observed values.
    sim (array): Simulated values.

    ## Returns:
    pss (float): Perkins Skill Score.
    """
    # Find global min and max values across observed and historical simulated data
    global_max = np.maximum(np.nanmax(obs), np.nanmax(sim))
    global_min = np.minimum(np.nanmin(obs), np.nanmin(sim))
    
    # Define bins for the histogram
    wide = np.abs(global_max - global_min) / n_quantiles
    if np.isnan(wide):
        return np.NAN
    bins = np.arange(global_min, global_max + wide, wide)
    
    pdf_obs = get_pdf(obs, bins)
    pdf_sim = get_pdf(sim, bins)
    
    pss = np.sum(np.minimum(pdf_obs, pdf_sim))
    
    return pss


def psd_distance(obs, sim, x_length, y_length):
    # Get the 2D PSD for observations
    obs_wavelengths, obs_psd = get_2dpsd(obs, x_length, y_length)
    dx = obs_wavelengths[0][1] - obs_wavelengths[0][0]
    dy = obs_wavelengths[1][1] - obs_wavelengths[1][0]

    if len(sim.shape) > len(obs.shape):
        num_ensembles = sim.shape[0]
        psd_distances = []
        
        for i in range(num_ensembles):
            # Calculate the PSD distance for each ensemble member
            _, sim_psd = get_2dpsd(sim[i], x_length, y_length)
            psd_diff = np.abs(sim_psd - obs_psd)
            # Integral approximation
            psd_distance = np.sum(psd_diff) * dx * dy
            psd_distances.append(psd_distance)
        
        # Average over ensemble members
        av_psd_dist = np.mean(psd_distances)
    else:
        # Calculate the PSD distance for a single simulation
        _, sim_psd = get_2dpsd(sim, x_length, y_length)
        psd_diff = np.abs(sim_psd - obs_psd)
        # Integral approximation
        av_psd_dist = np.sum(psd_diff) * dx * dy

    return av_psd_dist


def rank_histogram(obs, sim):
    """
    Calculate the Rank Histogram between two arrays.

    ## Parameters:
    obs (array): Observed values.
    sim (array): Simulated values.

    ## Returns:
    rank_histogram (float): Rank Histogram.
    """
    combined=np.vstack((obs[np.newaxis],sim))

    ranks=np.apply_along_axis(lambda x: rankdata(x,method='min'), 0, combined)

    ties=np.sum(ranks[0]==ranks[1:], axis=0)
    ranks=ranks[0]
    tie=np.unique(ties)

    for i in range(1,len(tie)):
        index=ranks[ties==tie[i]]
        ranks[ties==tie[i]]=[np.random.randint(index[j],index[j]+tie[i]+1,tie[i])[0] for j in range(len(index))]

    return np.histogram(ranks, bins=np.linspace(0.5, combined.shape[0]+0.5, combined.shape[0]+1))


def root_mean_squared_error(obs, sim):
    """
    Calculate the Mean Squared Error (MSE) between two arrays.

    ## Parameters:
    obs (array): Observed values.
    sim (array): Simulated values.

    ## Returns:
    mse (float): Mean Squared Error.
    """
    return np.sqrt(np.mean((obs - sim)**2))


def wasserstein_distance(obs, sim):
    """
    Calculate the Wasserstein distance between two arrays.
    Currently using the properscoring library.
    
    ## Parameters:
    obs (array): Observed values.
    sim (array): Simulated values.
    
    ## Returns:
    wasserstein_distance (float): Wasserstein distance.
    """
    if len(sim.shape) > len(obs.shape):
        num_ensembles = sim.shape[0]
        wass_d = 0.0
        for i in range(num_ensembles):
            wass_d += wass_dist(obs.flatten(), sim[i].flatten())
        wass_d /= num_ensembles
    
    else:
        wass_d = wass_dist(obs.flatten(), sim.flatten())
    
    return wass_d