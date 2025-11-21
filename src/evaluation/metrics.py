import numpy as np
from scipy.ndimage import uniform_filter
from scipy.stats import rankdata
from scipy.stats import wasserstein_distance as wass_dist

from engineering.spectrum import get_2dpsd
from utils import get_cdf, get_pdf

# Metric functions in almost alphabetical order:


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
    sim = np.asarray(sim)
    obs = np.asarray(obs)

    if obs.shape == sim.shape:
        sim = np.expand_dims(sim, axis=0)

    if obs.shape != sim.shape[1:]:
        print("obs shape:", obs.shape)
        print("sim shape:", sim.shape)
        raise ValueError("Observation shape must match simulation shape.")

    # term1: mean absolute error between ensemble members and obs
    term1 = np.mean(np.abs(sim - obs), axis=0)

    # term2: mean pairwise absolute differences between ensemble members
    # Instead of creating the full diff array, use broadcasting and sum
    n = sim.shape[0]
    term2 = 0
    for i in range(n):
        for j in range(i + 1, n):
            term2 += np.abs(sim[i] - sim[j])
    term2 = term2 * 2 / (n * n)  # since we only sum i < j, multiply by 2
    crps_field = term1 - 0.5 * term2

    return np.mean(crps_field)


def mean_absolute_error(obs, sim, axis=(0, 1, 2)):
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
    combined = np.vstack((obs[np.newaxis], sim))

    ranks = np.apply_along_axis(lambda x: rankdata(x, method="min"), 0, combined)

    ties = np.sum(ranks[0] == ranks[1:], axis=0)
    ranks = ranks[0]
    tie = np.unique(ties)

    for i in range(1, len(tie)):
        index = ranks[ties == tie[i]]
        ranks[ties == tie[i]] = [
            np.random.randint(index[j], index[j] + tie[i] + 1, tie[i])[0]
            for j in range(len(index))
        ]

    return np.histogram(
        ranks, bins=np.linspace(0.5, combined.shape[0] + 0.5, combined.shape[0] + 1)
    )


def root_mean_squared_error(obs, sim):
    """
    Calculate the Mean Squared Error (MSE) between two arrays.

    ## Parameters:
    obs (array): Observed values.
    sim (array): Simulated values.

    ## Returns:
    mse (float): Mean Squared Error.
    """
    return np.sqrt(np.mean((obs - sim) ** 2))


def wasserstein_distance(obs, sim):
    """
    Calculate the Wasserstein distance between two arrays.

    ## Parameters:
    obs (array): Observed values.
    sim (array): Simulated values.

    ## Returns:
    wasserstein_distance (float): Wasserstein distance.
    """
    return wass_dist(obs.flatten(), sim.flatten())


# the following was copied from pysteps in
# https://github.com/pySTEPS/pysteps/blob/edd9be5c8124613082b359c451136b7d4e452815/pysteps/verification/spatialscores.py#L516
# because I couldn't add it as a package due to some weird error
def fss(X_f, X_o, thr, scale):
    """
    Compute the fractions skill score (FSS) for a deterministic forecast field
    and the corresponding observation field.

    Parameters
    ----------
    X_f: array_like
        Array of shape (m, n) containing the forecast field.
    X_o: array_like
        Array of shape (m, n) containing the observation field.
    thr: float
        The intensity threshold.
    scale: int
        The spatial scale in pixels. In practice, the scale represents the size
        of the moving window that it is used to compute the fraction of pixels
        above the threshold.

    Returns
    -------
    out: float
        The fractions skill score between 0 and 1.
    """

    fss = _fss_init(thr, scale)
    _fss_accum(fss, X_f, X_o)
    return _fss_compute(fss)


def _fss_init(thr, scale):
    """
    Initialize a fractions skill score (FSS) verification object.

    Parameters
    ----------
    thr: float
        The intensity threshold.
    scale: float
        The spatial scale in pixels. In practice, the scale represents the size
        of the moving window that it is used to compute the fraction of pixels
        above the threshold.

    Returns
    -------
    fss: dict
        The initialized FSS verification object.
    """
    fss = dict(thr=thr, scale=scale, sum_fct_sq=0.0, sum_fct_obs=0.0, sum_obs_sq=0.0)

    return fss


def _fss_accum(fss, X_f, X_o):
    """Accumulate forecast-observation pairs to an FSS object.

    Parameters
    -----------
    fss: dict
        The FSS object initialized with
        :py:func:`pysteps.verification.spatialscores.fss_init`.
    X_f: array_like
        Array of shape (m, n) containing the forecast field.
    X_o: array_like
        Array of shape (m, n) containing the observation field.
    """
    if len(X_f.shape) != 2 or len(X_o.shape) != 2 or X_f.shape != X_o.shape:
        message = "X_f and X_o must be two-dimensional arrays"
        message += " having the same shape"
        raise ValueError(message)

    X_f = X_f.copy()
    X_f[~np.isfinite(X_f)] = fss["thr"] - 1
    X_o = X_o.copy()
    X_o[~np.isfinite(X_o)] = fss["thr"] - 1

    # Convert to binary fields with the given intensity threshold
    I_f = (X_f >= fss["thr"]).astype(float)
    I_o = (X_o >= fss["thr"]).astype(float)

    # Compute fractions of pixels above the threshold within a square
    # neighboring area by applying a 2D moving average to the binary fields
    if fss["scale"] > 1:
        S_f = uniform_filter(I_f, size=fss["scale"], mode="constant", cval=0.0)
        S_o = uniform_filter(I_o, size=fss["scale"], mode="constant", cval=0.0)
    else:
        S_f = I_f
        S_o = I_o

    fss["sum_obs_sq"] += np.nansum(S_o**2)
    fss["sum_fct_obs"] += np.nansum(S_f * S_o)
    fss["sum_fct_sq"] += np.nansum(S_f**2)


def _fss_merge(fss_1, fss_2):
    """
    Merge two FSS objects.

    Parameters
    ----------
    fss_1: dict
      A FSS object initialized with
      :py:func:`pysteps.verification.spatialscores.fss_init`.
      and populated with
      :py:func:`pysteps.verification.spatialscores.fss_accum`.
    fss_2: dict
      Another FSS object initialized with
      :py:func:`pysteps.verification.spatialscores.fss_init`.
      and populated with
      :py:func:`pysteps.verification.spatialscores.fss_accum`.

    Returns
    -------
    out: dict
      The merged FSS object.
    """

    # checks
    if fss_1["thr"] != fss_2["thr"]:
        raise ValueError(
            "cannot merge: the thresholds are not same %s!=%s"
            % (fss_1["thr"], fss_2["thr"])
        )
    if fss_1["scale"] != fss_2["scale"]:
        raise ValueError(
            "cannot merge: the scales are not same %s!=%s"
            % (fss_1["scale"], fss_2["scale"])
        )

    # merge the FSS objects
    fss = fss_1.copy()
    fss["sum_obs_sq"] += fss_2["sum_obs_sq"]
    fss["sum_fct_obs"] += fss_2["sum_fct_obs"]
    fss["sum_fct_sq"] += fss_2["sum_fct_sq"]

    return fss


def _fss_compute(fss):
    """
    Compute the FSS.

    Parameters
    ----------
    fss: dict
       An FSS object initialized with
       :py:func:`pysteps.verification.spatialscores.fss_init`
       and accumulated with
       :py:func:`pysteps.verification.spatialscores.fss_accum`.

    Returns
    -------
    out: float
        The computed FSS value.
    """
    numer = fss["sum_fct_sq"] - 2.0 * fss["sum_fct_obs"] + fss["sum_obs_sq"]
    denom = fss["sum_fct_sq"] + fss["sum_obs_sq"]

    return 1.0 - numer / denom
