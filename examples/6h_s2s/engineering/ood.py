import os

import matplotlib.pyplot as plt
import numpy as np
import tomllib

from data.surface_data import SurfaceData

EVENT_TIME_LENGTH = 8


def interpolate_1km(arr, factor=15):
    """
    Coarsen a 1 km-resolution array to 'factor' km by block averaging.

    Parameters
    ----------
    arr : np.ndarray
        Input array of shape (time, lat, lon).
    factor : int, optional
        Coarsening factor (default 15). The output grid cell covers
        factor x factor original pixels.

    Returns
    -------
    np.ndarray
        Coarsened array of shape (time, lat//factor, lon//factor).
        Values are the mean of all non‑NaN pixels within each block.
        If a block contains only NaNs, the output is NaN.
    """
    T, H, W = arr.shape

    # Crop to the largest dimensions divisible by factor
    H_new = H - (H % factor)
    W_new = W - (W % factor)
    arr_cropped = arr[:, :H_new, :W_new]

    # Reshape to separate block dimensions:
    # (T, H_new//factor, factor, W_new//factor, factor)
    reshaped = arr_cropped.reshape(T, H_new // factor, factor, W_new // factor, factor)

    # Average over the block axes (2 and 4)
    # np.nanmean ignores NaNs; warnings for all‑NaN slices are suppressed
    with np.errstate(invalid="ignore"):
        coarsened = np.nanmean(reshaped, axis=(2, 4))

    return coarsened


def main():
    # directory paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(script_dir, "../dirs.toml"), "rb") as f:
        dirs = tomllib.load(f)
    base = dirs["main"]["base"]
    train_data_dir = os.path.join(base, dirs["subs"]["train"])
    val_data_dir = os.path.join(base, dirs["subs"]["validation"])
    test_data_dir = os.path.join(base, dirs["subs"]["test"])

    # file paths
    train_cpc_path = os.path.join(train_data_dir, "cpc.h5")
    val_cpc_path = os.path.join(val_data_dir, "cpc.h5")
    test_cpc_path = os.path.join(test_data_dir, "cpc.h5")

    # Load data (precipitation only)
    cpc_train = SurfaceData.load_from_h5(train_cpc_path, ["precip"])
    cpc_val = SurfaceData.load_from_h5(val_cpc_path, ["precip"])
    cpc_test = SurfaceData.load_from_h5(test_cpc_path, ["precip"])

    # Extract the precipitation arrays (assumes shape: time, lat, lon)
    # Adjust attribute name if needed (e.g., .precip, .data, .values)
    train_precip = cpc_train.precip  # or cpc_train.data
    val_precip = cpc_val.precip
    test_precip = cpc_test.precip

    # Combine train and validation
    train_val_precip = np.concatenate([train_precip, val_precip], axis=0)

    # Verify test time length
    print(f"test precip shape: {test_precip.shape}")
    assert test_precip.shape[0] == 2 * EVENT_TIME_LENGTH, (
        f"Expected test time length {2 * EVENT_TIME_LENGTH}, got {test_precip.shape[0]}"
    )

    # Split test into 2018 event (first EVENT_TIME_LENGTH steps) and 2021 event (remaining)
    test_2018 = test_precip[:EVENT_TIME_LENGTH, ...]
    test_2021 = test_precip[EVENT_TIME_LENGTH:, ...]

    # Coarsen from 1 km to 15 km
    train_val_precip = interpolate_1km(train_val_precip, factor=15)
    test_2018 = interpolate_1km(test_2018, factor=15)
    test_2021 = interpolate_1km(test_2021, factor=15)

    # Flatten all arrays for histogram (ignore NaN values)
    train_val_flat = train_val_precip.flatten()
    train_val_flat = train_val_flat[~np.isnan(train_val_flat)]

    test_2018_flat = test_2018.flatten()
    test_2018_flat = test_2018_flat[~np.isnan(test_2018_flat)]

    test_2021_flat = test_2021.flatten()
    test_2021_flat = test_2021_flat[~np.isnan(test_2021_flat)]

    # interpolation

    # filter for > 0.1mm/h
    train_val_flat = np.log(train_val_flat[train_val_flat > 0.1])
    test_2018_flat = np.log(test_2018_flat[test_2018_flat > 0.1])
    test_2021_flat = np.log(test_2021_flat[test_2021_flat > 0.1])

    # Plot histograms
    fig, ax = plt.subplots(1, 1, figsize=(5, 5), sharey=True)

    ax.hist(
        train_val_flat,
        bins=50,
        color="blue",
        alpha=0.5,
        density=True,
        label="Train + Validation",
    )
    ax.set_title("CombiPrecip 15km agg.")
    ax.set_xlabel("Log precipitation (mm/h)")
    ax.set_ylabel("Density")

    ax.hist(
        test_2018_flat,
        bins=50,
        color="green",
        alpha=0.5,
        density=True,
        label="2018 event",
    )

    ax.hist(
        test_2021_flat,
        bins=50,
        color="red",
        alpha=0.5,
        density=True,
        label="2021 event",
    )

    plt.tight_layout()
    plt.legend()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    figs_dir = os.path.join(script_dir, "figs")
    os.makedirs(figs_dir, exist_ok=True)
    plt.savefig(os.path.join(figs_dir, "ood.png"), dpi=150)


if __name__ == "__main__":
    main()
