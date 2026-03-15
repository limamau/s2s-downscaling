import os

import matplotlib.pyplot as plt
import numpy as np
import tomllib

from data.surface_data import (
    ForecastEnsembleSurfaceData,
    ForecastSurfaceData,
    SurfaceData,
)
from engineering.spectrum import get_1dpsd


def plot_psds(cpc, det, ens, lambda_star, psd_star, figs_dir):
    for lead_time_idx in range(1, 4):
        spatial_lenghts = cpc.get_spatial_lengths()
        k, cpc_psd = get_1dpsd(cpc.precip, *spatial_lenghts)
        _, det_psd = get_1dpsd(det.precip[lead_time_idx], *spatial_lenghts)
        _, ens_psd = get_1dpsd(
            np.mean(ens.precip[lead_time_idx], axis=0),
            *spatial_lenghts,
        )
        wavelengths = 2 * np.pi / k[::-1]

        # get nyquist wavelnegths
        nyquist_wavelngths = (
            2 * np.pi / (2 * spatial_lenghts[0]),
            2 * np.pi / (2 * spatial_lenghts[1]),
        )

        # mask wavelengths above nyquist
        mask = wavelengths > max(nyquist_wavelngths)

        fig, ax = plt.subplots(figsize=(8, 5))

        ax.plot(
            wavelengths[mask],
            det_psd[mask][::-1],
            label="IFS det. + NN + low-pass",
            color="C0",
        )
        ax.plot(
            wavelengths[mask],
            ens_psd[mask][::-1],
            label="IFS ens. + NN + low-pass",
            color="C1",
        )
        ax.plot(
            wavelengths[mask],
            cpc_psd[mask][::-1],
            label="CombiPrecip",
            color="C4",
        )
        ax.legend()

        if lambda_star is not None:
            ax.axvline(lambda_star, color="black", linestyle="--")
            ax.text(lambda_star * 1.1, 1e-5, r"$\lambda^\star$", fontsize="large")

        if psd_star is not None:
            ax.axhline(psd_star, color="black", linestyle="--")
            ax.text(1e1, psd_star * 1.1, r"$\sigma^\star$", fontsize="large")

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Wavelengths (km)")
        ax.set_ylabel("Power spectral density")

        ax.set_xlabel("Wavelengths (km)")
        ax.set_ylabel("Power spectral density")
        ax.legend()

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        arrowprops = dict(arrowstyle="-|>", color="black", lw=1.5)

        # X-axis arrow (at the right end)
        ax.annotate(
            "",
            xy=(xlim[1], ylim[0]),  # end of x-axis
            xytext=(xlim[1] * 0.975, ylim[0]),  # start slightly before
            arrowprops=arrowprops,
            annotation_clip=False,
        )

        # Y-axis arrow (at the top end)
        ax.annotate(
            "",
            xy=(xlim[0], ylim[1]),  # top of y-axis
            xytext=(xlim[0], ylim[1] / 1.3),  # start slightly below
            arrowprops=arrowprops,
            annotation_clip=False,
        )
        plt.tight_layout()

        lead_time_name = det.lead_time[lead_time_idx].replace(" ", "-")
        fig.savefig(os.path.join(figs_dir, f"tuning_psd_{lead_time_name}.png"))


def print_target_noise(psd_star, cpc):
    Nx, Ny = cpc.latitude.size, cpc.longitude.size
    sigma_star = np.sqrt(psd_star * Nx * Ny)
    print("Target noise:", sigma_star)


def main():
    # directory paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(script_dir, "../dirs.toml"), "rb") as f:
        dirs = tomllib.load(f)
    base = dirs["main"]["base"]
    test_data_dir = os.path.join(base, dirs["subs"]["test"])

    # extra configuration
    cpc_path = os.path.join(test_data_dir, "cpc.h5")
    det_path = os.path.join(test_data_dir, "det_s2s_nearest_low-pass.h5")
    ens_path = os.path.join(test_data_dir, "ens_s2s_nearest_low-pass.h5")

    cpc = SurfaceData.load_from_h5(cpc_path, ["precip"])
    det = ForecastSurfaceData.load_from_h5(det_path, ["precip"])
    ens = ForecastEnsembleSurfaceData.load_from_h5(ens_path, ["precip"])

    # TODO: normalize this data using config from launched model

    script_dir = os.path.dirname(os.path.abspath(__file__))
    figs_dir = os.path.join(script_dir, "figs/tuning")
    os.makedirs(figs_dir, exist_ok=True)

    # to be tuned
    lambda_star = 2.3e2
    psd_star = 3.2e-2

    # main calls
    plot_psds(cpc, det, ens, lambda_star, psd_star, figs_dir)
    print("psds saved")
    print_target_noise(psd_star, cpc)


if __name__ == "__main__":
    main()
