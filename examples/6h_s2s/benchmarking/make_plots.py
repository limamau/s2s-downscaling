import os

import imageio
import matplotlib.pyplot as plt
import numpy as np
import tomllib

from data.surface_data import (
    ForecastEnsembleSurfaceData,
    ForecastSurfaceData,
    SurfaceData,
)
from engineering.spectrum import get_1dpsd
from evaluation.plots import CURVE_CMAP as cmap
from evaluation.plots import plot_maps
from utils import get_cdf

EVENT_LENGTH = 8


### auxiliary functions ###
def plot_left_tale_det(
    bins_range,
    det_s2s_cdf,
    ens_s2s_cdf,
    det_diff_cdf,
    ens_diff_cdf,
    cpc_cdf,
    lead_time_name,
    figs_dir,
):
    fig, ax = plt.subplots(figsize=(8, 4))

    ax.plot(bins_range, det_s2s_cdf, label="S2S det.", color=cmap(1), linewidth=2)
    ax.plot(bins_range, ens_s2s_cdf, label="S2S ens.", color=cmap(2), linewidth=2)
    ax.plot(bins_range, det_diff_cdf, label="det-diff", color=cmap(5), linewidth=2)
    ax.plot(bins_range, ens_diff_cdf, label="ens-diff", color=cmap(6), linewidth=2)
    ax.plot(bins_range, cpc_cdf, label="CombiPrecip", color=cmap(0), linewidth=2)

    ax.set_xlim(0, 5)
    ax.set_ylim(0, 1.0)
    ax.set_xlabel("Precipitation (mm/h)")
    ax.set_ylabel("Cumulative distribution function")

    ax.legend()
    plt.title(f"lead time = {lead_time_name}")
    fig.savefig(os.path.join(figs_dir, f"dist_{lead_time_name}_left.png"))


def plot_right_tale_ens(
    bins_range,
    det_s2s_cdf,
    ens_s2s_cdf,
    det_diff_cdf,
    ens_diff_cdf,
    wrf_cdf,
    cpc_cdf,
    lead_time_name,
    figs_dir,
):
    fig, ax = plt.subplots(figsize=(8, 5))

    # ensemble mean and spread
    ens_s2s_mean = np.mean(ens_s2s_cdf, axis=0)
    det_diff_mean = np.mean(det_diff_cdf, axis=0)
    ens_diff_mean = np.mean(ens_diff_cdf, axis=0)
    wrf_mean = np.mean(wrf_cdf, axis=0)
    # use std for spread
    ens_s2s_std = np.std(ens_s2s_cdf, axis=0)
    det_diff_std = np.std(det_diff_cdf, axis=0)
    ens_diff_std = np.std(ens_diff_cdf, axis=0)
    wrf_std = np.std(wrf_cdf, axis=0)
    # get bounds
    ens_s2s_lower_bound = np.maximum(ens_s2s_mean - ens_s2s_std, 0)
    ens_s2s_upper_bound = np.maximum(ens_s2s_mean + ens_s2s_std, 0)
    det_diff_lower_bound = np.maximum(det_diff_mean - det_diff_std, 0)
    det_diff_upper_bound = np.maximum(det_diff_mean + det_diff_std, 0)
    ens_diff_lower_bound = np.maximum(ens_diff_mean - ens_diff_std, 0)
    ens_diff_upper_bound = np.maximum(ens_diff_mean + ens_diff_std, 0)
    wrf_lower_bound = np.maximum(wrf_mean - wrf_std, 0)
    wrf_upper_bound = np.maximum(wrf_mean + wrf_std, 0)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(0.3, None)
    ax.set_ylim(0.8, 1.0)
    ax.set_xlabel("Precipitation (mm/h)")
    ax.set_ylabel("Cumulative distribution function")

    ax.plot(bins_range, det_s2s_cdf, label="S2S det.", color=cmap(1), linewidth=2)
    ax.fill_between(
        bins_range,
        ens_s2s_lower_bound,
        ens_s2s_upper_bound,
        color=cmap(2),
        alpha=0.3,
        label="S2S ens.",
    )
    ax.fill_between(
        bins_range,
        det_diff_lower_bound,
        det_diff_upper_bound,
        color=cmap(5),
        alpha=0.3,
        label="det-diff",
    )
    ax.fill_between(
        bins_range,
        ens_diff_lower_bound,
        ens_diff_upper_bound,
        color=cmap(6),
        alpha=0.3,
        label="ens-diff",
    )
    ax.fill_between(
        bins_range,
        wrf_lower_bound,
        wrf_upper_bound,
        color=cmap(3),
        alpha=0.3,
        label="WRF",
    )
    ax.plot(bins_range, cpc_cdf, label="CombiPrecip", color=cmap(0), linewidth=2)

    ax.legend()
    plt.title(f"lead time = {lead_time_name}")
    fig.savefig(os.path.join(figs_dir, f"dist_{lead_time_name}_right.png"))


def plot_left_tale_ens(
    bins_range,
    det_s2s_cdf,
    ens_s2s_cdf,
    det_diff_cdf,
    ens_diff_cdf,
    wrf_cdf,
    cpc_cdf,
    lead_time_name,
    figs_dir,
):
    fig, ax = plt.subplots(figsize=(8, 4))

    # ensemble mean and spread
    ens_s2s_mean = np.mean(ens_s2s_cdf, axis=0)
    det_diff_mean = np.mean(det_diff_cdf, axis=0)
    ens_diff_mean = np.mean(ens_diff_cdf, axis=0)
    wrf_mean = np.mean(wrf_cdf, axis=0)
    # use std for spread
    ens_s2s_std = np.std(ens_s2s_cdf, axis=0)
    det_diff_std = np.std(det_diff_cdf, axis=0)
    ens_diff_std = np.std(ens_diff_cdf, axis=0)
    wrf_std = np.std(wrf_cdf, axis=0)
    # get bounds
    ens_s2s_lower_bound = np.maximum(ens_s2s_mean - ens_s2s_std, 0)
    ens_s2s_upper_bound = np.maximum(ens_s2s_mean + ens_s2s_std, 0)
    det_diff_lower_bound = np.maximum(det_diff_mean - det_diff_std, 0)
    det_diff_upper_bound = np.maximum(det_diff_mean + det_diff_std, 0)
    ens_diff_lower_bound = np.maximum(ens_diff_mean - ens_diff_std, 0)
    ens_diff_upper_bound = np.maximum(ens_diff_mean + ens_diff_std, 0)
    wrf_lower_bound = np.maximum(wrf_mean - wrf_std, 0)
    wrf_upper_bound = np.maximum(wrf_mean + wrf_std, 0)

    ax.plot(bins_range, det_s2s_cdf, label="S2S det.", color=cmap(1), linewidth=2)
    ax.fill_between(
        bins_range,
        ens_s2s_lower_bound,
        ens_s2s_upper_bound,
        color=cmap(2),
        alpha=0.3,
        label="S2S ens.",
    )
    ax.fill_between(
        bins_range,
        det_diff_lower_bound,
        det_diff_upper_bound,
        color=cmap(5),
        alpha=0.3,
        label="det. diff.",
    )
    ax.fill_between(
        bins_range,
        ens_diff_lower_bound,
        ens_diff_upper_bound,
        color=cmap(6),
        alpha=0.3,
        label="ens. diff.",
    )
    ax.fill_between(
        bins_range,
        wrf_lower_bound,
        wrf_upper_bound,
        color=cmap(3),
        alpha=0.3,
        label="WRF",
    )
    ax.plot(bins_range, cpc_cdf, label="CombiPrecip", color=cmap(0), linewidth=2)

    ax.set_xlim(0, 5)
    ax.set_ylim(0, 1.0)
    ax.set_xlabel("Precipitation (mm/h)")
    ax.set_ylabel("Cumulative distribution function")

    ax.legend()
    plt.title(f"lead time = {lead_time_name}")
    fig.savefig(os.path.join(figs_dir, f"dist_{lead_time_name}_left.png"))


def plot_lead_time_distribution(
    det_s2s,
    ens_s2s,
    det_diff,
    ens_diff,
    wrf,
    cpc,
    time_idxs,
    lead_time_idx,
    figs_dir,
    bins=100,
):
    # get data
    det_s2s_data = det_s2s.precip[lead_time_idx, time_idxs]
    ens_s2s_data = ens_s2s.precip[lead_time_idx, :, time_idxs]
    det_diff_data = det_diff.precip[lead_time_idx, :, time_idxs]
    ens_diff_data = ens_diff.precip[lead_time_idx, :, time_idxs]
    wrf_data = wrf.precip[lead_time_idx, :, time_idxs]
    cpc_data = cpc.precip[time_idxs].flatten()

    bins_range = np.linspace(0, 5, bins)

    def get_cdf_ens(data):
        num_ensembles = data.shape[0]
        cdf_ens = np.zeros((num_ensembles, bins))
        for i in range(num_ensembles):
            cdf_ens[i, :] = get_cdf(data[i, ...], bins_range)
        return cdf_ens

    # compute CDFs
    det_s2s_cdf = get_cdf(det_s2s_data, bins_range)
    ens_s2s_cdf = get_cdf_ens(ens_s2s_data)
    det_diff_cdf = get_cdf_ens(det_diff_data)
    ens_diff_cdf = get_cdf_ens(ens_diff_data)
    wrf_cdf = get_cdf_ens(wrf_data)
    cpc_cdf = get_cdf(cpc_data, bins_range)

    # call left and right tale plots
    lead_time_name = det_s2s.lead_time[lead_time_idx]
    plot_left_tale_ens(
        bins_range,
        det_s2s_cdf,
        ens_s2s_cdf,
        det_diff_cdf,
        ens_diff_cdf,
        wrf_cdf,
        cpc_cdf,
        lead_time_name,
        figs_dir,
    )
    plot_right_tale_ens(
        bins_range,
        det_s2s_cdf,
        ens_s2s_cdf,
        det_diff_cdf,
        ens_diff_cdf,
        wrf_cdf,
        cpc_cdf,
        lead_time_name,
        figs_dir,
    )


def plot_lead_time_map(
    det_s2s,
    ens_s2s,
    det_diff,
    ens_diff,
    wrf,
    cpc,
    lead_time_idx,
    num_idx,
    time_idxs,
    figs_dir,
):
    lead_time_name = det_s2s.lead_time[lead_time_idx]
    print(f"lead time = {lead_time_name}")
    image_paths = []

    # save temporary images for each time_idx
    for time_idx in time_idxs:
        arrays = (
            det_s2s.precip[lead_time_idx, time_idx],
            ens_s2s.precip[lead_time_idx, num_idx, time_idx],
            det_diff.precip[lead_time_idx, num_idx, time_idx],
            ens_diff.precip[lead_time_idx, num_idx, time_idx],
            wrf.precip[lead_time_idx, num_idx, time_idx],
            cpc.precip[time_idx],
        )
        titles = (
            "S2S ens.",
            "ens. s2s.",
            "det .diff.",
            "ens. diff.",
            "WRF",
            "CombiPrecip",
        )
        cpc_extent = cpc.get_extent()
        extents = (cpc_extent,) * 6
        fig, _ = plot_maps(arrays, titles, extents)

        image_path = os.path.join(
            figs_dir, f"temp_map_{lead_time_name}_t{time_idx}.png"
        )
        fig.savefig(image_path)
        plt.close(fig)  # important to avoid memory leak
        image_paths.append(image_path)

    # create the GIFs
    for event_idx in range(1, len(time_idxs) // EVENT_LENGTH + 1):
        gif_path = os.path.join(figs_dir, f"maps_{lead_time_name}_e{event_idx}.gif")
        with imageio.get_writer(gif_path, mode="I", duration=1000) as writer:
            for image_path_idx in range(
                (event_idx - 1) * EVENT_LENGTH, event_idx * EVENT_LENGTH
            ):
                image_path = image_paths[image_path_idx]
                image = imageio.v2.imread(image_path)
                writer.append_data(image)

    # clean up temporary files
    for image_path in image_paths:
        os.remove(image_path)


def plot_lead_time_timeseries(
    det_s2s,
    ens_s2s,
    det_diff,
    ens_diff,
    wrf,
    cpc,
    time_idxs,
    lead_time_idx,
    figs_dir,
):
    # get timeseries
    det_s2s_timeseries = np.mean(
        det_s2s.precip[lead_time_idx, time_idxs],
        axis=(-2, -1),
    )
    ens_s2s_timeseries = np.mean(
        ens_s2s.precip[lead_time_idx, :, time_idxs],
        axis=(-2, -1),
    )
    det_diff_timeseries = np.mean(
        det_diff.precip[lead_time_idx, :, time_idxs],
        axis=(-2, -1),
    )
    ens_diff_timeseries = np.mean(
        ens_diff.precip[lead_time_idx, :, time_idxs],
        axis=(-2, -1),
    )
    wrf_timeseries = np.mean(
        wrf.precip[lead_time_idx, :, time_idxs],
        axis=(-2, -1),
    )
    cpc_timeseries = np.mean(
        cpc.precip[time_idxs],
        axis=(-2, -1),
    )
    dates = cpc.time[time_idxs]

    # compute ensemble mean and spread
    ens_s2s = np.mean(ens_s2s_timeseries, axis=1)
    det_diff_mean = np.mean(det_diff_timeseries, axis=1)
    ens_diff_mean = np.mean(ens_diff_timeseries, axis=1)
    wrf_diff_mean = np.mean(wrf_timeseries, axis=1)
    # use std for spread
    ens_s2s_std = np.std(ens_s2s_timeseries, axis=1)
    det_diff_std = np.std(det_diff_timeseries, axis=1)
    ens_diff_std = np.std(ens_diff_timeseries, axis=1)
    wrf_diff_std = np.std(wrf_timeseries, axis=1)
    # get bounds
    ens_s2s_lower_bound = np.maximum(ens_s2s - ens_s2s_std, 0)
    ens_s2s_upper_bound = np.maximum(ens_s2s + ens_s2s_std, 0)
    det_diff_lower_bound = np.maximum(det_diff_mean - det_diff_std, 0)
    det_diff_upper_bound = np.maximum(det_diff_mean + det_diff_std, 0)
    ens_diff_lower_bound = np.maximum(ens_diff_mean - ens_diff_std, 0)
    ens_diff_upper_bound = np.maximum(ens_diff_mean + ens_diff_std, 0)
    wrf_diff_lower_bound = np.maximum(wrf_diff_mean - wrf_diff_std, 0)
    wrf_diff_upper_bound = np.maximum(wrf_diff_mean + wrf_diff_std, 0)

    # plot timeseries for each event
    for event_idx in range(1, len(time_idxs) // EVENT_LENGTH + 1):
        idxs = slice(EVENT_LENGTH * (event_idx - 1), EVENT_LENGTH * event_idx)

        # create figure and axis
        fig, ax = plt.subplots(figsize=(8, 4))

        # beginning
        ax.plot(dates[idxs], det_s2s_timeseries[idxs], label="S2S det.", color=cmap(1))

        # add shaded region for ensemble spread
        ax.fill_between(
            dates[idxs],
            ens_s2s_lower_bound[idxs],
            ens_s2s_upper_bound[idxs],
            color=cmap(2),
            alpha=0.3,
            label="S2S ens.",
        )
        ax.fill_between(
            dates[idxs],
            det_diff_lower_bound[idxs],
            det_diff_upper_bound[idxs],
            color=cmap(5),
            alpha=0.3,
            label="det. diff.",
        )
        ax.fill_between(
            dates[idxs],
            ens_diff_lower_bound[idxs],
            ens_diff_upper_bound[idxs],
            color=cmap(6),
            alpha=0.3,
            label="ens. diff.",
        )
        ax.fill_between(
            dates[idxs],
            wrf_diff_lower_bound[idxs],
            wrf_diff_upper_bound[idxs],
            color=cmap(3),
            alpha=0.3,
            label="WRF",
        )

        # end
        ax.plot(dates[idxs], cpc_timeseries[idxs], label="CombiPrecip", color=cmap(0))

        # add legend and save
        ax.set_xlabel("Dates")
        ax.set_ylabel("Mean precipitation (mm/h)")
        plt.legend()

        lead_time_name = det_s2s.lead_time[lead_time_idx]
        plt.title(f"lead time = {lead_time_name}, event = {event_idx}")
        fig.savefig(os.path.join(figs_dir, f"ts_{lead_time_name}_e{event_idx}.png"))


def plot_lead_time_psd(
    det_s2s,
    ens_s2s,
    det_diff,
    ens_diff,
    wrf,
    cpc,
    time_idxs,
    lead_time_idx,
    figs_dir,
):
    spatial_lenghts = det_s2s.get_spatial_lengths()
    k, det_s2s_psd = get_1dpsd(
        det_s2s.precip[lead_time_idx, time_idxs], *spatial_lenghts
    )
    _, ens_s2s_psd = get_1dpsd(
        np.mean(ens_s2s.precip[lead_time_idx, :, time_idxs], axis=0),
        *spatial_lenghts,
    )
    _, det_diff_psd = get_1dpsd(
        np.mean(det_diff.precip[lead_time_idx, :, time_idxs], axis=0),
        *spatial_lenghts,
    )
    _, ens_diff_psd = get_1dpsd(
        np.mean(ens_diff.precip[lead_time_idx, :, time_idxs], axis=0),
        *spatial_lenghts,
    )
    _, wrf_psd = get_1dpsd(
        np.mean(ens_diff.precip[lead_time_idx, :, time_idxs], axis=0),
        *spatial_lenghts,
    )
    _, cpc_psd = get_1dpsd(cpc.precip[time_idxs], *spatial_lenghts)
    wavelengths = 2 * np.pi / k[::-1]

    # get nyquist wavelnegths
    nyquist_wavelngths = (
        2 * np.pi / (2 * spatial_lenghts[0]),
        2 * np.pi / (2 * spatial_lenghts[1]),
    )

    # mask wavelengths above nyquist
    mask = wavelengths > max(nyquist_wavelngths)

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(wavelengths[mask], det_s2s_psd[mask][::-1], label="S2S det.", color=cmap(1))
    ax.plot(wavelengths[mask], ens_s2s_psd[mask][::-1], label="S2S ens.", color=cmap(2))
    ax.plot(
        wavelengths[mask], det_diff_psd[mask][::-1], label="det. diff.", color=cmap(5)
    )
    ax.plot(
        wavelengths[mask], ens_diff_psd[mask][::-1], label="ens. diff.", color=cmap(6)
    )
    ax.plot(wavelengths[mask], wrf_psd[mask][::-1], label="WRF", color=cmap(3))
    ax.plot(wavelengths[mask], cpc_psd[mask][::-1], label="CombiPrecip", color=cmap(0))

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Wavelengths (km)")
    ax.set_ylabel("Power spectral density")
    ax.legend()

    lead_time_name = det_s2s.lead_time[lead_time_idx]
    plt.title(f"lead time = {lead_time_name}")
    fig.savefig(os.path.join(figs_dir, f"psd_{lead_time_name}.png"))


def make_plots(
    s2s_det_path,
    s2s_ens_path,
    diff_det_path,
    diff_ens_path,
    wrf_path,
    cpc_path,
    time_idxs,
    num_idx,
):
    det_s2s = ForecastSurfaceData.load_from_h5(s2s_det_path, ["precip"])
    ens_s2s = ForecastEnsembleSurfaceData.load_from_h5(s2s_ens_path, ["precip"])
    det_diff = ForecastEnsembleSurfaceData.load_from_h5(diff_det_path, ["precip"])
    ens_diff = ForecastEnsembleSurfaceData.load_from_h5(diff_ens_path, ["precip"])
    wrf = ForecastEnsembleSurfaceData.load_from_h5(wrf_path, ["precip"])
    cpc = SurfaceData.load_from_h5(cpc_path, ["precip"])

    # define figs directory
    script_dir = os.path.dirname(os.path.realpath(__file__))
    figs_dir = os.path.join(script_dir, "figs/analysis")
    os.makedirs(figs_dir, exist_ok=True)

    # plot gifs for each lead time (and each event)
    for lead_time_idx in range(3):
        plot_lead_time_map(
            det_s2s,
            ens_s2s,
            det_diff,
            ens_diff,
            wrf,
            cpc,
            lead_time_idx,
            num_idx,
            time_idxs,
            figs_dir,
        )
    print("maps saved")

    # plot timeseries for each lead time (and each event)
    for lead_time_idx in range(3):
        plot_lead_time_timeseries(
            det_s2s,
            ens_s2s,
            det_diff,
            ens_diff,
            wrf,
            cpc,
            time_idxs,
            lead_time_idx,
            figs_dir,
        )
    print("timeseries saved")

    # plot distribution for each lead time
    for lead_time_idx in range(3):
        plot_lead_time_distribution(
            det_s2s,
            ens_s2s,
            det_diff,
            ens_diff,
            wrf,
            cpc,
            time_idxs,
            lead_time_idx,
            figs_dir,
        )
    print("distributions saved")

    # plot psds for each lead time
    for lead_time_idx in range(3):
        plot_lead_time_psd(
            det_s2s,
            ens_s2s,
            det_diff,
            ens_diff,
            wrf,
            cpc,
            time_idxs,
            lead_time_idx,
            figs_dir,
        )
    print("psds saved")


def main():
    # directory paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(script_dir, "../dirs.toml"), "rb") as f:
        dirs = tomllib.load(f)
    base = dirs["main"]["base"]
    test_data_dir = os.path.join(base, dirs["subs"]["test"])
    simulations_dir = os.path.join(base, dirs["subs"]["simulations"])

    # extra configurations
    s2s_det_path = os.path.join(test_data_dir, "det_s2s_nearest.h5")
    s2s_ens_path = os.path.join(test_data_dir, "ens_s2s_nearest.h5")
    cli = 50
    diff_det_path = os.path.join(
        simulations_dir,
        "diffusion",
        f"det_heavy_cli{cli}_ens50.h5",
    )
    diff_ens_path = os.path.join(
        simulations_dir,
        "diffusion",
        f"ens_heavy_cli{cli}_ens50.h5",
    )
    wrf_path = os.path.join(
        simulations_dir,
        "wrf",
        "wrf.h5",
    )
    cpc_path = os.path.join(test_data_dir, "cpc.h5")
    # time_idx for snapshots
    # limamau: back to 16 when the two events will be available
    time_idxs = [i for i in range(8)]
    # ensemble member for snapshots
    num_idx = 0

    # main calls
    make_plots(
        s2s_det_path,
        s2s_ens_path,
        diff_det_path,
        diff_ens_path,
        wrf_path,
        cpc_path,
        time_idxs,
        num_idx,
    )


if __name__ == "__main__":
    main()
