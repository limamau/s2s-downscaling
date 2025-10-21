import os

import imageio
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import tomllib
from matplotlib.ticker import MaxNLocator

from data.surface_data import (
    ForecastEnsembleSurfaceData,
    ForecastSurfaceData,
    SurfaceData,
)
from engineering.spectrum import get_1dpsd
from evaluation.metrics import fss

# from evaluation.plots import CURVE_CMAP as cmap
from evaluation.plots import plot_maps
from utils import get_cdf

EVENT_LENGTH = 8
NUMBER_OF_EVENTS = 2
MODEL_COLOR_DICT = {
    "S2S det.": "C0",
    "S2S ens.": "C1",
    "Diff. det.": "C0",
    "Diff. ens.": "C1",
    "WRF": "C2",
    "CombiPrecip": "C3",
}
MODEL_LINESTYLE = {
    "S2S det.": "--",
    "S2S ens.": "--",
    "Diff. det.": "-",
    "Diff. ens.": "-",
    "WRF": "-",
    "CombiPrecip": "-",
}
TIME_IDXS = [i for i in range(EVENT_LENGTH * NUMBER_OF_EVENTS)]


### auxiliary functions ###
def _make_arrows(ax, xpos, ypos):
    # make arrows
    ax.plot(
        (1),
        (0),
        ls="",
        marker=">",
        ms=5,
        color="k",
        transform=ax.get_yaxis_transform(),
        clip_on=False,
    )
    ax.plot(
        (xpos),
        (ypos),
        ls="",
        marker="^",
        ms=5,
        color="k",
        transform=ax.get_xaxis_transform(),
        clip_on=False,
    )
    # clean axis
    ax.spines["left"].set_position("zero")
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_position("zero")
    ax.spines["top"].set_visible(False)
    ax.xaxis.set_ticks_position("bottom")
    ax.yaxis.set_ticks_position("left")


def rank_histogram(ens, obs):
    n_ens = ens.shape[0]

    # flatten space and time so we treat each point independently
    ens_flat = ens.reshape(n_ens, -1)  # (n_ens, n_samples)
    obs_flat = obs.reshape(-1)  # (n_samples,)

    # sort ensemble values along ensemble axis
    sorted_ens = jnp.sort(ens_flat, axis=0)

    # compare observation to sorted ensemble members
    # this yields a boolean array: obs > sorted member?
    comparisons = obs_flat[None, :] > sorted_ens  # shape (n_ens, n_samples)

    # sum over ensemble axis → gives rank (0..n_ens)
    ranks = jnp.sum(comparisons, axis=0)

    return ranks


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

    # plots
    ax.plot(
        bins_range,
        det_s2s_cdf,
        label="S2S det.",
        color=MODEL_COLOR_DICT["S2S det."],
        linewidth=2,
    )
    ax.fill_between(
        bins_range,
        ens_s2s_lower_bound,
        ens_s2s_upper_bound,
        color=MODEL_COLOR_DICT["S2S ens."],
        alpha=0.3,
        label="S2S ens.",
    )
    ax.fill_between(
        bins_range,
        det_diff_lower_bound,
        det_diff_upper_bound,
        color=MODEL_COLOR_DICT["Diff. det."],
        alpha=0.3,
        label="Diff. det.",
    )
    ax.fill_between(
        bins_range,
        ens_diff_lower_bound,
        ens_diff_upper_bound,
        color=MODEL_COLOR_DICT["Diff. ens."],
        alpha=0.3,
        label="Diff. ens.",
    )
    ax.fill_between(
        bins_range,
        wrf_lower_bound,
        wrf_upper_bound,
        color=MODEL_COLOR_DICT["WRF"],
        alpha=0.3,
        label="WRF",
    )
    ax.plot(
        bins_range,
        cpc_cdf,
        label="CombiPrecip",
        color=MODEL_COLOR_DICT["CombiPrecip"],
        linewidth=2,
    )

    ax.legend()
    plt.title(f"lead time = {lead_time_name}")
    fig.savefig(os.path.join(figs_dir, f"cdfs/dist_{lead_time_name}_right.png"))


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

    # plots
    ax.plot(
        bins_range,
        det_s2s_cdf,
        label="S2S det.",
        color=MODEL_COLOR_DICT["S2S det."],
        linewidth=2,
    )
    ax.fill_between(
        bins_range,
        ens_s2s_lower_bound,
        ens_s2s_upper_bound,
        color=MODEL_COLOR_DICT["S2S ens."],
        alpha=0.3,
        label="S2S ens.",
    )
    ax.fill_between(
        bins_range,
        det_diff_lower_bound,
        det_diff_upper_bound,
        color=MODEL_COLOR_DICT["Diff. det."],
        alpha=0.3,
        label="Diff. det.",
    )
    ax.fill_between(
        bins_range,
        ens_diff_lower_bound,
        ens_diff_upper_bound,
        color=MODEL_COLOR_DICT["Diff. ens."],
        alpha=0.3,
        label="Diff. ens.",
    )
    ax.fill_between(
        bins_range,
        wrf_lower_bound,
        wrf_upper_bound,
        color=MODEL_COLOR_DICT["WRF"],
        alpha=0.3,
        label="WRF",
    )
    ax.plot(
        bins_range,
        cpc_cdf,
        label="CombiPrecip",
        color=MODEL_COLOR_DICT["CombiPrecip"],
        linewidth=2,
    )

    ax.set_xlim(0, 5)
    ax.set_ylim(0, 1.0)
    ax.set_xlabel("Precipitation (mm/h)")
    ax.set_ylabel("Cumulative distribution function")

    ax.legend()
    plt.title(f"lead time = {lead_time_name}")
    fig.savefig(os.path.join(figs_dir, f"cdfs/dist_{lead_time_name}_left.png"))


def plot_lead_time_distribution(
    det_s2s,
    ens_s2s,
    det_diff,
    ens_diff,
    wrf,
    cpc,
    lead_time_idx,
    figs_dir,
    bins=100,
):
    # get data
    det_s2s_data = det_s2s.precip[lead_time_idx]
    ens_s2s_data = ens_s2s.precip[lead_time_idx]
    det_diff_data = det_diff.precip[lead_time_idx, :, :]
    ens_diff_data = ens_diff.precip[lead_time_idx, :, :]
    wrf_data = wrf.precip[lead_time_idx, :, :]
    cpc_data = cpc.precip.flatten()

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


def plot_lead_time_map_raw(
    det_s2s,
    cpc,
    lead_time_idx,
    figs_dir,
):
    image_paths = []

    # save temporary images for each time_idx
    for time_idx in TIME_IDXS:
        arrays = (
            cpc.precip[time_idx],
            det_s2s.precip[0, time_idx],
            det_s2s.precip[1, time_idx],
            det_s2s.precip[2, time_idx],
        )
        titles = (
            "a) CombiPrecip",
            "a) S2S det. (1-week)",
            "b) S2S det. (2-week)",
            "c) S2S det. (3-week)",
        )
        cpc_extent = cpc.get_extent()
        extents = (cpc_extent,) * 6
        fig, _ = plot_maps(arrays, titles, extents)

        image_path = os.path.join(figs_dir, f"temp_map_t{time_idx}.png")
        fig.savefig(image_path)
        plt.close(fig)  # important to avoid memory leak
        image_paths.append(image_path)

    # create the GIFs
    for event_idx in range(1, len(TIME_IDXS) // EVENT_LENGTH + 1):
        gif_path = os.path.join(figs_dir, f"maps/maps_raw_e{event_idx}.gif")
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


def plot_lead_time_map_complete(
    det_s2s,
    ens_s2s,
    det_diff,
    ens_diff,
    wrf,
    cpc,
    lead_time_idx,
    num_idx,
    figs_dir,
):
    lead_time_name = det_s2s.lead_time[lead_time_idx]
    print(f"lead time = {lead_time_name}")
    image_paths = []

    # save temporary images for each time_idx
    for time_idx in TIME_IDXS:
        arrays = (
            det_s2s.precip[lead_time_idx, time_idx],
            ens_s2s.precip[lead_time_idx, num_idx, time_idx],
            det_diff.precip[lead_time_idx, num_idx, time_idx],
            ens_diff.precip[lead_time_idx, num_idx, time_idx],
            wrf.precip[lead_time_idx, num_idx, time_idx],
            cpc.precip[time_idx],
        )
        titles = (
            "a) S2S det.",
            "b) S2S ens.",
            "c) Diff. det.",
            "d) Diff. ens.",
            "e) WRF",
            "f) CombiPrecip",
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
    for event_idx in range(1, len(TIME_IDXS) // EVENT_LENGTH + 1):
        gif_path = os.path.join(
            figs_dir, f"maps/maps_{lead_time_name}_e{event_idx}.gif"
        )
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
    lead_time_idx,
    figs_dir,
):
    # get timeseries
    det_s2s_timeseries = np.mean(
        det_s2s.precip[lead_time_idx],
        axis=(1, 2),
    )
    ens_s2s_timeseries = np.mean(
        ens_s2s.precip[lead_time_idx],
        axis=(2, 3),
    )
    det_diff_timeseries = np.mean(
        det_diff.precip[lead_time_idx],
        axis=(2, 3),
    )
    ens_diff_timeseries = np.mean(
        ens_diff.precip[lead_time_idx],
        axis=(2, 3),
    )
    wrf_timeseries = np.mean(
        wrf.precip[lead_time_idx],
        axis=(2, 3),
    )
    cpc_timeseries = np.mean(
        cpc.precip,
        axis=(1, 2),
    )
    dates = cpc.time

    # compute ensemble mean and spread
    ens_s2s_mean = np.mean(ens_s2s_timeseries, axis=0)
    det_diff_mean = np.mean(det_diff_timeseries, axis=0)
    ens_diff_mean = np.mean(ens_diff_timeseries, axis=0)
    wrf_diff_mean = np.mean(wrf_timeseries, axis=0)
    # use std for spread
    ens_s2s_std = np.std(ens_s2s_timeseries, axis=0)
    det_diff_std = np.std(det_diff_timeseries, axis=0)
    ens_diff_std = np.std(ens_diff_timeseries, axis=0)
    wrf_diff_std = np.std(wrf_timeseries, axis=0)
    # get bounds
    ens_s2s_lower_bound = np.maximum(ens_s2s_mean - ens_s2s_std, 0)
    ens_s2s_upper_bound = np.maximum(ens_s2s_mean + ens_s2s_std, 0)
    det_diff_lower_bound = np.maximum(det_diff_mean - det_diff_std, 0)
    det_diff_upper_bound = np.maximum(det_diff_mean + det_diff_std, 0)
    ens_diff_lower_bound = np.maximum(ens_diff_mean - ens_diff_std, 0)
    ens_diff_upper_bound = np.maximum(ens_diff_mean + ens_diff_std, 0)
    wrf_diff_lower_bound = np.maximum(wrf_diff_mean - wrf_diff_std, 0)
    wrf_diff_upper_bound = np.maximum(wrf_diff_mean + wrf_diff_std, 0)

    # plot timeseries for each event
    for event_idx in range(1, len(TIME_IDXS) // EVENT_LENGTH + 1):
        idxs = slice(EVENT_LENGTH * (event_idx - 1), EVENT_LENGTH * event_idx)

        # create figure and axis
        fig, ax = plt.subplots(figsize=(8, 4))

        # beginning
        ax.plot(
            dates[idxs],
            det_s2s_timeseries[idxs],
            label="S2S det.",
            color=MODEL_COLOR_DICT["S2S det."],
        )

        # add shaded region for ensemble spread
        ax.fill_between(
            dates[idxs],
            ens_s2s_lower_bound[idxs],
            ens_s2s_upper_bound[idxs],
            color=MODEL_COLOR_DICT["S2S ens."],
            alpha=0.3,
            label="S2S ens.",
        )
        ax.fill_between(
            dates[idxs],
            det_diff_lower_bound[idxs],
            det_diff_upper_bound[idxs],
            color=MODEL_COLOR_DICT["Diff. det."],
            alpha=0.3,
            label="Diff. det.",
        )
        ax.fill_between(
            dates[idxs],
            ens_diff_lower_bound[idxs],
            ens_diff_upper_bound[idxs],
            color=MODEL_COLOR_DICT["Diff. ens."],
            alpha=0.3,
            label="Diff. ens.",
        )
        ax.fill_between(
            dates[idxs],
            wrf_diff_lower_bound[idxs],
            wrf_diff_upper_bound[idxs],
            color=MODEL_COLOR_DICT["WRF"],
            alpha=0.3,
            label="WRF",
        )

        # end
        ax.plot(
            dates[idxs],
            cpc_timeseries[idxs],
            label="CombiPrecip",
            color=MODEL_COLOR_DICT["CombiPrecip"],
        )

        # add legend and save
        ax.set_xlabel("Dates")
        ax.set_ylabel("Mean precipitation (mm/h)")
        plt.legend()

        lead_time_name = det_s2s.lead_time[lead_time_idx]
        plt.title(f"lead time = {lead_time_name}, event = {event_idx}")
        fig.savefig(
            os.path.join(figs_dir, f"timeseries/ts_{lead_time_name}_e{event_idx}.png")
        )


def plot_lead_time_psd(
    det_s2s,
    ens_s2s,
    det_diff,
    ens_diff,
    wrf,
    cpc,
    lead_time_idx,
    figs_dir,
):
    spatial_lenghts = det_s2s.get_spatial_lengths()
    k, det_s2s_psd = get_1dpsd(det_s2s.precip[lead_time_idx], *spatial_lenghts)
    _, ens_s2s_psd = get_1dpsd(
        np.mean(ens_s2s.precip[lead_time_idx], axis=0),
        *spatial_lenghts,
    )
    _, det_diff_psd = get_1dpsd(
        np.mean(det_diff.precip[lead_time_idx], axis=0),
        *spatial_lenghts,
    )
    _, ens_diff_psd = get_1dpsd(
        np.mean(ens_diff.precip[lead_time_idx], axis=0),
        *spatial_lenghts,
    )
    _, wrf_psd = get_1dpsd(
        np.mean(wrf.precip[lead_time_idx], axis=0),
        *spatial_lenghts,
    )
    _, cpc_psd = get_1dpsd(cpc.precip, *spatial_lenghts)
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
        det_s2s_psd[mask][::-1],
        label="S2S det.",
        color=MODEL_COLOR_DICT["S2S det."],
    )
    ax.plot(
        wavelengths[mask],
        ens_s2s_psd[mask][::-1],
        label="S2S ens.",
        color=MODEL_COLOR_DICT["S2S ens."],
    )
    ax.plot(
        wavelengths[mask],
        det_diff_psd[mask][::-1],
        label="det. diff.",
        color=MODEL_COLOR_DICT["Diff. det."],
    )
    ax.plot(
        wavelengths[mask],
        ens_diff_psd[mask][::-1],
        label="ens. diff.",
        color=MODEL_COLOR_DICT["Diff. ens."],
    )
    ax.plot(
        wavelengths[mask],
        wrf_psd[mask][::-1],
        label="WRF",
        color=MODEL_COLOR_DICT["WRF"],
    )
    ax.plot(
        wavelengths[mask],
        cpc_psd[mask][::-1],
        label="CombiPrecip",
        color=MODEL_COLOR_DICT["CombiPrecip"],
    )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Wavelengths (km)")
    ax.set_ylabel("Power spectral density")
    ax.legend()

    lead_time_name = det_s2s.lead_time[lead_time_idx]
    plt.title(f"lead time = {lead_time_name}")
    fig.savefig(os.path.join(figs_dir, f"psds/psd_{lead_time_name}.png"))


def plot_rank_histogram(
    ens_s2s,
    det_diff,
    ens_diff,
    wrf,
    cpc,
    lead_time_idx,
    figs_dir,
):
    ens_s2s_values = ens_s2s.precip[lead_time_idx]
    det_diff_values = det_diff.precip[lead_time_idx]
    ens_diff_values = ens_diff.precip[lead_time_idx]
    wrf_values = wrf.precip[lead_time_idx]
    cpc_values = cpc.precip

    # calculate ranks
    ens_s2s_ranks = rank_histogram(ens_s2s_values, cpc_values)
    det_diff_ranks = rank_histogram(det_diff_values, cpc_values)
    ens_diff_ranks = rank_histogram(ens_diff_values, cpc_values)
    wrf_ranks = rank_histogram(wrf_values, cpc_values)

    # plot rank histograms
    fig, ax = plt.subplots(2, 2, figsize=(8, 4), dpi=300)
    ax[0, 0].hist(
        ens_s2s_ranks,
        color=MODEL_COLOR_DICT["S2S ens."],
        histtype="step",
        density=True,
    )
    ax[0, 0].set_title("a) S2S ens.")
    ax[0, 1].hist(det_diff_ranks, color=MODEL_COLOR_DICT["Diff. det."], density=True)
    ax[0, 1].set_title("b) Diff. det.")
    ax[1, 0].hist(ens_diff_ranks, color=MODEL_COLOR_DICT["Diff. ens."], density=True)
    ax[1, 0].set_title("c) Diff. ens.")
    ax[1, 1].hist(wrf_ranks, color=MODEL_COLOR_DICT["WRF"], density=True)
    ax[1, 1].set_title("d) WRF")

    # labels
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    ax[1, 0].set_xlabel("Rank")
    ax[1, 1].set_xlabel("Rank")
    ax[0, 0].set_ylabel("Frequency")
    ax[1, 0].set_ylabel("Frequency")

    plt.tight_layout()

    # save
    lead_time_name = ens_s2s.lead_time[lead_time_idx]
    fig.savefig(os.path.join(figs_dir, f"ranks/rank_histograms_{lead_time_name}.png"))


def get_av_fss(forecast, observations, threshold, num_neighbor):
    members = np.arange(forecast.shape[0])
    times = np.arange(forecast.shape[1])
    avfss = 0.0
    discount = 0
    for m in members:
        for t in times:
            calcfss = fss(forecast[m, t], observations[t], threshold, num_neighbor)
            if np.isnan(calcfss):
                discount += 1
            else:
                avfss += calcfss
            # print("avfss:", avfss)
    if discount == len(members) * len(times):
        return np.nan
    else:
        return avfss / (len(members) * len(times) - discount)


def plot_avFSS(
    det_s2s,
    ens_s2s,
    det_diff,
    ens_diff,
    wrf,
    cpc,
    lead_time_idx,
    figs_dir,
):
    lead_time_name = det_s2s.lead_time[lead_time_idx]
    print("avFSS -", lead_time_name)

    # get values for lead time
    det_s2s_values = np.expand_dims(det_s2s.precip[lead_time_idx], axis=0)
    ens_s2s_values = ens_s2s.precip[lead_time_idx]
    det_diff_values = det_diff.precip[lead_time_idx]
    ens_diff_values = ens_diff.precip[lead_time_idx]
    wrf_values = wrf.precip[lead_time_idx]
    cpc_values = cpc.precip

    # define FSS parameters
    thresholds = [0.5, 1.0, 1.5, 2.0, 2.5]
    num_neighbors = [5, 15]

    # arrays for model and parameters
    det_s2s_avfss_arr = np.zeros((len(thresholds), len(num_neighbors)))
    ens_s2s_avfss_arr = np.zeros((len(thresholds), len(num_neighbors)))
    det_diff_avfss_arr = np.zeros((len(thresholds), len(num_neighbors)))
    ens_diff_avfss_arr = np.zeros((len(thresholds), len(num_neighbors)))
    wrf_avfss_arr = np.zeros((len(thresholds), len(num_neighbors)))

    # get avFSS arrays
    for i_thr, thr in enumerate(thresholds):
        print("threshold:", thr)
        for i_num, num in enumerate(num_neighbors):
            det_s2s_avfss_arr[i_thr, i_num] = get_av_fss(
                det_s2s_values, cpc_values, thr, num
            )
            ens_s2s_avfss_arr[i_thr, i_num] = get_av_fss(
                ens_s2s_values, cpc_values, thr, num
            )
            det_diff_avfss_arr[i_thr, i_num] = get_av_fss(
                det_diff_values, cpc_values, thr, num
            )
            ens_diff_avfss_arr[i_thr, i_num] = get_av_fss(
                ens_diff_values, cpc_values, thr, num
            )
            wrf_avfss_arr[i_thr, i_num] = get_av_fss(wrf_values, cpc_values, thr, num)

    model_labels = ["S2S det.", "S2S ens.", "Diff. det.", "Diff. ens.", "WRF"]
    model_arrs = [
        det_s2s_avfss_arr,
        ens_s2s_avfss_arr,
        det_diff_avfss_arr,
        ens_diff_avfss_arr,
        wrf_avfss_arr,
    ]

    # plot avFSS vs. thresholds for each num_neighbors
    for i_num, num in enumerate(num_neighbors):
        fig, ax = plt.subplots(figsize=(4, 3))
        for model_label, model_arr in zip(model_labels, model_arrs):
            ax.plot(
                thresholds,
                model_arr[:, i_num],
                label=model_label,
                color=MODEL_COLOR_DICT[model_label],
                linestyle=MODEL_LINESTYLE[model_label],
            )
        # add legend and save
        ax.set_xlabel("Threshold (mm/hr)")
        ax.set_xticks(thresholds)
        ax.set_xlim(thresholds[0] - 0.05, thresholds[-1] + 0.05)
        ax.set_ylabel("avFSS")
        ax.set_yticks(np.arange(0.0, 0.4, 0.1))
        ax.set_ylim(0.0, 0.25)
        _make_arrows(ax, thresholds[0] - 0.05, 1)
        ax.spines["left"].set_position(("data", thresholds[0] - 0.05))
        plt.legend(loc="upper right")
        plt.tight_layout()
        fig.savefig(
            os.path.join(figs_dir, f"fss/fss_{lead_time_name}_num{num}.png"),
            dpi=300,
        )
        plt.close()


def make_plots(
    s2s_det_path,
    s2s_ens_path,
    diff_det_path,
    diff_ens_path,
    wrf_path,
    cpc_path,
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
    os.makedirs(figs_dir + "/maps", exist_ok=True)
    os.makedirs(figs_dir + "/timeseries", exist_ok=True)
    os.makedirs(figs_dir + "/cdfs", exist_ok=True)
    os.makedirs(figs_dir + "/psds", exist_ok=True)
    os.makedirs(figs_dir + "/ranks", exist_ok=True)
    os.makedirs(figs_dir + "/fss", exist_ok=True)

    # plot gifs for each lead time (and each event)
    plot_lead_time_map_raw(
        det_s2s,
        cpc,
        num_idx,
        figs_dir,
    )
    print("maps raw saved")

    # plot gifs for each lead time (and each event)
    for lead_time_idx in range(3):
        plot_lead_time_map_complete(
            det_s2s,
            ens_s2s,
            det_diff,
            ens_diff,
            wrf,
            cpc,
            lead_time_idx,
            num_idx,
            figs_dir,
        )
    print("maps complete saved")

    # plot timeseries for each lead time (and each event)
    for lead_time_idx in range(3):
        plot_lead_time_timeseries(
            det_s2s,
            ens_s2s,
            det_diff,
            ens_diff,
            wrf,
            cpc,
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
            lead_time_idx,
            figs_dir,
        )
    print("psds saved")

    # plot rank histogram for each lead time
    for lead_time_idx in range(3):
        plot_rank_histogram(
            ens_s2s,
            det_diff,
            ens_diff,
            wrf,
            cpc,
            lead_time_idx,
            figs_dir,
        )
    print("rank histograms saved")

    # plot avFSS for lead time
    for lead_time_idx in range(3):
        plot_avFSS(
            det_s2s,
            ens_s2s,
            det_diff,
            ens_diff,
            wrf,
            cpc,
            lead_time_idx,
            figs_dir,
        )
    print("avFSS saved")


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
    weight = "heavy"
    cli = 50
    diff_det_path = os.path.join(
        simulations_dir,
        "diffusion",
        # change to new here!
        f"det_{weight}_cli{cli}_ens50.h5",
    )
    diff_ens_path = os.path.join(
        simulations_dir,
        "diffusion",
        # change to new here!
        f"ens_{weight}_cli{cli}_ens50.h5",
    )
    wrf_path = os.path.join(
        simulations_dir,
        "wrf",
        "wrf.h5",
    )
    cpc_path = os.path.join(test_data_dir, "cpc.h5")

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
        num_idx,
    )


if __name__ == "__main__":
    main()
