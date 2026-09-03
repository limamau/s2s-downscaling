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
from utils import get_cdf, get_pdf

EVENT_LENGTH = 8
NUMBER_OF_EVENTS = 2
MODEL_COLOR_DICT = {
    "IFS det + NNI": "C0",
    "IFS ens + NNI": "C1",
    "DDPM det": "C0",
    "DDPM ens": "C1",
    "WRF": "C2",
    "CombiPrecip": "C4",
}
MODEL_LINESTYLE = {
    "IFS det + NNI": "--",
    "IFS ens + NNI": "--",
    "DDPM det": "-",
    "DDPM ens": "-",
    "WRF": "-",
    "CombiPrecip": "-",
}
TIME_IDXS = [i for i in range(EVENT_LENGTH * NUMBER_OF_EVENTS)]


### auxiliary functions ###
def _make_arrows(ax, xpos=(1), ypos=(1)):
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
        xpos,
        ypos,
        ls="",
        marker="^",
        ms=5,
        color="k",
        transform=ax.get_xaxis_transform(),
        clip_on=False,
    )
    # clean axis
    # ax.spines["left"].set_position("zero")
    ax.spines["right"].set_visible(False)
    # ax.spines["bottom"].set_position("zero")
    ax.spines["top"].set_visible(False)
    # ax.xaxis.set_ticks_position("bottom")
    # ax.yaxis.set_ticks_position("left")


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


def plot_left_tale(
    det_s2s_data,
    ens_s2s_data,
    det_diff_data,
    ens_diff_data,
    wrf_data,
    cpc_data,
    bins,
    figs_dir,
    lead_time_name,
    event_idx,
):
    # compute PDFs
    max_precip = 25.0
    bins_range = np.linspace(0, max_precip, bins)
    bins_mid = bins_range[:-1] + max_precip / bins / 2
    det_s2s_pdf = get_pdf(det_s2s_data, bins_range) * bins_mid
    ens_s2s_pdf = get_pdf(ens_s2s_data, bins_range) * bins_mid
    det_diff_pdf = get_pdf(det_diff_data, bins_range) * bins_mid
    ens_diff_pdf = get_pdf(ens_diff_data, bins_range) * bins_mid
    wrf_pdf = get_pdf(wrf_data, bins_range) * bins_mid
    cpc_pdf = get_pdf(cpc_data, bins_range) * bins_mid

    # plots
    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.plot(
        bins_mid,
        det_s2s_pdf,
        label="IFS det + NNI",
        color=MODEL_COLOR_DICT["IFS det + NNI"],
        linestyle=MODEL_LINESTYLE["IFS det + NNI"],
    )
    ax.plot(
        bins_mid,
        ens_s2s_pdf,
        label="IFS ens + NNI",
        color=MODEL_COLOR_DICT["IFS ens + NNI"],
        linestyle=MODEL_LINESTYLE["IFS ens + NNI"],
    )
    ax.plot(
        bins_mid,
        det_diff_pdf,
        label="DDPM det",
        color=MODEL_COLOR_DICT["DDPM det"],
        linestyle=MODEL_LINESTYLE["DDPM det"],
    )
    ax.plot(
        bins_mid,
        ens_diff_pdf,
        label="DDPM ens",
        color=MODEL_COLOR_DICT["DDPM ens"],
        linestyle=MODEL_LINESTYLE["DDPM ens"],
    )
    ax.plot(
        bins_mid,
        wrf_pdf,
        label="WRF",
        color=MODEL_COLOR_DICT["WRF"],
        linestyle=MODEL_LINESTYLE["WRF"],
    )
    ax.plot(
        bins_mid,
        cpc_pdf,
        label="CombiPrecip",
        color=MODEL_COLOR_DICT["CombiPrecip"],
        linestyle=MODEL_LINESTYLE["CombiPrecip"],
    )

    ax.set_xlabel("Precipitation (mm/h)")
    ax.set_ylabel("PDF x precip.")
    ax.legend()
    ax.set_xlim(0, 5.0)
    ax.set_ylim(0, None)
    _make_arrows(ax, (0), (1))
    plt.tight_layout()
    fig.savefig(
        os.path.join(
            figs_dir, f"dists/dist_{lead_time_name}_e{event_idx + 1}_left.png"
        ),
        dpi=300,
    )


def plot_right_tale(
    det_s2s_data,
    ens_s2s_data,
    det_diff_data,
    ens_diff_data,
    wrf_data,
    cpc_data,
    bins,
    figs_dir,
    lead_time_name,
    event_idx,
):
    # compute PDFs
    max_precip = 25.0
    bins_range = np.linspace(0, max_precip, bins)
    det_s2s_cdf = get_cdf(det_s2s_data, bins_range)
    ens_s2s_cdf = get_cdf(ens_s2s_data, bins_range)
    det_diff_cdf = get_cdf(det_diff_data, bins_range)
    ens_diff_cdf = get_cdf(ens_diff_data, bins_range)
    wrf_cdf = get_cdf(wrf_data, bins_range)
    cpc_cdf = get_cdf(cpc_data, bins_range)

    # plots
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(
        bins_range,
        det_s2s_cdf,
        label="IFS det + NNI",
        color=MODEL_COLOR_DICT["IFS det + NNI"],
        linestyle=MODEL_LINESTYLE["IFS det + NNI"],
    )
    ax.plot(
        bins_range,
        ens_s2s_cdf,
        label="IFS ens + NNI",
        color=MODEL_COLOR_DICT["IFS ens + NNI"],
        linestyle=MODEL_LINESTYLE["IFS ens + NNI"],
    )
    ax.plot(
        bins_range,
        det_diff_cdf,
        label="DDPM det",
        color=MODEL_COLOR_DICT["DDPM det"],
        linestyle=MODEL_LINESTYLE["DDPM det"],
    )
    ax.plot(
        bins_range,
        ens_diff_cdf,
        label="DDPM ens",
        color=MODEL_COLOR_DICT["DDPM ens"],
        linestyle=MODEL_LINESTYLE["DDPM ens"],
    )
    ax.plot(
        bins_range,
        wrf_cdf,
        label="WRF",
        color=MODEL_COLOR_DICT["WRF"],
        linestyle=MODEL_LINESTYLE["WRF"],
    )
    ax.plot(
        bins_range,
        cpc_cdf,
        label="CombiPrecip",
        color=MODEL_COLOR_DICT["CombiPrecip"],
        linestyle=MODEL_LINESTYLE["CombiPrecip"],
    )

    ax.set_xlabel("Precipitation (mm/h)")
    ax.set_ylabel("PDF x precip.")
    ax.legend()
    ax.set_xlim(5.0, 10.0)
    ax.set_xscale("log")
    ax.set_yscale("log")
    mask = bins_range > 5.0
    ymin = cpc_cdf[mask].min()
    ymax = cpc_cdf[mask].max()
    ax.set_ylim(ymin * 0.9999, ymax * 1.0001)
    _make_arrows(ax, (0), (1))
    fig.savefig(
        os.path.join(
            figs_dir, f"dists/dist_{lead_time_name}_e{event_idx + 1}_right.png"
        ),
        dpi=300,
    )


def plot_lead_time_dists(
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
    lead_time_name = det_s2s.lead_time[lead_time_idx]

    for event_idx in range(NUMBER_OF_EVENTS):
        det_s2s_data = det_s2s.precip[
            lead_time_idx + 1, event_idx * EVENT_LENGTH : (event_idx + 1) * EVENT_LENGTH
        ].flatten()
        ens_s2s_data = ens_s2s.precip[
            lead_time_idx + 1,
            :,
            event_idx * EVENT_LENGTH : (event_idx + 1) * EVENT_LENGTH,
        ].flatten()
        det_diff_data = det_diff.precip[
            lead_time_idx, :, event_idx * EVENT_LENGTH : (event_idx + 1) * EVENT_LENGTH
        ].flatten()
        ens_diff_data = ens_diff.precip[
            lead_time_idx, :, event_idx * EVENT_LENGTH : (event_idx + 1) * EVENT_LENGTH
        ].flatten()
        wrf_data = wrf.precip[
            lead_time_idx, :, event_idx * EVENT_LENGTH : (event_idx + 1) * EVENT_LENGTH
        ].flatten()
        cpc_data = cpc.precip[
            event_idx * EVENT_LENGTH : (event_idx + 1) * EVENT_LENGTH
        ].flatten()

        # # plot ppdfs for the left tale
        # plot_left_tale(
        #     det_s2s_data,
        #     ens_s2s_data,
        #     det_diff_data,
        #     ens_diff_data,
        #     wrf_data,
        #     cpc_data,
        #     bins,
        #     figs_dir,
        #     lead_time_name,
        #     event_idx,
        # )

        plot_right_tale(
            det_s2s_data,
            ens_s2s_data,
            det_diff_data,
            ens_diff_data,
            wrf_data,
            cpc_data,
            bins,
            figs_dir,
            lead_time_name,
            event_idx,
        )


def plot_lead_time_agg_raw(climatology, det_s2s, cpc, figs_dir):
    time_idxs2018 = slice(0, EVENT_LENGTH)
    arrays2018 = (
        np.mean(cpc.precip[time_idxs2018], axis=0),
        climatology.precip[0],  # makes no difference for climatology!
        np.mean(det_s2s.precip[0, time_idxs2018], axis=0),
        np.mean(det_s2s.precip[1, time_idxs2018], axis=0),
        np.mean(det_s2s.precip[2, time_idxs2018], axis=0),
        np.mean(det_s2s.precip[3, time_idxs2018], axis=0),
    )
    time_idxs2021 = slice(EVENT_LENGTH, 2 * EVENT_LENGTH)
    arrays2021 = (
        np.mean(cpc.precip[time_idxs2021], axis=0),
        climatology.precip[0],  # makes no difference for climatology!
        np.mean(det_s2s.precip[0, time_idxs2021], axis=0),
        np.mean(det_s2s.precip[1, time_idxs2021], axis=0),
        np.mean(det_s2s.precip[2, time_idxs2021], axis=0),
        np.mean(det_s2s.precip[3, time_idxs2021], axis=0),
    )
    arrays = [element for tupl in zip(arrays2018, arrays2021) for element in tupl]
    titles = (
        "a) CombiPrecip",
        "b) CombiPrecip",
        "c) Climatology",
        "d) Climatology",
        "e) IFS det + NNI (0-day)",
        "f) IFS det + NNI (0-day)",
        "g) IFS det + NNI (1-week)",
        "h) IFS det + NNI (1-week)",
        "i) IFS det + NNI (2-week)",
        "j) IFS det + NNI (2-week)",
        "k) IFS det + NNI (3-week)",
        "l) IFS det + NNI (3-week)",
    )
    cpc_extent = cpc.get_extent()
    extents = (cpc_extent,) * 12
    fig, _ = plot_maps(arrays, titles, extents)

    image_path = os.path.join(figs_dir, "maps/agg_raw.png")
    fig.savefig(image_path)
    plt.close(fig)


def plot_lead_time_gifs_raw(det_s2s, cpc, figs_dir):
    image_paths = []

    # save temporary images for each time_idx
    for time_idx in TIME_IDXS:
        arrays = (
            cpc.precip[time_idx],
            det_s2s.precip[1, time_idx],
            det_s2s.precip[2, time_idx],
            det_s2s.precip[3, time_idx],
        )
        titles = (
            "a) CombiPrecip",
            "a) IFS det + NNI (1-week)",
            "b) IFS det + NNI (2-week)",
            "c) IFS det + NNI (3-week)",
        )
        cpc_extent = cpc.get_extent()
        extents = (cpc_extent,) * 6
        fig, _ = plot_maps(arrays, titles, extents)

        image_path = os.path.join(figs_dir, f"temp_map_t{time_idx}.png")
        fig.savefig(image_path)
        plt.close(fig)
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
    lead_time_name = det_s2s.lead_time[lead_time_idx + 1]
    print(f"lead time = {lead_time_name}")
    image_paths = []

    print("det_s2s shape", det_s2s.precip.shape)
    print("ens_s2s shape", ens_s2s.precip.shape)
    print("det_diff shape", det_diff.precip.shape)
    print("ens_diff shape", ens_diff.precip.shape)
    print("wrf shape", wrf.precip.shape)
    print("cpc shape", cpc.precip.shape)

    # save temporary images for each time_idx
    for time_idx in TIME_IDXS:
        arrays = (
            det_s2s.precip[lead_time_idx + 1, time_idx],
            ens_s2s.precip[lead_time_idx + 1, num_idx, time_idx],
            det_diff.precip[lead_time_idx, num_idx, time_idx],
            ens_diff.precip[lead_time_idx, num_idx, time_idx],
            wrf.precip[lead_time_idx, num_idx, time_idx],
            cpc.precip[time_idx],
        )
        titles = (
            "a) IFS det + NNI",
            "b) IFS ens + NNI",
            "c) DDPM det",
            "d) DDPM ens",
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


def plot_lead_time_agg(
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
    lead_time_name = det_s2s.lead_time[lead_time_idx + 1]
    print(f"lead time = {lead_time_name}")

    def _extract_event_data(event_idx):
        time_slice = slice((event_idx - 1) * EVENT_LENGTH, event_idx * EVENT_LENGTH)

        # det_s2s
        det_arr = np.mean(det_s2s.precip[lead_time_idx + 1, time_slice], axis=0)

        # ens_s2s
        ens_event = ens_s2s.precip[lead_time_idx + 1, :, time_slice]
        best_ens_idx = int(np.argmax(np.sum(ens_event, axis=(1, 2, 3))))
        ens_arr = np.mean(ens_event[best_ens_idx], axis=0)
        num_ens = ens_event.shape[0]

        # det_diff
        det_diff_event = det_diff.precip[lead_time_idx, :, time_slice]
        best_det_diff_idx = int(np.argmax(np.sum(det_diff_event, axis=(1, 2, 3))))
        det_diff_arr = np.mean(det_diff_event[best_det_diff_idx], axis=0)
        num_det_diff = det_diff_event.shape[0]

        # ens_diff
        ens_diff_event = ens_diff.precip[lead_time_idx, :, time_slice]
        best_ens_diff_idx = int(np.argmax(np.sum(ens_diff_event, axis=(1, 2, 3))))
        ens_diff_arr = np.mean(ens_diff_event[best_ens_diff_idx], axis=0)
        num_ens_diff = ens_diff_event.shape[0]

        # wrf
        wrf_event = wrf.precip[lead_time_idx, :, time_slice]
        best_wrf_idx = int(np.argmax(np.sum(wrf_event, axis=(1, 2, 3))))
        wrf_arr = np.mean(wrf_event[best_wrf_idx], axis=0)
        num_wrf = wrf_event.shape[0]

        # cpc
        cpc_arr = np.mean(cpc.precip[time_slice], axis=0)

        arrays = (det_arr, ens_arr, det_diff_arr, ens_diff_arr, wrf_arr, cpc_arr)
        labels = (
            "IFS det + NNI",
            f"IFS ens + NNI (member {best_ens_idx + 1}/{num_ens})",
            f"DDPM det (member {best_det_diff_idx + 1}/{num_det_diff})",
            f"DDPM ens (member {best_ens_diff_idx + 1}/{num_ens_diff})",
            f"WRF (member {best_wrf_idx + 1}/{num_wrf})",
            "CombiPrecip",
        )
        return arrays, labels

    # Process 2018 (event 1) and 2021 (event 2)
    arrays2018, labels2018 = _extract_event_data(1)
    arrays2021, labels2021 = _extract_event_data(2)

    # Interleave arrays row by row: [2018_row0, 2021_row0, 2018_row1, 2021_row1, ...]
    arrays = [val for pair in zip(arrays2018, arrays2021) for val in pair]

    # Generate sequential panel titles from a) to l)
    letters = [f"{chr(97 + i)})" for i in range(12)]
    raw_labels = [val for pair in zip(labels2018, labels2021) for val in pair]
    titles = [f"{let} {lbl}" for let, lbl in zip(letters, raw_labels)]

    cpc_extent = cpc.get_extent()
    extents = (cpc_extent,) * 12

    fig, _ = plot_maps(arrays, titles, extents)

    image_path = os.path.join(figs_dir, f"maps/agg_map_{lead_time_name}.png")
    fig.savefig(image_path, bbox_inches="tight")
    plt.close(fig)


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
        det_s2s.precip[lead_time_idx + 1],
        axis=(1, 2),
    )
    ens_s2s_timeseries = np.mean(
        ens_s2s.precip[lead_time_idx + 1],
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
            label="IFS det + NNI",
            color=MODEL_COLOR_DICT["IFS det + NNI"],
            linestyle=MODEL_LINESTYLE["IFS det + NNI"],
            linewidth=2,
        )

        # add shaded region for ensemble spread
        ax.fill_between(
            dates[idxs],
            ens_s2s_lower_bound[idxs],
            ens_s2s_upper_bound[idxs],
            color=MODEL_COLOR_DICT["IFS ens + NNI"],
            label="IFS ens + NNI",
            facecolor="none",
            hatch="//",
            edgecolor=MODEL_COLOR_DICT["IFS ens + NNI"],
        )
        ax.fill_between(
            dates[idxs],
            det_diff_lower_bound[idxs],
            det_diff_upper_bound[idxs],
            color=MODEL_COLOR_DICT["DDPM det"],
            alpha=0.5,
            label="DDPM det",
        )
        ax.fill_between(
            dates[idxs],
            ens_diff_lower_bound[idxs],
            ens_diff_upper_bound[idxs],
            color=MODEL_COLOR_DICT["DDPM ens"],
            alpha=0.5,
            label="DDPM ens",
        )
        ax.fill_between(
            dates[idxs],
            wrf_diff_lower_bound[idxs],
            wrf_diff_upper_bound[idxs],
            color=MODEL_COLOR_DICT["WRF"],
            alpha=0.5,
            label="WRF",
        )

        # end
        ax.plot(
            dates[idxs],
            cpc_timeseries[idxs],
            label="CombiPrecip",
            color=MODEL_COLOR_DICT["CombiPrecip"],
            linewidth=2,
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
    k, det_s2s_psd = get_1dpsd(det_s2s.precip[lead_time_idx + 1], *spatial_lenghts)
    _, ens_s2s_psd = get_1dpsd(
        np.mean(ens_s2s.precip[lead_time_idx + 1], axis=0),
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

    fig, ax = plt.subplots(figsize=(7, 5))

    ax.plot(
        wavelengths[mask],
        det_s2s_psd[mask][::-1],
        label="IFS det + NNI",
        color=MODEL_COLOR_DICT["IFS det + NNI"],
        linestyle=MODEL_LINESTYLE["IFS det + NNI"],
    )
    ax.plot(
        wavelengths[mask],
        ens_s2s_psd[mask][::-1],
        label="IFS ens + NNI",
        color=MODEL_COLOR_DICT["IFS ens + NNI"],
        linestyle=MODEL_LINESTYLE["IFS ens + NNI"],
    )
    ax.plot(
        wavelengths[mask],
        det_diff_psd[mask][::-1],
        label="DDPM det",
        color=MODEL_COLOR_DICT["DDPM det"],
        linestyle=MODEL_LINESTYLE["DDPM det"],
    )
    ax.plot(
        wavelengths[mask],
        ens_diff_psd[mask][::-1],
        label="DDPM ens",
        color=MODEL_COLOR_DICT["DDPM ens"],
        linestyle=MODEL_LINESTYLE["DDPM ens"],
    )
    ax.plot(
        wavelengths[mask],
        wrf_psd[mask][::-1],
        label="WRF",
        color=MODEL_COLOR_DICT["WRF"],
        linestyle=MODEL_LINESTYLE["WRF"],
    )
    ax.plot(
        wavelengths[mask],
        cpc_psd[mask][::-1],
        label="CombiPrecip",
        color=MODEL_COLOR_DICT["CombiPrecip"],
        linestyle=MODEL_LINESTYLE["CombiPrecip"],
    )

    LAMBDA_STAR = 2.3e2
    ax.axvline(LAMBDA_STAR, color="black", linestyle="--")
    ax.text(LAMBDA_STAR * 1.1, 1e-5, r"$\lambda^\star$", fontsize="large")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xticks([1, 10, 100])
    ax.set_xticklabels([r"$10^0$", r"$10^1$", r"$10^2$"])
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
    lead_time_name = det_s2s.lead_time[lead_time_idx + 1]
    fig.savefig(os.path.join(figs_dir, f"psds/psd_{lead_time_name}.png"), dpi=300)


def plot_rank_histogram(
    ens_s2s,
    det_diff,
    ens_diff,
    wrf,
    cpc,
    lead_time_idx,
    figs_dir,
):
    for event_idx in range(1, len(TIME_IDXS) // EVENT_LENGTH + 1):
        time_idxs = slice(EVENT_LENGTH * (event_idx - 1), EVENT_LENGTH * event_idx)

        ens_s2s_values = ens_s2s.precip[lead_time_idx + 1, :, time_idxs]
        det_diff_values = det_diff.precip[lead_time_idx, :, time_idxs]
        ens_diff_values = ens_diff.precip[lead_time_idx, :, time_idxs]
        wrf_values = wrf.precip[lead_time_idx, :, time_idxs]
        cpc_values = cpc.precip[time_idxs]

        # calculate ranks
        ens_s2s_ranks = rank_histogram(ens_s2s_values, cpc_values)
        det_diff_ranks = rank_histogram(det_diff_values, cpc_values)
        ens_diff_ranks = rank_histogram(ens_diff_values, cpc_values)
        wrf_ranks = rank_histogram(wrf_values, cpc_values)

        # plot rank histograms
        fig, ax = plt.subplots(2, 2, figsize=(8, 4), dpi=300)
        ax[0, 0].hist(
            ens_s2s_ranks,
            color=MODEL_COLOR_DICT["IFS ens + NNI"],
            histtype="step",
            hatch="//",
            density=True,
        )
        ax[0, 0].set_title("a) IFS ens + NNI")
        ax[0, 1].hist(det_diff_ranks, color=MODEL_COLOR_DICT["DDPM det"], density=True)
        ax[0, 1].set_title("b) DDPM det")
        ax[1, 0].hist(ens_diff_ranks, color=MODEL_COLOR_DICT["DDPM ens"], density=True)
        ax[1, 0].set_title("c) DDPM ens")
        ax[1, 1].hist(wrf_ranks, color=MODEL_COLOR_DICT["WRF"], density=True)
        ax[1, 1].set_title("d) WRF")

        # labels
        plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
        ax[1, 0].set_xlabel("Rank")
        ax[1, 1].set_xlabel("Rank")
        ax[0, 0].set_ylabel("Frequency")
        ax[1, 0].set_ylabel("Frequency")

        # save
        plt.tight_layout()
        lead_time_name = ens_s2s.lead_time[lead_time_idx + 1]
        fig.savefig(
            os.path.join(
                figs_dir, f"ranks/rank_histograms_{lead_time_name}_e{event_idx}.png"
            )
        )


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


def plot_avFSS_table(
    det_s2s,
    ens_s2s,
    det_diff,
    ens_diff,
    wrf,
    cpc,
    figs_dir,
):

    # define FSS parameters
    thresholds = [0.5, 1.0, 1.5, 2.0, 2.5]
    num_neighbors = [15]  # , 30, 45, 60]

    for event_idx in range(1, len(TIME_IDXS) // EVENT_LENGTH + 1):
        time_slice = slice((event_idx - 1) * EVENT_LENGTH, event_idx * EVENT_LENGTH)
        fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(8, 6))
        axes = axes.flatten()

        for lead_time_idx in range(4):
            lead_time_name = det_s2s.lead_time[lead_time_idx]
            print("avFSS -", lead_time_name)

            # get values for lead time
            det_s2s_values = np.expand_dims(
                det_s2s.precip[lead_time_idx, time_slice], axis=0
            )
            ens_s2s_values = ens_s2s.precip[lead_time_idx, :, time_slice]
            if lead_time_idx > 0:
                det_diff_values = det_diff.precip[lead_time_idx - 1, :, time_slice]
                ens_diff_values = ens_diff.precip[lead_time_idx - 1, :, time_slice]
                wrf_values = wrf.precip[lead_time_idx - 1, :, time_slice]
            cpc_values = cpc.precip[time_slice]

            # arrays for model and parameters
            det_s2s_avfss_arr = np.zeros((len(thresholds), len(num_neighbors)))
            ens_s2s_avfss_arr = np.zeros((len(thresholds), len(num_neighbors)))
            if lead_time_idx > 0:
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
                    if lead_time_idx > 0:
                        det_diff_avfss_arr[i_thr, i_num] = get_av_fss(
                            det_diff_values, cpc_values, thr, num
                        )
                        ens_diff_avfss_arr[i_thr, i_num] = get_av_fss(
                            ens_diff_values, cpc_values, thr, num
                        )
                        wrf_avfss_arr[i_thr, i_num] = get_av_fss(
                            wrf_values, cpc_values, thr, num
                        )

            if lead_time_idx == 0:
                model_arrs = [det_s2s_avfss_arr, ens_s2s_avfss_arr]
                model_labels = ["IFS det + NNI", "IFS ens + NNI"]
            else:
                model_arrs = [
                    det_s2s_avfss_arr,
                    ens_s2s_avfss_arr,
                    det_diff_avfss_arr,
                    ens_diff_avfss_arr,
                    wrf_avfss_arr,
                ]
                model_labels = [
                    "IFS det + NNI",
                    "IFS ens + NNI",
                    "DDPM det",
                    "DDPM ens",
                    "WRF",
                ]

            # plot avFSS vs. thresholds for each num_neighbors
            for i_num, num in enumerate(num_neighbors):
                for model_label, model_arr in zip(model_labels, model_arrs):
                    axes[lead_time_idx].plot(
                        thresholds,
                        model_arr[:, i_num],
                        label=model_label,
                        color=MODEL_COLOR_DICT[model_label],
                        linestyle=MODEL_LINESTYLE[model_label],
                    )
                # add legend and save
                axes[lead_time_idx].set_xlim(
                    thresholds[0] - 0.05, thresholds[-1] + 0.05
                )
                if lead_time_idx > 1:
                    axes[lead_time_idx].set_xlabel("Threshold (mm/hr)")
                    axes[lead_time_idx].set_xticks(thresholds)
                if lead_time_idx % 2 == 0:
                    axes[lead_time_idx].set_ylabel("avFSS")
                if lead_time_idx == 1:
                    axes[lead_time_idx].legend(loc="upper right")
                axes[lead_time_idx].set_title(lead_time_name)
                axes[lead_time_idx].spines["left"].set_position(
                    ("data", thresholds[0] - 0.05)
                )
                axes[lead_time_idx].set_yticks(np.arange(0.0, 0.3, 0.05))
                axes[lead_time_idx].set_ylim(0.0, 0.25)
                _make_arrows(axes[lead_time_idx], thresholds[0] - 0.05, 1)

        plt.tight_layout()
        fig.savefig(
            os.path.join(figs_dir, f"fss/fss_num{num}_e{event_idx}.png"),
            dpi=300,
        )
        plt.close()


def make_plots(
    climatology_path,
    s2s_det_path,
    s2s_ens_path,
    diff_det_path,
    diff_ens_path,
    wrf_path,
    cpc_path,
    num_idx,
):
    climatology = SurfaceData.load_from_h5(climatology_path, ["precip"])
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
    os.makedirs(figs_dir + "/dists", exist_ok=True)
    os.makedirs(figs_dir + "/psds", exist_ok=True)
    os.makedirs(figs_dir + "/ranks", exist_ok=True)
    os.makedirs(figs_dir + "/fss", exist_ok=True)

    # # plots, plot and plots #

    # plot maps
    plot_lead_time_agg_raw(
        climatology,
        det_s2s,
        cpc,
        figs_dir,
    )
    print("agg maps raw saved")

    # aggs of all
    for lead_time_idx in range(3):
        plot_lead_time_agg(
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

    # # plot gifs for each lead time
    # plot_lead_time_gifs_raw(
    #     det_s2s,
    #     cpc,
    #     figs_dir,
    # )
    # print("gif maps raw saved")

    # # plot gifs for each lead time
    # for lead_time_idx in range(3):
    #     plot_lead_time_map_complete(
    #         det_s2s,
    #         ens_s2s,
    #         det_diff,
    #         ens_diff,
    #         wrf,
    #         cpc,
    #         lead_time_idx,
    #         num_idx,
    #         figs_dir,
    #     )
    # print("maps complete saved")

    # # plot timeseries for each lead time (and each event)
    # for lead_time_idx in range(3):
    #     plot_lead_time_timeseries(
    #         det_s2s,
    #         ens_s2s,
    #         det_diff,
    #         ens_diff,
    #         wrf,
    #         cpc,
    #         lead_time_idx,
    #         figs_dir,
    #     )
    # print("timeseries saved")

    # # plot distribution for each lead time
    # for lead_time_idx in range(3):
    #     plot_lead_time_dists(
    #         det_s2s,
    #         ens_s2s,
    #         det_diff,
    #         ens_diff,
    #         wrf,
    #         cpc,
    #         lead_time_idx,
    #         figs_dir,
    #     )
    # print("distributions saved")

    # # plot psds for each lead time
    # for lead_time_idx in range(3):
    #     plot_lead_time_psd(
    #         det_s2s,
    #         ens_s2s,
    #         det_diff,
    #         ens_diff,
    #         wrf,
    #         cpc,
    #         lead_time_idx,
    #         figs_dir,
    #     )
    # print("psds saved")

    # # plot rank histogram for each lead time
    # for lead_time_idx in range(3):
    #     plot_rank_histogram(
    #         ens_s2s,
    #         det_diff,
    #         ens_diff,
    #         wrf,
    #         cpc,
    #         lead_time_idx,
    #         figs_dir,
    #     )
    # print("rank histograms saved")

    # # 0.1mm/h trim
    # all_datasets = [climatology, det_s2s, ens_s2s, det_diff, ens_diff, wrf, cpc]
    # for dataset in all_datasets:
    #     dataset.precip = np.where(
    #         (dataset.precip >= 0) & (dataset.precip < 0.1), 0.0, dataset.precip
    #     )

    # # plot avFSS for lead time
    # plot_avFSS_table(
    #     det_s2s,
    #     ens_s2s,
    #     det_diff,
    #     ens_diff,
    #     wrf,
    #     cpc,
    #     figs_dir,
    # )
    # print("avFSS saved")


def main():
    # directory paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(script_dir, "../dirs.toml"), "rb") as f:
        dirs = tomllib.load(f)
    base = dirs["main"]["base"]
    test_data_dir = os.path.join(base, dirs["subs"]["test"])
    simulations_dir = os.path.join(base, dirs["subs"]["simulations"])

    # extra configurations
    climatology_path = os.path.join(test_data_dir, "climatology.h5")
    s2s_det_path = os.path.join(test_data_dir, "det_s2s_nearest.h5")
    s2s_ens_path = os.path.join(test_data_dir, "ens_s2s_nearest.h5")
    weight = "heavy"
    cli = 50
    num_samples = 50
    diff_det_path = os.path.join(
        simulations_dir,
        "diffusion",
        f"det_{weight}_cli{cli}_ens{num_samples}.h5",
    )
    diff_ens_path = os.path.join(
        simulations_dir,
        "diffusion",
        f"ens_{weight}_cli{cli}_ens{num_samples}.h5",
    )
    wrf_path = os.path.join(
        simulations_dir,
        "wrf",
        "wrf.h5",
    )
    cpc_path = os.path.join(test_data_dir, "cpc.h5")

    # ensemble member for snapshots
    num_idx = 2

    # main calls
    make_plots(
        climatology_path,
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
