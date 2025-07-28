import os

import matplotlib.pyplot as plt
import numpy as np
import tomllib
from configs.single import get_config

from data.surface_data import SurfaceData
from evaluation.plots import CURVE_CMAP, plot_maps

# limamau: take out hard coding
INITIAL_IDX = 0
EVENT_LENGTH = 8
S2S_COLOR = CURVE_CMAP(2)
WRF_COLOR = CURVE_CMAP(3)
CPC_COLOR = CURVE_CMAP(0)
S2S_DIVIDER = 6


def save_maps(wrf, s2s, cpc, lead_time_idx, member_idx, figs_dir):
    for time_idx in range(INITIAL_IDX, len(wrf.time) + INITIAL_IDX):
        wrf_precip = wrf.precip[time_idx] / S2S_DIVIDER
        arrays = (
            wrf_precip,
            s2s.precip[lead_time_idx, member_idx, time_idx],
            cpc.precip[time_idx],
        )
        titles = ("WRF", "S2S", "CombiPrecip")
        cpc_extent = cpc.get_extent()
        extents = (cpc_extent,) * len(arrays)
        fig, _ = plot_maps(arrays, titles, extents)
        os.makedirs(figs_dir, exist_ok=True)
        fig.savefig(os.path.join(figs_dir, f"maps_{time_idx}.png"))


def save_timeseries(wrf, s2s, cpc, lead_time_idx, member_idx, figs_dir):
    # time idxs
    time_idxs = slice(INITIAL_IDX, len(wrf.time) + INITIAL_IDX)

    # get timeseries
    s2s_timeseries = np.mean(
        s2s.precip[lead_time_idx, member_idx, time_idxs],
        axis=(-1, -2),
    )
    wrf_timeseries = np.mean(
        wrf.precip[time_idxs],
        axis=(-1, -2),
    )
    cpc_timeseries = np.mean(
        cpc.precip[time_idxs],
        axis=(-1, -2),
    )
    dates = cpc.time[time_idxs]

    # plot timeseries for each event
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(dates, s2s_timeseries, label="S2S det.", color=S2S_COLOR)
    ax.plot(dates, wrf_timeseries, label="WRF", color=WRF_COLOR)
    ax.plot(dates, cpc_timeseries, label="CombiPrecip", color=CPC_COLOR)
    ax.set_xlabel("Dates")
    ax.set_ylabel("Mean precipitation (mm/h)")
    plt.legend()
    fig.savefig(os.path.join(figs_dir, "timeseries.png"))


def run_analysis(wrf, s2s, cpc, lead_time_idx, member_idx):
    # dir to save figs
    script_dir = os.path.dirname(os.path.realpath(__file__))
    figs_dir = os.path.join(script_dir, "figs", "analysis")

    # maps
    save_maps(wrf, s2s, cpc, lead_time_idx, member_idx, figs_dir)

    # timeseries
    save_timeseries(wrf, s2s, cpc, lead_time_idx, member_idx, figs_dir)


def main():
    # directory paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(script_dir, "../dirs.toml"), "rb") as f:
        dirs = tomllib.load(f)
    base = dirs["main"]["base"]
    simulations_dir = os.path.join(base, dirs["subs"]["simulations"])
    test_data_dir = os.path.join(base, dirs["subs"]["test"])

    # extra configurations
    config = get_config()
    cpc_file = os.path.join(test_data_dir, config.cpc_file)
    cpc = SurfaceData.load_from_h5(cpc_file, ["precip"])
    s2s_file = os.path.join(test_data_dir, config.s2s_file)
    s2s = SurfaceData.load_from_h5(s2s_file, ["precip"])
    wrf_simulations_dir = os.path.join(simulations_dir, config.single_wrf_simulations)
    wrf = SurfaceData.load_from_h5(
        os.path.join(
            wrf_simulations_dir,
            "{}_{}.h5".format(config.forecast_date, config.member_idx),
        ),
        ["precip"],
    )
    member_idx = config.member_idx
    lead_time_idx = config.lead_time_idx

    # main calls
    run_analysis(wrf, s2s, cpc, lead_time_idx, member_idx)


if __name__ == "__main__":
    main()
