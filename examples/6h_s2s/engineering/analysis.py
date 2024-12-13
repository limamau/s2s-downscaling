import os, tomllib
import numpy as np
import matplotlib.pyplot as plt

from data.surface_data import SurfaceData, ForecastSurfaceData, ForecastEnsembleSurfaceData
from evaluation.plots import plot_maps, CURVE_CMAP as cmap


def plot_lead_time_distribution(
    cpc,
    lowpass_det_s2s,
    lowpass_ens_s2s,
    lead_time_idx,
    bins=100,
):
    # get data
    cpc_data = cpc.precip
    det_s2s_data = lowpass_det_s2s.precip[lead_time_idx]
    ens_s2s_data = lowpass_ens_s2s.precip[lead_time_idx]
    
    # create figure and axis
    fig, ax = plt.subplots(figsize=(8, 4))
    
    # plot histograms
    ax.hist(
        cpc_data.flatten(), bins=bins, alpha=0.5, label="CombiPrecip", color=cmap(0),
        density=True,
    )
    ax.hist(
        det_s2s_data.flatten(), bins=bins, alpha=0.5, label="S2S det.", color=cmap(1),
        density=True,
    )
    ax.hist(
        ens_s2s_data.flatten()/len(lowpass_ens_s2s.number), 
        bins=bins, alpha=0.5, label="S2S ens.", color=cmap(3),
        density=True,
    )
    
    # set axis limits and axis labels
    ax.set_xlim(0, 5)
    ax.set_ylim(0, 10)
    ax.set_xlabel("Precipitation (mm/h)")
    ax.set_ylabel("Density")
    
    
    # add legend and save
    plt.legend()
    script_dir = os.path.dirname(os.path.realpath(__file__))
    figs_dir = os.path.join(script_dir, "figs")
    os.makedirs(figs_dir, exist_ok=True)
    fig.savefig(os.path.join(figs_dir, f"distribution_lt{lead_time_idx}.png"))
        

def plot_lead_time_map(
    lowpass_ens_s2s,
    lowpass_det_s2s,
    cpc,
    lead_time_idx,
    number_idx1,
    number_idx2,
    plot_time_idx,
):
    # Plot maps
    arrays = (
        lowpass_ens_s2s.precip[lead_time_idx, number_idx1, plot_time_idx],
        lowpass_ens_s2s.precip[lead_time_idx, number_idx2, plot_time_idx],
        lowpass_det_s2s.precip[lead_time_idx, plot_time_idx],
        cpc.precip[plot_time_idx],
    )
    titles = (
        f"S2S ens. #{number_idx1}",
        f"S2S ens. #{number_idx2}",
        "S2S det.",
        "CombiPrecip"
    )
    cpc_extent = cpc.get_extent()
    extents = (cpc_extent,) * 4 # here we're cuttign s2s on the plot
    fig, _ = plot_maps(arrays, titles, extents)
    script_dir = os.path.dirname(os.path.realpath(__file__))
    figs_dir = os.path.join(script_dir, "figs")
    os.makedirs(figs_dir, exist_ok=True)
    lead_time_name = lowpass_ens_s2s.lead_time[lead_time_idx]
    # TODO: fix lead time name
    fig.savefig(os.path.join(figs_dir, f"maps_lt{lead_time_idx}.png"))
        
        
def plot_lead_time_timeseries(
    cpc,
    det_s2s,
    ens_s2s,
    lead_time_idx,
    event_length=8,
):
    # get timeseries
    cpc_timeseries = np.mean(cpc.precip, axis=(-1, -2))
    det_s2s_timeseries = np.mean(det_s2s.precip[lead_time_idx], axis=(-1, -2))
    ens_s2s_timeseries = np.mean(ens_s2s.precip[lead_time_idx], axis=(-1, -2))
    dates = cpc.time
    
    # compute ensemble mean and spread
    ens_mean = np.mean(ens_s2s_timeseries, axis=0)
    ens_std = np.std(ens_s2s_timeseries, axis=0)  # use std for spread
    lower_bound = np.maximum(ens_mean - ens_std, 0)  # clip lower bound at zero
    upper_bound = np.maximum(ens_mean + ens_std, 0)

    # plot timeseries for each event
    for event in (1, 2):
        idxs = slice(event_length * (event - 1), event_length * event)
        
        # create figure and axis
        fig, ax = plt.subplots(figsize=(8, 4))
        
        # plot deterministic and reference data
        ax.plot(dates[idxs], cpc_timeseries[idxs], label="CombiPrecip", color=cmap(0))
        ax.plot(dates[idxs], det_s2s_timeseries[idxs], label="S2S det.", color=cmap(1))
        
        # plot ensemble mean
        ax.plot(dates[idxs], ens_mean[idxs], label="S2S ens. mean", color=cmap(3))
        
        # add shaded region for ensemble spread
        ax.fill_between(
            dates[idxs],
            lower_bound[idxs],  # lower bound clipped at zero
            upper_bound[idxs],  # upper bound
            color=cmap(3),
            alpha=0.3,
        )
        
        # add legend and save
        plt.legend()
        script_dir = os.path.dirname(os.path.realpath(__file__))
        figs_dir = os.path.join(script_dir, "figs")
        os.makedirs(figs_dir, exist_ok=True)
        fig.savefig(os.path.join(figs_dir, f"timeseries_lt{lead_time_idx}_e{event}.png"))


def make_plots(
    test_data_dir, 
    plot_time_idx,
    lead_time_idxs,
    number_idx1,
    number_idx2,
):
    # cpc data
    cpc_file = os.path.join(test_data_dir, "cpc.h5")
    cpc = SurfaceData.load_from_h5(cpc_file, ["precip"])
    # deterministic data
    det_s2s_file = os.path.join(test_data_dir, "det_s2s.h5")
    det_s2s = ForecastSurfaceData.load_from_h5(det_s2s_file, ["precip"])
    nearest_s2s_file = os.path.join(test_data_dir, "det_s2s_nearest.h5")
    nearest_s2s = ForecastSurfaceData.load_from_h5(nearest_s2s_file, ["precip"])
    lowpass_det_s2s_file = os.path.join(test_data_dir, "det_s2s_nearest_low-pass.h5")
    lowpass_det_s2s = ForecastSurfaceData.load_from_h5(lowpass_det_s2s_file, ["precip"])
    # ensemble data
    ens_s2s_file = os.path.join(test_data_dir, "ens_s2s.h5")
    ens_s2s = ForecastEnsembleSurfaceData.load_from_h5(ens_s2s_file, ["precip"])
    nearest_ens_s2s_file = os.path.join(test_data_dir, "ens_s2s_nearest.h5")
    nearest_ens_s2s = ForecastEnsembleSurfaceData.load_from_h5(nearest_ens_s2s_file, ["precip"])
    lowpass_ens_s2s_file = os.path.join(test_data_dir, "ens_s2s_nearest_low-pass.h5")
    lowpass_ens_s2s = ForecastEnsembleSurfaceData.load_from_h5(lowpass_ens_s2s_file, ["precip"])
    
    # check times
    print("Plotting time: ", cpc.time[plot_time_idx])
    print("Plotting time: ", lowpass_det_s2s.time[plot_time_idx])
    print("Plotting time: ", lowpass_ens_s2s.time[plot_time_idx])
    
    # plot maps for each lead time
    for lead_time_idx in lead_time_idxs:
        plot_lead_time_map(
            lowpass_ens_s2s,
            lowpass_det_s2s,
            cpc,
            lead_time_idx,
            number_idx1,
            number_idx2,
            plot_time_idx,
        )
    print("maps saved")
    
    # plot timeseries for each lead time (and each event)
    for lead_time_idx in lead_time_idxs:
        plot_lead_time_timeseries(
            cpc,
            lowpass_det_s2s,
            lowpass_ens_s2s,
            lead_time_idx,
        )
    print("timeseries saved")
    
    # plot distribution for each lead time (and each event)
    for lead_time_idx in lead_time_idxs:
        plot_lead_time_distribution(
            cpc,
            lowpass_det_s2s,
            lowpass_ens_s2s,
            lead_time_idx,
        )
    print("distributions saved")
    
    
def main():
    # directory paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(script_dir, "../dirs.toml"), "rb") as f:
        dirs = tomllib.load(f)
    base = dirs["main"]["base"]
    test_data_dir = os.path.join(base, dirs["subs"]["test"])
    
    # extra configurations
    plot_time_idx = 2
    lead_time_idx = [0, 1, 2]
    number_idx1 = 25
    number_idx2 = 30
    
    # main call
    make_plots(test_data_dir, plot_time_idx, lead_time_idx, number_idx1, number_idx2)


if __name__ == "__main__":
    main()
