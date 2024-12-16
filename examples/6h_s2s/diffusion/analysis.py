import imageio, os, tomllib
import matplotlib.pyplot as plt
import numpy as np

from evaluation.plots import plot_maps, CURVE_CMAP as cmap
from data.surface_data import SurfaceData, ForecastSurfaceData, ForecastEnsembleSurfaceData
from engineering.spectrum import get_1dpsd
from utils import get_cdf


def plot_left_tale(
    bins_range,
    s2s_cdf,
    det_cdf,
    ens_cdf,
    cpc_cdf,
    lead_time_name,
    figs_dir,
):
    fig, ax = plt.subplots(figsize=(8, 4))

    ax.plot(bins_range, s2s_cdf, label="S2S det.", color=cmap(1), linewidth=2)
    ax.plot(bins_range, det_cdf, label="det-diff", color=cmap(5), linewidth=2)
    ax.plot(bins_range, ens_cdf, label="ens-diff", color=cmap(6), linewidth=2)
    ax.plot(bins_range, cpc_cdf, label="CombiPrecip", color=cmap(0), linewidth=2)
    
    ax.set_xlim(0, 5)
    ax.set_ylim(0, 1.0)
    ax.set_xlabel("Precipitation (mm/h)")
    ax.set_ylabel("Cumulative distribution function")

    ax.legend()
    plt.title(f"lead time = {lead_time_name}")
    fig.savefig(os.path.join(figs_dir, f"dist_{lead_time_name}_left.png"))


def plot_right_tale(
    bins_range,
    s2s_cdf,
    det_cdf,
    ens_cdf,
    cpc_cdf,
    lead_time_name,
    figs_dir,
):
    fig, ax = plt.subplots(figsize=(8, 5))

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(0.3, None)
    ax.set_ylim(0.8, 1.0)
    ax.set_xlabel("Precipitation (mm/h)")
    ax.set_ylabel("Cumulative distribution function")

    ax.plot(bins_range, s2s_cdf, label="S2S det.", color=cmap(1), linewidth=2)
    ax.plot(bins_range, det_cdf, label="det-diff", color=cmap(5), linewidth=2)
    ax.plot(bins_range, ens_cdf, label="ens-diff", color=cmap(6), linewidth=2)
    ax.plot(bins_range, cpc_cdf, label="CombiPrecip", color=cmap(0), linewidth=2)
    
    ax.legend()
    plt.title(f"lead time = {lead_time_name}")
    fig.savefig(os.path.join(figs_dir, f"dist_{lead_time_name}_right.png"))


def plot_lead_time_distribution(
    s2s, det, ens, cpc,
    lead_time_idx,
    figs_dir,
    bins=100,
):
    # get data
    cpc_data = cpc.precip.flatten()
    s2s_data = s2s.precip[lead_time_idx].flatten()
    det_data = det.precip[lead_time_idx].flatten()
    ens_data = ens.precip[lead_time_idx].flatten()

    # compute CDFs
    bins_range = np.linspace(0, 5, bins)
    s2s_cdf = get_cdf(s2s_data, bins_range)
    det_cdf = get_cdf(det_data, bins_range)
    ens_cdf = get_cdf(ens_data, bins_range)
    cpc_cdf = get_cdf(cpc_data, bins_range)
    
    # call left and right tale plots
    lead_time_name = s2s.lead_time[lead_time_idx].replace(" ", "-")
    plot_left_tale(bins_range, s2s_cdf, det_cdf, ens_cdf, cpc_cdf, lead_time_name, figs_dir)
    plot_right_tale(bins_range, s2s_cdf, det_cdf, ens_cdf, cpc_cdf, lead_time_name, figs_dir)


def plot_lead_time_map(
    s2s, det, ens, cpc,
    lead_time_idx,
    num_idx,
    time_idxs,
    figs_dir,
    event_length=8,
):
    lead_time_name = s2s.lead_time[lead_time_idx].replace(" ", "-")
    image_paths = []

    # save temporary images for each time_idx
    for time_idx in time_idxs:
        arrays = (
            s2s.precip[lead_time_idx, time_idx],
            det.precip[lead_time_idx, num_idx, time_idx],
            ens.precip[lead_time_idx, num_idx, time_idx],
            cpc.precip[time_idx],
        )
        titles = (
            "S2S det.",
            f"det-diff #{num_idx}",
            f"ens-diff #{num_idx}",
            "CombiPrecip"
        )
        cpc_extent = cpc.get_extent()
        extents = (cpc_extent,) * 4
        fig, _ = plot_maps(arrays, titles, extents)

        image_path = os.path.join(figs_dir, f"temp_map_{lead_time_name}_t{time_idx}.png")
        fig.savefig(image_path)
        plt.close(fig) # important to avoid memory leak
        image_paths.append(image_path)

    # create the GIFs
    for event_idx in (1, 2):
        gif_path = os.path.join(figs_dir, f"maps_{lead_time_name}_e{event_idx}.gif")
        with imageio.get_writer(gif_path, mode='I', duration=1000) as writer:
            for image_path_idx in range((event_idx-1)*event_length, event_idx*event_length):
                image_path = image_paths[image_path_idx]
                image = imageio.v2.imread(image_path)
                writer.append_data(image)

    # clean up temporary files
    for image_path in image_paths:
        os.remove(image_path)

        
def plot_lead_time_timeseries(
    s2s, det, ens, cpc,
    lead_time_idx,
    figs_dir,
    event_length=8,
):
    # get timeseries
    s2s_timeseries = np.mean(s2s.precip[lead_time_idx], axis=(-1, -2))
    det_timeseries = np.mean(det.precip[lead_time_idx], axis=(-1, -2))
    ens_timeseries = np.mean(ens.precip[lead_time_idx], axis=(-1, -2))
    cpc_timeseries = np.mean(cpc.precip, axis=(-1, -2))
    dates = cpc.time
    
    # compute ensemble mean and spread
    det_mean = np.mean(det_timeseries, axis=0)
    ens_mean = np.mean(ens_timeseries, axis=0)
    # use std for spread
    det_std = np.std(det_timeseries, axis=0)
    ens_std = np.std(ens_timeseries, axis=0)
    # clip negative values
    det_lower_bound = np.maximum(det_mean - det_std, 0)
    det_upper_bound = np.maximum(det_mean + det_std, 0)
    ens_lower_bound = np.maximum(ens_mean - ens_std, 0)
    ens_upper_bound = np.maximum(ens_mean + ens_std, 0)

    # plot timeseries for each event
    for event in (1, 2):
        idxs = slice(event_length * (event - 1), event_length * event)
        
        # create figure and axis
        fig, ax = plt.subplots(figsize=(8, 4))
        
        # plot deterministic and reference data
        ax.plot(dates[idxs], cpc_timeseries[idxs], label="CombiPrecip", color=cmap(0))
        ax.plot(dates[idxs], s2s_timeseries[idxs], label="S2S det.", color=cmap(1))
        
        # add shaded region for ensemble spread
        ax.fill_between(
            dates[idxs],
            det_lower_bound[idxs],
            det_upper_bound[idxs],
            color=cmap(5),
            alpha=0.3,
            label="det-diff",
        )
        ax.fill_between(
            dates[idxs],
            ens_lower_bound[idxs],
            ens_upper_bound[idxs],
            color=cmap(6),
            alpha=0.3,
            label="ens-diff",
        )
        
        # add legend and save
        ax.set_xlabel("Dates")
        ax.set_ylabel("Mean precipitation (mm/h)")
        plt.legend()
        
        lead_time_name = s2s.lead_time[lead_time_idx].replace(" ", "-")
        plt.title(f"lead time = {lead_time_name}, event = {event}")
        fig.savefig(os.path.join(figs_dir, f"ts_{lead_time_name}_e{event}.png"))
        
        
def plot_lead_time_psd(
    s2s, det, ens, cpc,
    lead_time_idx,
    figs_dir,
):
    spatial_lenghts = s2s.get_spatial_lengths()
    k, s2s_psd = get_1dpsd(s2s.precip[lead_time_idx], *spatial_lenghts)
    _, det_psd = get_1dpsd(np.mean(det.precip[lead_time_idx], axis=0), *spatial_lenghts)
    _, ens_psd = get_1dpsd(np.mean(ens.precip[lead_time_idx], axis=0), *spatial_lenghts)
    _, cpc_psd = get_1dpsd(cpc.precip, *spatial_lenghts)
    wavelengths = 2*np.pi / k[::-1]
    
    # get nyquist wavelnegths
    nyquist_wavelngths = 2*np.pi / (2 * spatial_lenghts[0]), 2*np.pi / (2 * spatial_lenghts[1])
    
    # mask wavelengths above nyquist
    mask = wavelengths > max(nyquist_wavelngths)
    
    fig, ax = plt.subplots(figsize=(8, 5))
    
    ax.plot(wavelengths[mask], s2s_psd[mask][::-1], label="S2S det.", color=cmap(1))
    ax.plot(wavelengths[mask], det_psd[mask][::-1], label="det-diff", color=cmap(5))
    ax.plot(wavelengths[mask], ens_psd[mask][::-1], label="ens-diff", color=cmap(6))
    ax.plot(wavelengths[mask], cpc_psd[mask][::-1], label="CombiPrecip", color=cmap(0))
    
    ax.set_xscale("log")
    ax.set_yscale("log")    
    ax.set_xlabel("Wavelengths (km)")
    ax.set_ylabel("Power spectral density")
    ax.legend()
    
    lead_time_name = s2s.lead_time[lead_time_idx].replace(" ", "-")
    plt.title(f"lead time = {lead_time_name}")
    fig.savefig(os.path.join(figs_dir, f"psd_{lead_time_name}.png"))


def make_plots(s2s_path, det_path, ens_path, cpc_path, time_idxs, num_idx):
    s2s = ForecastSurfaceData.load_from_h5(s2s_path, ["precip"])
    det = ForecastEnsembleSurfaceData.load_from_h5(det_path, ["precip"])
    ens = ForecastEnsembleSurfaceData.load_from_h5(ens_path, ["precip"])
    cpc = SurfaceData.load_from_h5(cpc_path, ["precip"])

    # define figs directory
    script_dir = os.path.dirname(os.path.realpath(__file__))
    figs_dir = os.path.join(script_dir, "analysis_figs")
    os.makedirs(figs_dir, exist_ok=True)
    
    # # plot gifs for each lead time (and each event)
    # for lead_time_idx in range(3):
    #     plot_lead_time_map(
    #         s2s, det, ens, cpc,
    #         lead_time_idx,
    #         num_idx,
    #         time_idxs,
    #         figs_dir,
    #     )
    # print("maps saved")
    
    # plot timeseries for each lead time (and each event)
    for lead_time_idx in range(3):
        plot_lead_time_timeseries(
            s2s, det, ens, cpc,
            lead_time_idx,
            figs_dir,
        )
    print("timeseries saved")
    
    # # plot distribution for each lead time
    # for lead_time_idx in range(3):
    #     plot_lead_time_distribution(
    #         s2s, det, ens, cpc,
    #         lead_time_idx,
    #         figs_dir,
    #     )
    # print("distributions saved")
    
    # # plot psds for each lead time
    # for lead_time_idx in range(3):
    #     plot_lead_time_psd(
    #         s2s, det, ens, cpc,
    #         lead_time_idx,
    #         figs_dir,
    #     )
    # print("psds saved")


def main():
    # directory paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(script_dir, "../dirs.toml"), "rb") as f:
        dirs = tomllib.load(f)
    base = dirs["main"]["base"]
    test_data_dir = os.path.join(base, dirs["subs"]["test"])
    simulations_dir = os.path.join(base, dirs["subs"]["simulations"])
    
    # extra configurations
    s2s_path = os.path.join(test_data_dir, "det_s2s_nearest.h5")
    det_path = os.path.join(simulations_dir, "diffusion/det_light_longer_cli100_ens50.h5")
    ens_path = os.path.join(simulations_dir, "diffusion/ens_light_longer_cli100_ens50.h5")
    cpc_path = os.path.join(test_data_dir,"cpc.h5")
    # time_idx for snapshots
    time_idxs = [i for i in range(16)]
    # ensemble member for snapshots
    num_idx = 25

    # main call    
    make_plots(s2s_path, det_path, ens_path, cpc_path, time_idxs, num_idx)
    

    
if __name__ == "__main__":
    main()
