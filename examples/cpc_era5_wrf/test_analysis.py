import os
import xarray as xr
import matplotlib.pyplot as plt

from evaluation.plots import plot_maps, plot_cdfs, plot_psds
from evaluation.metrics import *
from utils import create_folder, get_spatial_lengths
    
    
def print_metrics(reference, forecasts, unit):
    metrics = [
        {
            "name": "MAE ({})".format(unit),
            "func": lambda ref, pred: mean_absolute_error(ref["data"], pred["data"])
        },
        {
            "name": "RMSE ({})".format(unit),
            "func": lambda ref, pred: root_mean_squared_error(ref["data"], pred["data"])
        },
        {
            "name": "logCDF-l2 (no units)",
            "func": lambda ref, pred: logcdf_distance(ref["data"], pred["data"], "l2")
        },
        {
            "name": "logCDF-max (no units)",
            "func": lambda ref, pred: logcdf_distance(ref["data"], pred["data"], "max")
        },
        {
            "name": "Perkins Skill Score (no units)",
            "func": lambda ref, pred: perkins_skill_score(ref["data"], pred["data"])
        },
        {
            "name": "CRPS ({})".format(unit),
            "func": lambda ref, pred: crps(ref["data"], pred["data"])
        },
        {
            "name": "logPSD-l2 (no units)",
            "func": lambda ref, pred: logpsd_distance(
                ref["data"], ref["x_length"], ref["y_length"], 
                pred["data"], pred["x_length"], pred["y_length"], 
                "l2"
            )
        },
        {
            "name": "logPSD-max (no units)",
            "func": lambda ref, pred: logpsd_distance(
                ref["data"], ref["x_length"], ref["y_length"], 
                pred["data"], pred["x_length"], pred["y_length"], 
                "max"
            )
        },
    ]

    for metric in metrics:
        print("+ "+metric["name"]+":")
        for forecast in forecasts:
            value = metric["func"](reference, forecast)
            print("  - {}: {:.3f}".format(forecast['name'], value))
        print("\n")


def main():
    # TODO: read this from example.yml
    raw_data_dir = "/work/FAC/FGSE/IDYST/tbeucler/downscaling/raw_data"
    raw_era5_ds = xr.open_dataset(os.path.join(raw_data_dir, "era5/precip/era5_tp_2021_06_28-30.nc"))
    
    data_dir = "/work/FAC/FGSE/IDYST/tbeucler/downscaling/mlima/data/test_data"
    cpc_ds = xr.open_dataset(os.path.join(data_dir, "cpc.h5"))
    wrf_ds = xr.open_dataset(os.path.join(data_dir, "wrf.h5"))
    era5_x25 = xr.open_dataset(os.path.join(data_dir, "era5_x25.h5"))
    era5_lowpass_ds = xr.open_dataset(os.path.join(data_dir, "era5_lowpass.h5"))
    era5_qm_ds = xr.open_dataset(os.path.join(data_dir, "era5_qm.h5"))
    
    # To plot
    raw_lons = raw_era5_ds.longitude.values
    raw_lats = raw_era5_ds.latitude.values
    raw_extent = (raw_lons.min(), raw_lons.max(), raw_lats.min(), raw_lats.max())
    
    lons = cpc_ds.longitude.values
    lats = cpc_ds.latitude.values
    extent = (lons.min(), lons.max(), lats.min(), lats.max())
    
    times = cpc_ds.time.values
    cpc_data = cpc_ds.precip.values
    wrf_data = wrf_ds.precip.values[12:60,:,:] # take out warm-up
    raw_era5_data = raw_era5_ds.tp.values[:48,::-1,:]*1000 # m/s to mm/h and reverse latitude
    lowpass_data = era5_lowpass_ds.precip.values[:48,:,:]
    qm_data = era5_qm_ds.precip.values[:48,:,:]
    
    # quick check on times
    wrf_times = wrf_ds.time.values[12:60]
    raw_era5_times = raw_era5_ds.time.values[:48]
    lowpass_times = era5_lowpass_ds.time.values[:48]
    qm_times = era5_qm_ds.time.values[:48]
    print("first time:")
    print("cpc:", times[0])
    print("wrf:", wrf_times[0])
    print("raw_era5:", raw_era5_times[0])
    print("lowpass:", lowpass_times[0])
    print("qm:", qm_times[0])
    print("last time:")
    print("cpc:", times[-1])
    print("wrf:", wrf_times[-1])
    print("raw_era5:", raw_era5_times[-1])
    print("lowpass:", lowpass_times[-1])
    print("qm:", qm_times[-1])
    print("\n")
    
    script_dir = os.path.dirname(os.path.realpath(__file__))
    figs_dir = os.path.join(script_dir, "figs/test")
    create_folder(figs_dir)
    
    # Plots
    time = 15
    print("Plot time:", times[time])
    
    # Maps
    arrays = (cpc_data[time,:,:], wrf_data[time,:,:], lowpass_data[time,:,:], qm_data[time,:,:])
    titles = ("CPC", "WRF", "ERA5 (low-pass)", "ERA5 (QM)")
    extents = (extent, extent, extent, extent)
    fig, _ = plot_maps(
        arrays,
        titles,
        extents,
        vmax=45 # as we're plotting an extreme convective event
    )
    fig.savefig(os.path.join(figs_dir, "maps_comparison.png"))
    
    # CDFs
    arrays = (cpc_data, wrf_data, lowpass_data, qm_data)
    labels = ("CPC", "WRF", "low-pass", "QM")
    cmap = plt.get_cmap('Dark2')
    colors = (cmap(2), cmap(0), cmap(3), cmap(1))
    fig, _ = plot_cdfs(arrays, labels, colors=colors)
    fig.savefig(os.path.join(figs_dir, "cdf_comparison.png"))
    
    # PSDs (all datasets have the same extent hence we can use the same x_length and y_length)
    lons = cpc_ds.longitude.values
    lats = cpc_ds.latitude.values
    x_length, y_length = get_spatial_lengths(lons, lats)
    spatial_lengths = ((x_length, y_length), (x_length, y_length), (x_length, y_length), (x_length, y_length))
    fig, _ = plot_psds(arrays, labels, spatial_lengths, colors=colors)
    fig.savefig(os.path.join(figs_dir, "psd_comparison.png"))
    
    
    # Make dictionaries to use print_metrics
    cpc = {
        "name": "CPC",
        "data": cpc_data,
        "x_length": x_length,
        "y_length": y_length
    }
    wrf = {
        "name": "WRF",
        "data": wrf_data,
        "x_length": x_length,
        "y_length": y_length
    }
    lowpass = {
        "name": "Low-pass",
        "data": lowpass_data,
        "x_length": x_length,
        "y_length": y_length
    }
    qm = {
        "name": "QM",
        "data": qm_data,
        "x_length": x_length,
        "y_length": y_length
    }
    
    # Print metrics
    reference = cpc
    forecasts = (wrf, lowpass, qm)
    metrics = ("MAE", "RMSE", "CDF-max", "CDF-l2", "Perkins Skill Score", "CRPS", "PSD-l2")
    print_metrics(reference, forecasts, "mm/h")
    

if __name__ == "__main__":
    main()