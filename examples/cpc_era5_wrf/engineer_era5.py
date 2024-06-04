import os
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

from engineering.spectrum import *
from engineering.regridding import *
from models.quantile_mapping import *
from evaluation.plots import plot_cdfs, plot_maps, plot_psds
from utils import *


def plot_timeseries(times, era5_data, cpc_data, mean_title, var_title, figs_dir, mean_filename, var_filename, qm_data=None):
    # Mean
    _, ax = plt.subplots(1, 1, figsize=(10, 5))
    
    era5_ts = np.mean(era5_data, axis=(1,2))
    cpc_ts = np.mean(cpc_data, axis=(1,2))
    plt.plot(times, era5_ts, label='ERA5', color='red')
    plt.plot(times, cpc_ts, label='CombiPrecip', color='blue')
    
    if qm_data is not None:
        qm_ts = np.mean(qm_data, axis=(1,2))
        plt.plot(times, qm_ts, label='QM', color='green')
    
    ax.set_title(mean_title)
    ax.set_ylabel('Precipitation (mm/h)')
    ax.set_xlabel('Time')
    plt.legend()
    plt.savefig(os.path.join(figs_dir, mean_filename))
    
    # Variance
    _, ax = plt.subplots(1, 1, figsize=(10, 5))
    era5_ts = np.var(era5_data, axis=(1,2))
    cpc_ts = np.var(cpc_data, axis=(1,2))
    plt.plot(times, era5_ts, label='ERA5', color='red')
    plt.plot(times, cpc_ts, label='CombiPrecip', color='blue')
    
    if qm_data is not None:
        qm_ts = np.var(qm_data, axis=(1,2))
        plt.plot(times, qm_ts, label='QM', color='green')
    
    ax.set_title(var_title)
    ax.set_ylabel('Precipitation (mm/h)²')
    ax.set_xlabel('Time')
    plt.legend()
    plt.savefig(os.path.join(figs_dir, var_filename))


# Here we redefine plot_psds to include the lambda_star and psd_star values.
# Moreover, as we already calculated the PSDs, we can directly plot them!
def plot_psds(era5_wavelengths, era5_psd, cpc_wavelengths, cpc_psd, figs_dir, filename, threshold=1e3, lambda_star=None, psd_star=None):
    plt.figure(figsize=(8, 6))
    
    mask = era5_psd < threshold
    plt.loglog(era5_wavelengths[mask], era5_psd[mask], label="ERA5 (x25)", color='red')
    mask = cpc_psd < threshold
    plt.loglog(cpc_wavelengths[mask], cpc_psd[mask], label="CombiPrecip", color='blue')
    
    if lambda_star is not None and psd_star is not None:
        plt.axvline(x=lambda_star, linestyle='--', color='grey')
        plt.text(lambda_star*1.01, psd_star*1.3, r'$\lambda^\star$', fontsize=14, color='grey')
    
    plt.xlabel(r"Wavelength $(km$)")
    plt.ylabel("PSD")
    plt.legend(fontsize='large')
    plt.savefig(os.path.join(figs_dir, filename))
    
    
def plot_psds_complete(
    raw_era5_wavelengths,
    raw_era5_psd,
    era5_x25_wavelengths,
    era5_x25_psd,
    lowpass_era5_wavelengths,
    lowpass_era5_psd,
    era5_qm_wavelengths,
    era5_qm_psd,
    cpc_wavelengths,
    cpc_psd,
    figs_dir,
    filename,
    cutoff,
    threshold=100,
):
    plt.figure(figsize=(8, 6))
    mask = raw_era5_psd < threshold
    plt.loglog(raw_era5_wavelengths[mask], raw_era5_psd[mask], label="ERA5 (raw)", color='red', linestyle=':')
    mask = era5_x25_psd < threshold
    plt.loglog(era5_x25_wavelengths[mask], era5_x25_psd[mask], label="ERA5 (x25)", color='red', linestyle='--')
    mask = lowpass_era5_psd < threshold
    plt.loglog(lowpass_era5_wavelengths[mask], lowpass_era5_psd[mask], label="ERA5 (low-pass)", color='red', linestyle='-.')
    mask = era5_qm_psd < threshold
    plt.loglog(era5_qm_wavelengths[mask], era5_qm_psd[mask], label="ERA5 (QM)", color='red', linestyle='-')
    mask = cpc_psd < threshold
    plt.loglog(cpc_wavelengths[mask], cpc_psd[mask], label="CombiPrecip", color='blue')
    plt.axvline(x=2*np.pi/cutoff, linestyle='--', color='grey')
    plt.text(2*np.pi/cutoff*1.01, 10e-3, 'cutoff', fontsize=14, color='grey')
    plt.xlabel(r"Wavelength $(km$)")
    plt.ylabel("PSD")
    plt.legend(fontsize='large')
    plt.savefig(os.path.join(figs_dir, filename))
    plt.show()
    
    
def plot_parseval(data, data_ps_t, data_ps_2d_t, figs_dir, filename):
    norm_data = normalize_data(data, sigma_data=0.5)[0]
    norm_data_var_t = np.mean((norm_data - np.mean(norm_data))**2, axis=(1,2))
    ps_var_t = np.sum(data_ps_t, axis=0)
    ps_2d_var_t = np.sum(data_ps_2d_t, axis=(1,2))
    
    print("Variance from normal space: {:.3f}".format(np.mean(norm_data_var_t)))
    print("Variance from wavenumber: {:.3f}".format(np.mean(ps_var_t)))
    print("Variance from wavenumber (2D): {:.3f}".format(np.mean(ps_2d_var_t)))
    
    plt.figure(figsize=(8, 6))
    plt.plot(norm_data_var_t, label="using data", color='blue')
    plt.plot(ps_var_t, label="using ps", color='red')
    plt.plot(ps_2d_var_t, label="using ps", color='green')
    plt.xlabel("Hours since t0")
    plt.ylabel("Variance")
    plt.legend(fontsize='large')
    plt.savefig(os.path.join(figs_dir, filename))
    

def main():
    ##################
    ######## Train set
    ##################
    
    # File paths
    raw_data_dir = "/work/FAC/FGSE/IDYST/tbeucler/downscaling/raw_data"
    raw_era5_file = os.path.join(raw_data_dir, "era5/precip/era5_tp_2024_01.nc")
    train_data_dir = "/work/FAC/FGSE/IDYST/tbeucler/downscaling/mlima/data/train_data"
    cpc_file = os.path.join(train_data_dir, "cpc.h5")
    
    # Read and plot original datasets
    raw_era5_ds = xr.open_dataset(raw_era5_file)
    
    # To plot
    time = 70 # (30: ok, 70: messy)
    print("Plot time:", raw_era5_ds.time.values[time])
    script_dir = os.path.dirname(os.path.realpath(__file__))
    figs_dir = os.path.join(script_dir, "figs/engineering")
    os.makedirs(figs_dir, exist_ok=True)
    os.makedirs(os.path.join(figs_dir, "maps"), exist_ok=True)
    os.makedirs(os.path.join(figs_dir, "psds"), exist_ok=True)
    os.makedirs(os.path.join(figs_dir, "timeseries"), exist_ok=True)
    raw_era5_lons = raw_era5_ds.longitude.values
    raw_era5_lats = raw_era5_ds.latitude.values
    raw_era5_extent = (np.min(raw_era5_lons), np.max(raw_era5_lons), np.min(raw_era5_lats), np.max(raw_era5_lats))
    
    # Read info from already preprocessed CPC data
    cpc_ds = xr.open_dataset(cpc_file, engine='h5netcdf')
    cpc_data = cpc_ds.precip.values
    new_lons = cpc_ds.longitude.values
    new_lats = cpc_ds.latitude.values
    times = cpc_ds.time.values
    extent = (np.min(new_lons), np.max(new_lons), np.min(new_lats), np.max(new_lats))
    
    
    # Interpolate ERA5
    ##################
    era5_data = regrid_era5(raw_era5_ds, times, new_lats, new_lons)
    train_era5_x25_data = era5_data
    
    # Plot maps
    arrays = (era5_data[time,:,:], cpc_data[time,:,:])
    titles = ("ERA5 (x25)", "CombiPrecip")
    extents = (extent, extent)
    fig, _ = plot_maps(arrays, titles, extents)
    fig.savefig(os.path.join(figs_dir, "maps/era5_x25_map.png"))
    
    # Plot timeseries
    plot_timeseries(
        times=times,
        era5_data=era5_data,
        cpc_data=cpc_data,
        mean_title="Mean precipitation time series",
        var_title="Variance precipitation time series",
        figs_dir=os.path.join(figs_dir, "timeseries"),
        mean_filename="mean_precip_timeseries.png",
        var_filename="var_precip_timeseries.png",
    )
    
    # Calculate PSD
    x_length, y_length = get_spatial_lengths(new_lons, new_lats)
    era5_x25_wavelengths, era5_x25_psd = get_psd(era5_data, x_length, y_length)
    
    
    # Apply low pass filter
    #######################
    
    # Get PSD from raw ERA5 and processed CombiPrecip
    x_length, y_length = get_spatial_lengths(raw_era5_lons, raw_era5_lats)
    raw_era5_wavelengths, raw_era5_psd = get_psd(raw_era5_ds.tp.values*1000, x_length, y_length)
    x_length, y_length = get_spatial_lengths(new_lons, new_lats)
    cpc_wavelengths, cpc_psd = get_psd(cpc_data, x_length, y_length)
    
    # Check wavelengths
    print("ERA5 wavelengths: [{:.3f}, {:.3f}]".format(np.min(raw_era5_wavelengths), np.max(raw_era5_wavelengths)))
    print("length", len(raw_era5_wavelengths))
    print("CombiPrecip wavelengths: [{:.3f}, {:.3f}]".format(np.min(cpc_wavelengths), np.max(cpc_wavelengths)))
    print("length:", len(cpc_wavelengths))
    
    # Parseval check
    _, cpc_ps_t, cpc_ps_2d_t = get_psd(cpc_data, x_length, y_length, parseval_check=True)
    plot_parseval(
        data=cpc_data,
        data_ps_t=cpc_ps_t,
        data_ps_2d_t=cpc_ps_2d_t,
        figs_dir=figs_dir,
        filename="parseval.png",
    )
    
    # Apply low-pass filter
    era5_x25_wavelengths, era5_x25_psd = get_psd(era5_data, x_length, y_length)
    nyquist_lowres = 2*np.pi/np.min(raw_era5_wavelengths)
    print("nyquist low res.: {:.3f}".format(nyquist_lowres))
    nyquist_highres = 2*np.pi/np.min(era5_x25_wavelengths)
    print("nyquist high res.: {:.3f}".format(nyquist_highres))
    cutoff = np.sqrt(nyquist_lowres*nyquist_highres)
    cutoff = 0.8*nyquist_lowres
    print("cutoff: {:.3f}".format(cutoff))
    era5_data = apply_low_pass_filter(era5_data, cutoff, x_length, y_length)
    train_era5_data_lowpass = era5_data
    arrays = (era5_data[time,:,:], cpc_data[time,:,:])
    titles = ("ERA5 (low-pass filter)", "CombiPrecip")
    fig, _ = plot_maps(arrays, titles, extents)
    fig.savefig(os.path.join(figs_dir, "maps/lowpass_map.png"))
    
    # Calculate PSD
    era5_lowpass_wavelengths, era5_lowpass_psd = get_psd(era5_data, x_length, y_length)
    
    
    # Correct bias with quantile mapping
    ####################################
    era5_data = quantile_mapping(cpc_data, era5_data, era5_data)
    train_era5_qm_data = era5_data
    
    # Plot maps
    arrays = (era5_data[time,:,:], cpc_data[time,:,:])
    titles = ("ERA5 (QM)", "CombiPrecip")
    fig, _ = plot_maps(arrays, titles, extents)
    fig.savefig(os.path.join(figs_dir, "maps/qm_map.png"))
    
    # Plot timeseries
    plot_timeseries(
        times=times,
        era5_data=train_era5_x25_data,
        cpc_data=cpc_data,
        mean_title="Mean precipitation time series (with QM)",
        var_title="Variance precipitation time series (with QM)",
        figs_dir=os.path.join(figs_dir, "timeseries"),
        mean_filename="mean_precip_timeseries_qm.png",
        var_filename="var_precip_timeseries_qm.png",
        qm_data=era5_data,
    )
    train_era5_qm_data = era5_data
    
    # Plots
    arrays = (cpc_data, train_era5_data_lowpass, train_era5_qm_data)
    labels = ("CombiPrecip", "ERA5 (low-pass)", "ERA5 (QM)")
    fig, _ = plot_cdfs(arrays, labels)
    fig.savefig(os.path.join(figs_dir, "cdfs.png"))
    
    # Calculate PSD
    era5_qm_wavelengths, era5_qm_psd = get_psd(era5_data, x_length, y_length)
    
    # Plot PSDs:
    # raw
    plot_psds(
        era5_wavelengths=raw_era5_wavelengths,
        era5_psd=raw_era5_psd,
        cpc_wavelengths=cpc_wavelengths,
        cpc_psd=cpc_psd,
        figs_dir=os.path.join(figs_dir, "psds"),
        filename="psd_raw.png",
    )
    
    # x25
    print("x25:")
    lambda_star, psd_star = find_intersection(cpc_wavelengths, era5_x25_psd, cpc_psd)
    Ny, Nx = len(new_lats), len(new_lons)
    sigma_star = np.sqrt(psd_star * Nx*Ny)
    print("lambda star: {:.2f}".format(lambda_star))
    print("psd star: {:.2e}".format(psd_star))
    print("sigma star: {:.2e}".format(sigma_star))
    plot_psds(
        era5_wavelengths=era5_x25_wavelengths,
        era5_psd=era5_x25_psd,
        cpc_wavelengths=cpc_wavelengths,
        cpc_psd=cpc_psd,
        figs_dir=os.path.join(figs_dir, "psds"),
        filename="psd_x25.png",
        lambda_star=lambda_star,
        psd_star=psd_star,
    )
    
    # low-pass
    print("low-pass:")
    lambda_star, psd_star = find_intersection(cpc_wavelengths, era5_lowpass_psd, cpc_psd)
    sigma_star = np.sqrt(psd_star * Nx*Ny)
    print("lambda star: {:.2f}".format(lambda_star))
    print("psd star: {:.2e}".format(psd_star))
    print("sigma star: {:.2e}".format(sigma_star))
    plot_psds(
        era5_wavelengths=era5_lowpass_wavelengths,
        era5_psd=era5_lowpass_psd,
        cpc_wavelengths=cpc_wavelengths,
        cpc_psd=cpc_psd,
        figs_dir=os.path.join(figs_dir, "psds"),
        filename="psd_lowpass.png",
        lambda_star=lambda_star,
        psd_star=psd_star,
    )
    
    # QM
    print("qm:")
    lambda_star, psd_star = find_intersection(cpc_wavelengths, era5_qm_psd, cpc_psd)
    sigma_star = np.sqrt(psd_star * Nx*Ny)
    print("lambda star: {:.2f}".format(lambda_star))
    print("psd star: {:.2e}".format(psd_star))
    print("sigma star: {:.2e}".format(sigma_star))
    plot_psds(
        era5_wavelengths=era5_qm_wavelengths,
        era5_psd=era5_qm_psd,
        cpc_wavelengths=cpc_wavelengths,
        cpc_psd=cpc_psd,
        figs_dir=os.path.join(figs_dir, "psds"),
        filename="psd_qm.png",
        lambda_star=lambda_star,
        psd_star=psd_star,
    )
    plot_psds_complete(
        raw_era5_wavelengths=raw_era5_wavelengths,
        raw_era5_psd =raw_era5_psd,
        era5_x25_wavelengths=era5_x25_wavelengths,
        era5_x25_psd=era5_x25_psd,
        lowpass_era5_wavelengths=era5_lowpass_wavelengths,
        lowpass_era5_psd=era5_lowpass_psd,
        era5_qm_wavelengths=era5_qm_wavelengths,
        era5_qm_psd=era5_qm_psd,
        cpc_wavelengths=cpc_wavelengths,
        cpc_psd=cpc_psd,
        figs_dir=os.path.join(figs_dir, "psds"),
        filename="complete_psd.png",
        cutoff=cutoff,
    )
    
    # Plot all preprocessing steps applied to ERA5
    arrays = (
        raw_era5_ds.tp.values[time,::-1,:]*1000,
        train_era5_x25_data[time,:,:], 
        train_era5_data_lowpass[time,:,:], 
        train_era5_qm_data[time,:,:]
    )
    titles = ("ERA5 (raw)", "ERA5 (x25)", "ERA5 (low-pass)", "ERA5 (QM)")
    extents = (raw_era5_extent, extent, extent, extent)
    fig, _ = plot_maps(arrays, titles, extents)
    fig.savefig(os.path.join(figs_dir, "maps/era5_preprocessing.png"))
    
    # Make negative values = 0
    train_era5_x25_data[train_era5_x25_data < 0] = 0
    train_era5_data_lowpass[train_era5_data_lowpass < 0] = 0
    train_era5_qm_data[train_era5_qm_data < 0] = 0
    cpc_data[cpc_data < 0] = 0
    
    # Create and save new datasets
    train_data_dir = "/work/FAC/FGSE/IDYST/tbeucler/downscaling/mlima/data/train_data"
    create_folder(train_data_dir)
    write_dataset(times, new_lats, new_lons, era5_data, os.path.join(train_data_dir, "era5.h5"))
    
    
    ##########
    # Test set
    ##########
    
    # Read file
    file_path = os.path.join(raw_data_dir, "era5/precip/era5_tp_2021_06_28-30.nc")
    ds = xr.open_dataset(file_path)
    
    times = ds.time.values
    
    # Intepolate
    era5_data = regrid_era5(ds, times, new_lats, new_lons)
    era5_data[era5_data < 0] = 0
    test_era5_x25_data = era5_data    
    
    # Low-pass filter
    era5_data = apply_low_pass_filter(era5_data, cutoff, x_length, y_length)
    era5_data[era5_data < 0] = 0
    test_era5_data_lowpass = era5_data
    
    # Quantile mapping
    era5_data = quantile_mapping(cpc_data, train_era5_data_lowpass, test_era5_data_lowpass)
    era5_data[era5_data < 0] = 0
    test_era5_qm_data = era5_data
    
    # Save datasets
    test_data_dir = "/work/FAC/FGSE/IDYST/tbeucler/downscaling/mlima/data/test_data"
    # x25
    write_dataset(times, new_lats, new_lons, test_era5_x25_data, os.path.join(test_data_dir, "era5_x25.h5"))
    # low-pass
    write_dataset(times, new_lats, new_lons, test_era5_data_lowpass, os.path.join(test_data_dir, "era5_lowpass.h5"))
    # qm
    write_dataset(times, new_lats, new_lons, test_era5_qm_data, os.path.join(test_data_dir, "era5_qm.h5"))
    
    print("Done!")
    
    
if __name__ == "__main__":
    main()
