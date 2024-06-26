import os, h5py
import numpy as np
from tqdm import tqdm
import xarray as xr
from datetime import datetime

from engineering.spectrum import get_psd, apply_low_pass_filter
from evaluation.plots import plot_maps, plot_psds
from utils import create_folder, write_dataset, get_spatial_lengths

from engineering_utils import concat_era5, regrid_era5

    

def main():
    # Train set
    ##################
    
    # # File directories
    raw_data_dir = "/work/FAC/FGSE/IDYST/tbeucler/downscaling/raw_data/era5/precip/"
    train_data_dir = "/work/FAC/FGSE/IDYST/tbeucler/downscaling/mlima/data/train_data"
    
    # # Concatenate files
    # initial_date = datetime(2021, 7, 1)
    # final_date = datetime(2023, 6, 30)
    # times, raw_era5_lats, raw_era5_lons, raw_era5_data = concat_era5(raw_data_dir, initial_date, final_date)
    # print("raw era5 shape:", raw_era5_data.shape)
    
    # # Sort data
    # if raw_era5_lats[0] > raw_era5_lats[-1]:
    #     raw_era5_lats = raw_era5_lats[::-1]
    #     raw_era5_data = raw_era5_data[:, ::-1, :]
    # if raw_era5_lons[0] > raw_era5_lons[-1]:
    #     raw_era5_lons = raw_era5_lons[::-1]
    #     raw_era5_data = raw_era5_data[:, :, ::-1]
        
    # # Scale
    M_TO_MM = 1000
    # raw_era5_data = raw_era5_data * M_TO_MM
    
    # # Read info from already preprocessed CPC data
    cpc_file = os.path.join(train_data_dir, "cpc.h5")
    with h5py.File(cpc_file, "r") as f:
        cpc_lons = f["longitude"][:]
        cpc_lats = f["latitude"][:]
        # t = -10 # just to plot
    #     cpc_data_t = f["precip"][t,:,:]
    # print("cpc lats:", cpc_lats.shape)
    # print("cpc lons:", cpc_lons.shape)
        
    # # Interpolate ERA5 to the same resolution as CPC
    # num_chunks = 20
    # interpolated_era5_data = np.zeros((times.size, cpc_lats.size, cpc_lons.size))
    # for i in tqdm(range(num_chunks), desc="Interpolating ERA5"):
    #     start = i * len(raw_era5_data) // num_chunks
    #     end = (i + 1) * len(raw_era5_data) // num_chunks
    #     chunk = raw_era5_data[start:end]
    #     interpolated_era5_data[start:end] = regrid_era5(raw_era5_lats, raw_era5_lons, chunk, cpc_lats, cpc_lons)
    # print("interpolated era5 shape:", interpolated_era5_data.shape)
    
    # # Low-pass filter
    # raw_era5_x_length, raw_era5_y_length = get_spatial_lengths(raw_era5_lons, raw_era5_lats)
    # raw_era5_wavelengths, _ = get_psd(raw_era5_data, raw_era5_x_length, raw_era5_y_length)
    # cutoff = 2*np.pi/np.min(raw_era5_wavelengths)
    # x_length, y_length = get_spatial_lengths(cpc_lons, cpc_lats)
    # lowpassed_era5_data = np.zeros((times.size, cpc_lats.size, cpc_lons.size))
    # num_chunks = 80
    # for i in tqdm(range(num_chunks), desc="Low-pass filtering"):
    #     start = i * len(interpolated_era5_data) // num_chunks
    #     end = (i + 1) * len(interpolated_era5_data) // num_chunks
    #     chunk = interpolated_era5_data[start:end]
    #     lowpassed_era5_data[start:end] = apply_low_pass_filter(chunk, cutoff, x_length, y_length)
    # print("lowpassed era5 shape:", lowpassed_era5_data.shape)
    
    # # Create and save new dataset
    # create_folder(train_data_dir)
    # write_dataset(times, raw_era5_lats, raw_era5_lons, raw_era5_data, os.path.join(train_data_dir, "era5_raw.h5"))
    # write_dataset(times, cpc_lats, cpc_lons, lowpassed_era5_data, os.path.join(train_data_dir, "era5_low-pass.h5"))
    
    # # Plot maps
    # arrays = (
    #     raw_era5_data[t,:,:],
    #     interpolated_era5_data[t,:,:],
    #     lowpassed_era5_data[t,:,:],
    #     cpc_data_t,
    # )
    # era5_extent = (raw_era5_lons[0], raw_era5_lons[-1], raw_era5_lats[0], raw_era5_lats[-1])
    # cpc_extent = (cpc_lons[0], cpc_lons[-1], cpc_lats[0], cpc_lats[-1])
    # extents = (era5_extent, cpc_extent, cpc_extent, cpc_extent)
    titles = ("ERA5", "Interpolated ERA5", "Low-pass filtered ERA5", "CPC")
    # fig, _ = plot_maps(arrays, titles, extents)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    figs_dir = os.path.join(script_dir, "figs")
    # fig.savefig(os.path.join(figs_dir, "maps/era5_steps_train.png"))
    
    # # Plot PSDs
    # random_idx = np.random.randint(0, len(times), size=10)
    # arrays = (
    #     raw_era5_data[random_idx],
    #     interpolated_era5_data[random_idx],
    #     lowpassed_era5_data[random_idx],
    # )
    # labels = ("ERA5", "Interpolated ERA5", "Low-pass filtered ERA5")
    # spatial_lengths = (
    #     (raw_era5_x_length, raw_era5_y_length),
    #     (x_length, y_length),
    #     (x_length, y_length),
    # )
    # fig, _ = plot_psds(arrays, labels, spatial_lengths)
    # fig.savefig(os.path.join(figs_dir, "psds/era5_psds_train.png"))
    
    
    # Test set
    ##########
    
    # Read file
    file_path = os.path.join(raw_data_dir, "era5_tp_2021_06_28-30.nc")
    ds = xr.open_dataset(file_path)
    
    times = ds.time.values
    raw_era5_lats = ds.latitude.values
    raw_era5_lons = ds.longitude.values
    raw_era5_data = ds.tp.values
    
    # Sort data
    if raw_era5_lats[0] > raw_era5_lats[-1]:
        raw_era5_lats = raw_era5_lats[::-1]
        raw_era5_data = raw_era5_data[:, ::-1, :]
    if raw_era5_lons[0] > raw_era5_lons[-1]:
        raw_era5_lons = raw_era5_lons[::-1]
        raw_era5_data = raw_era5_data[:, :, ::-1]
        
    # Scale
    raw_era5_data = raw_era5_data * M_TO_MM
    era5_extent = (raw_era5_lons[0], raw_era5_lons[-1], raw_era5_lats[0], raw_era5_lats[-1])
    cpc_extent = (cpc_lons[0], cpc_lons[-1], cpc_lats[0], cpc_lats[-1])
    extents = (era5_extent, cpc_extent, cpc_extent, cpc_extent)
    
    # Interpolate ERA5 to the same resolution as CombiPrecip
    interpolated_era5_data = regrid_era5(raw_era5_lats, raw_era5_lons, raw_era5_data, cpc_lats, cpc_lons)
    
    # Low-pass filter
    raw_era5_x_length, raw_era5_y_length = get_spatial_lengths(raw_era5_lons, raw_era5_lats)
    raw_era5_wavelengths, _ = get_psd(raw_era5_data, raw_era5_x_length, raw_era5_y_length)
    cutoff = 2*np.pi/np.min(raw_era5_wavelengths)
    x_length, y_length = get_spatial_lengths(cpc_lons, cpc_lats)
    lowpassed_era5_data = apply_low_pass_filter(interpolated_era5_data, cutoff, x_length, y_length)
    
    # Read info from already processed CPC data just to plot
    test_data_dir = "/work/FAC/FGSE/IDYST/tbeucler/downscaling/mlima/data/test_data"
    with h5py.File(os.path.join(test_data_dir, "cpc.h5"), "r") as f:
        t = -10 # just to plot again
        cpc_data_t = f["precip"][t,:,:]
    
    # Plot maps
    arrays = (
        raw_era5_data[t,:,:],
        interpolated_era5_data[t,:,:],
        lowpassed_era5_data[t,:,:],
        cpc_data_t,
    )
    fig, _ = plot_maps(arrays, titles, extents)
    fig.savefig(os.path.join(figs_dir, "maps/era5_steps_test.png"))
    
    # Plot PSDs
    labels = ("ERA5", "Interpolated ERA5", "Low-pass filtered ERA5")
    spatial_lengths = (
        (raw_era5_x_length, raw_era5_y_length),
        (x_length, y_length),
        (x_length, y_length),
    )
    random_idx = np.random.randint(0, len(times), size=10)
    arrays = (
        raw_era5_data[random_idx],
        interpolated_era5_data[random_idx],
        lowpassed_era5_data[random_idx],
    )
    fig, _ = plot_psds(arrays, labels, spatial_lengths)
    fig.savefig(os.path.join(figs_dir, "psds/era5_psds_test.png"))
    
    # Create and save new dataset
    create_folder(test_data_dir)
    write_dataset(times, raw_era5_lats, raw_era5_lons, raw_era5_data, os.path.join(test_data_dir, "era5_raw.h5"))
    write_dataset(times, cpc_lats, cpc_lons, interpolated_era5_data, os.path.join(test_data_dir, "era5_interpolated.h5"))
    write_dataset(times, cpc_lats, cpc_lons, lowpassed_era5_data, os.path.join(test_data_dir, "era5_low-pass.h5"))
    
    print("Done!")
    
    
if __name__ == "__main__":
    main()
