import copy, os, tomllib
import numpy as np
import xarray as xr

from data.surface_data import SurfaceData, ForecastEnsembleSurfaceData

from engineering_utils import get_s2s_idxs_to_keep
from configs.ens_s2s import get_config


# TODO: break this function with a for loop to aleviate the processing
def aggregate_ens_s2s_precip(lead_time_files, storm_dates):
    lead_times = []
    numbers = []
    times = []
    raw_s2s_lats = None
    aux_raw_s2s_data = None
    raw_s2s_lons = None
    raw_s2s_precip = None
    
    # Read and aggregate
    first = True
    for lead_time, file_tuple in lead_time_files.items():
        lead_times.append(lead_time)
        
        for i, file in enumerate(file_tuple):
            ds = xr.open_dataset(file)
            idxs_to_keep = get_s2s_idxs_to_keep(ds.time.values, storm_dates[i])
            idxs_to_keep_diff = np.concatenate([np.array([idxs_to_keep[0]-1]), idxs_to_keep])
            precip = np.expand_dims(np.diff(ds.tp.values[idxs_to_keep_diff], axis=0), axis=0)
            precip = np.reshape(precip, (1, precip.shape[2], precip.shape[1], *precip.shape[3:]))
            precip = precip / 6
            
            if raw_s2s_lats is None:
                numbers = ds.number.values
                raw_s2s_lats = ds.latitude.values
                raw_s2s_lons = ds.longitude.values
                raw_s2s_precip = precip
                print("First precip shape:", raw_s2s_precip.shape)
                times = ds.time.values[idxs_to_keep]
            
            elif first:
                raw_s2s_precip = np.concatenate([raw_s2s_precip, precip], axis=2)
                times = np.concatenate([times, ds.time.values[idxs_to_keep]])
            
            else:
                if aux_raw_s2s_data is None:
                    aux_raw_s2s_data = precip
                else:
                    aux_raw_s2s_data = np.concatenate([aux_raw_s2s_data, precip], axis=2)
        
        if not first:
            raw_s2s_precip = np.concatenate([raw_s2s_precip, aux_raw_s2s_data], axis=0)
            aux_raw_s2s_data = None
            
        first = False
    
    print("precip shape:", raw_s2s_precip.shape)
    return ForecastEnsembleSurfaceData(
        lead_times, numbers, times, raw_s2s_lats, raw_s2s_lons, precip=raw_s2s_precip,
    )


def run_engineering(storm_dates, lead_time_files, cpc_file):
    # Load CPC
    cpc = SurfaceData.load_from_h5(cpc_file, ["precip"])
        
    # Load S2S
    s2s = aggregate_ens_s2s_precip(lead_time_files, storm_dates)
    extent = cpc.get_extent()
    s2s.unflip_latlon()
    s2s.cut_data(extent)
    print(f"Cut S2S data shape: {s2s.precip.shape}")
    
    # Interpolate S2S to the same resolution as CombiPrecip
    nearest_s2s = copy.deepcopy(s2s)
    nearest_s2s.regrid(cpc, method='nearest')
    print(f"Nearest S2S data shape: {nearest_s2s.precip.shape}")
    
    # (Radial) low-pass filter in each lead time
    nearest_lowpass_s2s = copy.deepcopy(nearest_s2s)
    nearest_lowpass_s2s.low_pass_filter(s2s)
    print(f"Nearest low-passed S2S data shape: {nearest_lowpass_s2s.precip.shape}")
    
    return s2s, nearest_s2s, nearest_lowpass_s2s


def main():
    # directory paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(script_dir, "../dirs.toml"), "rb") as f:
        dirs = tomllib.load(f)
    base = dirs["main"]["base"]
    raw_data_dir = dirs["raw"]["s2s"]
    test_data_dir = os.path.join(base, dirs["subs"]["test"])
    
    # extra configurations
    config = get_config(raw_data_dir, test_data_dir)
    storm_dates = config.storm_dates
    lead_time_files = config.lead_time_files
    cpc_file = config.cpc_file
    
    # main calls
    s2s, nearest_s2s, nearest_lowpass_s2s = run_engineering(storm_dates, lead_time_files, cpc_file)
    s2s.save_to_h5(os.path.join(test_data_dir, "ens_s2s.h5"))
    nearest_s2s.save_to_h5(os.path.join(test_data_dir, "ens_s2s_nearest.h5"))
    nearest_lowpass_s2s.save_to_h5(os.path.join(test_data_dir, "ens_s2s_nearest_low-pass.h5"))


if __name__ == "__main__":
    main()
