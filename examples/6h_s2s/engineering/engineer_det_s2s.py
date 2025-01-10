import copy, os, tomllib
import numpy as np

from data.surface_data import SurfaceData

from engineering_utils import aggregate_det_s2s_precip
from configs.det_s2s import get_config


def run_engineering(storm_dates, lead_time_files, cpc_file):
    # Load CPC
    cpc = SurfaceData.load_from_h5(cpc_file, ["precip"])
        
    # Load S2S
    s2s = aggregate_det_s2s_precip(lead_time_files, storm_dates)
    extent = cpc.get_extent()
    s2s.cut_data(extent)
    s2s.unflip_latlon()
    print(f"Cut S2S data shape: {s2s.precip.shape}")
    
    # Interpolate S2S to the same resolution as CombiPrecip
    nearest_s2s = copy.deepcopy(s2s)
    nearest_s2s.regrid(cpc, method='nearest')
    print(f"Nearest S2S data shape: {nearest_s2s.precip.shape}")
    
    # (Radial) low-pass filter in each lead time
    nearest_lowpass_s2s = copy.deepcopy(nearest_s2s)
    nearest_lowpass_s2s.low_pass_filter(s2s)
    print(f"Nearest low-passed S2S data shape: {nearest_lowpass_s2s.precip.shape}")
    
    # filter negative values from nearest_lowpass_s2s
    nearest_lowpass_s2s.precip = np.where(
        nearest_lowpass_s2s.precip < 0, 0, nearest_lowpass_s2s.precip
    )
    
    return s2s, nearest_s2s, nearest_lowpass_s2s


def check_negative_values(data):
    if (data < -1).any():
        print("Negative values found in the data")
    else:
        print("No negative values found in the data")
    


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
    print("cpc:")
    check_negative_values(SurfaceData.load_from_h5(cpc_file, ["precip"]).precip)
    s2s, nearest_s2s, nearest_lowpass_s2s = run_engineering(
        storm_dates, lead_time_files, cpc_file,
    )
    s2s.save_to_h5(os.path.join(test_data_dir, "det_s2s.h5"))
    print("s2s:")
    check_negative_values(nearest_s2s.precip)
    nearest_s2s.save_to_h5(os.path.join(test_data_dir, "det_s2s_nearest.h5"))
    print("s2s nearest:")
    check_negative_values(nearest_lowpass_s2s.precip)
    nearest_lowpass_s2s.save_to_h5(os.path.join(test_data_dir, "det_s2s_nearest_low-pass.h5"))
    print("s2s nearest low-pass:")
    check_negative_values(nearest_lowpass_s2s.precip)


if __name__ == "__main__":
    main()
