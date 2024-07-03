import os
import h5py
import numpy as np
import xarray as xr
from pandas import to_datetime

from utils import filter_dry_images

def extract_month_hour(times):
    times = to_datetime(times)
    months = np.array([dt.month for dt in times])
    hours = np.array([dt.hour for dt in times])
    return months, hours

def main():
    years = [2020, 2021, 2022, 2023, 2024]
    train_dir = "/work/FAC/FGSE/IDYST/tbeucler/downscaling/mlima/data/train_data"
    data_arrays = []
    months_arrays = []
    hours_arrays = []
    
    exclude_dates = ["2021-06-28", "2021-06-29"]
    exclude_dates = to_datetime(exclude_dates)
    
    for year in years:
        print(f"Processing year {year}...")
        with h5py.File(os.path.join(train_dir, f"cpc_{year}.h5"), "r") as f:
            data = f["precip"][:]
        times = xr.open_dataset(os.path.join(train_dir, f"cpc_{year}.h5")).time.values
        
        # Filter out very dry images
        indices = filter_dry_images(data, return_indices=True)
        data = data[indices]
        times = to_datetime(times[indices])
        
        # Exclude specific dates
        mask = ~times.isin(exclude_dates)
        data = data[mask]
        times = times[mask]
        
        months, hours = extract_month_hour(times)
        
        # Exclude June and July
        june_july_mask = (months == 6) | (months == 7)
        data = data[june_july_mask]
        months = months[june_july_mask]
        hours = hours[june_july_mask]
        
        data_arrays.append(data)
        months_arrays.append(months)
        hours_arrays.append(hours)
        
    # Concatenate data
    print("Concatenating data...")
    data = np.concatenate(data_arrays, axis=0)
    months = np.concatenate(months_arrays, axis=0)
    hours = np.concatenate(hours_arrays, axis=0)
    
    print("Data shape:", data.shape)
    
    # Write data
    with h5py.File(os.path.join(train_dir, "cpc_june-july-dry-filter.h5"), "w") as f:
        f.create_dataset("precip", data=data)
        f.create_dataset("month", data=months)
        f.create_dataset("hour", data=hours)
        

if __name__ == '__main__':
    main()
