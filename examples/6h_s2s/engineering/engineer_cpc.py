# here I'm using the result of the datasets produced by
# examples/cpc_era5_wrf/engineering/engineer_cpc.py for
# simplicity :D

import os, h5py
import numpy as np
import pandas as pd
import xarray as xr
from datetime import datetime, timedelta
from tqdm import tqdm

from utils import write_dataset

from configs.cpc import get_config


def generate_summer_range(time):
    # Generate 6-hour intervals within summer months, from the first to the last datetime in `time`
    start = time[0]
    end = time[-1]
    complete_range = np.arange(start, end + np.timedelta64(6, 'h'), np.timedelta64(6, 'h'))
    
    # Filter for dates in June, July, and August
    summer_range = [date for date in complete_range if date.astype('datetime64[M]').astype(int) % 12 + 1 in [6, 7, 8]]
    
    return summer_range

def aggregate_and_save(file_dir, pre_file_name, file_name):
    with h5py.File(os.path.join(file_dir, pre_file_name), 'r') as f:
        precip = f['precip'][...]
        lats = f['latitude'][:]
        lons = f['longitude'][:]
    
    times = xr.open_dataset(os.path.join(file_dir, pre_file_name))['time'].values
    times = np.sort(times)
    
    # Generate the full summer 6-hourly range
    summer_range = generate_summer_range(times)
    
    # Aggregate precipitation data following the summer range
    agg_precip = []
    agg_times = []
    for agg_time in summer_range:
        # Find indices in `times` that match the current 6-hour window
        time_indices = [i for i, t in enumerate(times) if agg_time <= t < agg_time + np.timedelta64(6, 'h')]
        
        # Check if the 6-hour window has complete data
        if len(time_indices) == 6:
            precip_sum = np.sum(precip[time_indices], axis=0)
            agg_precip.append(precip_sum)
            agg_times.append(agg_time)

    # Save combined train and validation data to a single file
    os.makedirs(file_dir, exist_ok=True)
    write_dataset(
        np.array(agg_times),
        lats,
        lons,
        np.array(agg_precip),
        os.path.join(file_dir, file_name)
    )


def main():
    config = get_config()
    
    test_data_dir = config.test_data_dir
    validation_data_dir = config.validation_data_dir
    train_data_dir = config.train_data_dir
    cpc_preprocessed_file_name = config.cpc_preprocessed_file_name
    cpc_aggregated_file_name = config.cpc_aggregated_file_name
    
    aggregate_and_save(test_data_dir, cpc_preprocessed_file_name, cpc_aggregated_file_name)
    aggregate_and_save(validation_data_dir, cpc_preprocessed_file_name, cpc_aggregated_file_name)
    aggregate_and_save(train_data_dir, cpc_preprocessed_file_name, cpc_aggregated_file_name)


if __name__ == "__main__":
    main()
