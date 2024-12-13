import glob, h5py, os, tomllib
import numpy as np
from pyproj import Transformer
from tqdm import tqdm

from engineering.regridding import interpolate_data, regularize_grid, cut_data
from data.surface_data import SurfaceData

from configs.cpc import get_config


### auxiliary functions ###
def concat_cpc_6h(data_dir, initial_date, final_date):
    print(f"Processing from {initial_date} to {final_date}")
    
    # Initialize lists for times and aggregated data
    times = []
    aggregated_data = []

    # Precompute time ranges for efficiency
    total_days = np.timedelta64(final_date - initial_date, 'D').astype(int) + 1
    intervals = [1, 7, 13, 19]

    for d in tqdm(range(total_days), desc="Aggregating data"):
        current_date = initial_date + np.timedelta64(d, 'D')
        next_date = current_date + np.timedelta64(1, 'D')
        # TODO: this breaks for the last date of the year (that's not a problem for the current code)

        # Load current day and next day files
        current_pattern = f"CPC{str(current_date)[2:4]}{(current_date - np.datetime64(str(current_date)[:4] + '-01-01') + 1).astype(int):03d}"
        next_pattern = f"CPC{str(next_date)[2:4]}{(next_date - np.datetime64(str(next_date)[:4] + '-01-01') + 1).astype(int):03d}"
        
        current_files = sorted(glob.glob(os.path.join(data_dir, current_pattern) + '*'))
        next_files = sorted(glob.glob(os.path.join(data_dir, next_pattern) + '*'))

        # Ensure there are enough files for all intervals
        if len(current_files) < 23 or not next_files:
            print(f"Skipping day {current_date}: incomplete data")
            continue

        # Combine relevant files: T01-T23 from current day + T00 from next day
        all_files = current_files[1:] + [next_files[0]]

        # Load and cache file data
        daily_data = np.array([
            h5py.File(file, 'r')['dataset1/data1/data'][...] for file in all_files
        ])

        # Aggregate 6-hour intervals
        for start_hour in intervals:
            end_hour = start_hour + 5
            hour_indices = np.arange(start_hour - 1, end_hour)
            aggregated_data.append(np.mean(daily_data[hour_indices], axis=0))
            times.append(current_date + np.timedelta64(end_hour, 'h'))

    return np.array(times), np.array(aggregated_data)


def regrid_cpc(data, xs, ys, extent):
    lon_2d, lat_2d = transform_cpc_coordinates(xs, ys)
    new_lon, new_lat, new_lon_2d, new_lat_2d = regularize_grid(lon_2d, lat_2d, xs.size, ys.size)
    new_data = interpolate_data(data, lon_2d, lat_2d, new_lon_2d, new_lat_2d)
    return cut_data(new_lon, new_lat, new_data, extent)


def transform_cpc_coordinates(xs, ys):
    transformer = Transformer.from_proj(2056, 4326, always_xy=True)
    xx, yy = np.meshgrid(xs, ys)
    lon_2d, lat_2d = transformer.transform(xx, yy)
    return lon_2d, lat_2d


def process_month(raw_data_dir, initial_date, final_date, xs, ys, new_extent):
    times, raw_data = concat_cpc_6h(
        os.path.join(raw_data_dir, str(initial_date)[:4]), initial_date, final_date
    )
    lats, lons, data = regrid_cpc(raw_data, xs, ys, new_extent)
    return times, lats, lons, data


### main calls ###
def preprocess_cpc_data(raw_data_dir, xs, ys, new_extent, years, months):
    first = True
    for year in years:
        for month in months:
            start_date = np.datetime64(f'{year}-{month:02d}-01')
            end_date = np.datetime64(f'{year}-{month+1:02d}-01') - np.timedelta64(1, 'D')
            if first:
                times, lats, lons, data = process_month(
                    raw_data_dir, start_date, end_date, xs, ys, new_extent
                )
                first = False
            else:
                aux_times, _, _, aux_data = process_month(
                    raw_data_dir, start_date, end_date, xs, ys, new_extent
                )
                times = np.concatenate([times, aux_times], axis=0)
                data = np.concatenate([data, aux_data], axis=0)
    
    # pass time to nanoseconds to avoid warning in xarray
    times = times.astype('datetime64[ns]')
    
    return SurfaceData(times, lats, lons, precip=data)


def save_data_from_dates(sfc_data, data_dir, dates):
    new_sfc_data = sfc_data.take_out_from_date_range(dates)
    os.makedirs(data_dir, exist_ok=True)
    new_sfc_data.save_to_h5(os.path.join(data_dir, "cpc.h5"))
    return sfc_data


def main():
    # directory paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(script_dir, "../dirs.toml"), "rb") as f:
        dirs = tomllib.load(f)
    base = dirs["main"]["base"]
    raw_data_dir = dirs["raw"]["cpc"]
    test_data_dir = os.path.join(base, dirs["subs"]["test"])
    validation_data_dir = os.path.join(base, dirs["subs"]["validation"])
    train_data_dir = os.path.join(base, dirs["subs"]["train"])

    # extra configurations
    config = get_config()
    extent = config.extent
    new_extent = config.new_extent
    years = config.years
    months = config.months
    validation_dates = config.validation_dates
    test_dates = config.test_dates
    # maybe there's better way of doing that without these xs and ys
    xs = np.arange(extent[0], extent[1], 1000)
    ys = np.arange(extent[2], extent[3], 1000)[::-1]

    # main calls
    sfc_data = preprocess_cpc_data(raw_data_dir, xs, ys, new_extent, years, months)
    sfc_data = save_data_from_dates(sfc_data, test_data_dir, test_dates)
    sfc_data = save_data_from_dates(sfc_data, validation_data_dir, validation_dates)
    sfc_data.save_to_h5(os.path.join(train_data_dir, "cpc.h5"))


if __name__ == '__main__':
    main()
