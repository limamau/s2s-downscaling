import os, time, logging, h5py
import numpy as np
from cartopy import crs as ccrs
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed

from evaluation.plots import plot_maps
from utils import *

from engineering_utils import concat_cpc, regrid_cpc, check_and_sort_times, split_date_range


# Set up logging
script_dir = os.path.dirname(os.path.realpath(__file__))
log_file_path = os.path.join(script_dir, "eng.out")
with open(log_file_path, 'w') as f:
    f.write('')
logging.basicConfig(filename=log_file_path, level=logging.INFO, format='%(asctime)s %(message)s')


def process_cpc(raw_data_dir, initial_date, final_date, xs, ys, new_extent):
    times, raw_cpc_data = concat_cpc(os.path.join(raw_data_dir, str(initial_date.year)), initial_date, final_date)
    lats, lons, train_data = regrid_cpc(raw_cpc_data, xs, ys, new_extent)
    return times, lats, lons, train_data


def parallel_process_cpc(raw_data_dir, initial_date, final_date, xs, ys, new_extent, chunk_size=7, num_workers=None):
    date_ranges = split_date_range(initial_date, final_date, chunk_size)
    results = []
    num_chunks = len(date_ranges)
    start_time = time.time()

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        future_to_date_range = {
            executor.submit(process_cpc, raw_data_dir, start, end, xs, ys, new_extent): (start, end) 
            for start, end in date_ranges
        }

        for i, future in enumerate(as_completed(future_to_date_range), 1):
            date_range = future_to_date_range[future]
            
            try:
                times, lats, lons, train_data = future.result()
                results.append((times, lats, lons, train_data))
            
            except Exception as e:
                logging.error(f"Error processing date range {date_range[0]} to {date_range[1]}: {e}")
                
            chunk_end_time = time.time()
            total_elapsed_time = chunk_end_time - start_time
            average_time_per_chunk = total_elapsed_time / i
            remaining_chunks = num_chunks - i
            estimated_remaining_time = average_time_per_chunk * remaining_chunks
            logging.info(f"Chunk {i}/{num_chunks} processed. Estimated time remaining: {estimated_remaining_time:.2f} seconds.")

    # Combine results from different chunks into numpy arrays
    lats = results[0][1]
    lons = results[0][2]
    combined_times = np.concatenate([result[0] for result in results])
    combined_train_data = np.concatenate([result[3] for result in results])

    # Check if combined_times is correctly ordered and sort if necessary
    times, train_data = check_and_sort_times(combined_times, combined_train_data)

    return times, lats, lons, train_data


def main():
    raw_data_dir = "/work/FAC/FGSE/IDYST/tbeucler/downscaling/MeteoSwiss_Products/CPCH"
    
    ##########
    # Test set
    ##########
    xmin, xmax, ymin, ymax = 2255000, 2965000, 840000, 1480000
    xs = np.arange(xmin, xmax, 1000)
    ys = np.arange(ymin, ymax, 1000)[::-1] # reversed y
    new_extent = (5.9, 10.6, 45.8, 47.9)
    
    # Process
    test_year = 2021
    initial_date = datetime(2021, 6, 28)
    final_date = datetime(2021, 6, 29)
    print("Test set:")
    times, raw_cpc_data = concat_cpc(os.path.join(raw_data_dir, str(test_year)), initial_date, final_date)
    lats, lons, test_data = regrid_cpc(raw_cpc_data, xs, ys, new_extent)
    
    # Check plot
    time = 42
    print("Plot time:", times[time])
    script_dir = os.path.dirname(os.path.realpath(__file__))
    figs_dir = os.path.join(script_dir, "figs/maps")
    create_folder(figs_dir)
    raw_cpc_extent = (np.min(xs), np.max(xs), np.min(ys), np.max(ys))
    new_extent = (np.min(lons), np.max(lons), np.min(lats), np.max(lats))
    arrays = (raw_cpc_data[time,::-1,:], test_data[time,:,:])
    titles = ("CombiPrecip (original)", "CombiPrecip (processed)")
    extents = (raw_cpc_extent, new_extent)
    projections = (ccrs.epsg(2056), ccrs.PlateCarree())
    axis_labels = (("x", "y"), ("lon", "lat"))
    fig, _ = plot_maps(arrays, titles, extents, projections, axis_labels=axis_labels)
    fig.savefig(os.path.join(figs_dir, "cpc_maps_test.png"))
    
    # Save test data
    test_data_dir = "/work/FAC/FGSE/IDYST/tbeucler/downscaling/mlima/data/test_data"
    create_folder(test_data_dir)
    write_dataset(times, lats, lons, test_data, os.path.join(test_data_dir, "cpc.h5"))
    
    
    ###########
    # Train set
    ###########
    
    # Process
    num_workers = 8
    years = [2020, 2021, 2022, 2023, 2024]
    for year in years:
        initial_date = datetime(year, 1, 1)
        if year == 2024:
            final_date = datetime(year, 3, 30)
        else:
            final_date = datetime(year, 12, 31)
        print("Train set:")
        logging.info("Starting processing")
        times, lats, lons, train_data = parallel_process_cpc(raw_data_dir, initial_date, final_date, xs, ys, new_extent, num_workers=num_workers)
        logging.info("Processing completed")
        
        # Save train data
        train_data_dir = "/work/FAC/FGSE/IDYST/tbeucler/downscaling/mlima/data/train_data"
        create_folder(train_data_dir)
        write_dataset(times, lats, lons, train_data, os.path.join(train_data_dir, "cpc_{}.h5".format(year)))
        
    # Some check plotting
    initial_date = datetime(2024, 1, 31)
    final_date = datetime(2024, 1, 31)
    raw_cpc_data = concat_cpc(os.path.join(raw_data_dir, str(year)), initial_date, final_date)[1]
    
    with h5py.File(os.path.join(train_data_dir, "cpc_2024.h5"), "r") as f:
        train_data = f["precip"][24*31,:,:]
    
    # Check plot
    arrays = (raw_cpc_data[0,::-1,:], train_data[:,:])
    fig, _ = plot_maps(arrays, titles, extents, projections, axis_labels=axis_labels)
    fig.savefig(os.path.join(figs_dir, "cpc_maps_train.png"))
    
    
if __name__ == '__main__':
    main()
