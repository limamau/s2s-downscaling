import os
import numpy as np
from datetime import datetime

from engineering.regridding import *
from evaluation.plots import *
from utils import *

def main():
    # TODO: write this info in processing.yml file
    raw_data_dir = "/work/FAC/FGSE/IDYST/tbeucler/downscaling/MeteoSwiss_Products/CPCH"
    test_year = 2021
    train_year = 2024
    
    ##########
    # Test set
    ##########
    xmin, xmax, ymin, ymax = 2255000, 2965000, 840000, 1480000
    xs = np.arange(xmin, xmax, 1000)
    ys = np.arange(ymin, ymax, 1000)[::-1] # reversed y
    new_extent = (5.9, 10.6, 45.8, 47.9)
    
    # Process
    initial_date = datetime(2021, 6, 28)
    final_date = datetime(2021, 6, 29)
    print("Test set:")
    times, raw_cpc_data = concat_cpc(os.path.join(raw_data_dir, str(test_year)), initial_date, final_date)
    lats, lons, test_data = regrid_cpc(raw_cpc_data, xs, ys, new_extent)
    
    # Check plot
    time = 42
    print("Plot time:", times[time])
    script_dir = os.path.dirname(os.path.realpath(__file__))
    figs_dir = os.path.join(script_dir, "figs/engineering/maps")
    os.makedirs(figs_dir, exist_ok=True)
    os.makedirs(os.path.join(figs_dir, "maps"), exist_ok=True)
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
    initial_date = datetime(2024, 1, 1)
    final_date = datetime(2024, 1, 31)
    print("Train set:")
    times, raw_cpc_data = concat_cpc(os.path.join(raw_data_dir, str(train_year)), initial_date, final_date)
    lats, lons, train_data = regrid_cpc(raw_cpc_data, xs, ys, new_extent)
    
    # Check plot
    time = 70
    print("Plot time:", times[time])
    arrays = (raw_cpc_data[time,::-1,:], train_data[time,:,:])
    fig, _ = plot_maps(arrays, titles, extents, projections, axis_labels=axis_labels)
    fig.savefig(os.path.join(figs_dir, "cpc_maps_train.png"))
    
    # Save train data
    train_data_dir = "/work/FAC/FGSE/IDYST/tbeucler/downscaling/mlima/data/train_data"
    create_folder(train_data_dir)
    write_dataset(times, lats, lons, train_data, os.path.join(train_data_dir, "cpc.h5"))
    
    
if __name__ == '__main__':
    main()