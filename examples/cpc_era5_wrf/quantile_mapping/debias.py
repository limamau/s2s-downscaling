import os, time, h5py
import xarray as xr
from config import get_config

from debiasing.quantile_mapping import QuantileMapping
from utils import write_dataset

def debias(train_data_dir, test_data_dir, sim_data_dir, n_quantiles, method_point):
    # Read training data
    with h5py.File(os.path.join(train_data_dir, "era5_nearest_low-pass.h5"), "r") as f:
        lats = f["latitude"][:]
        lons = f["longitude"][:]
        era5_train_data = f["precip"][:,:,:]
    print("ERA5 shape: ", era5_train_data.shape)
        
    with h5py.File(os.path.join(train_data_dir, "cpc.h5"), "r") as f:
        cpc_train_data = f["precip"][:,:,:]
    print("CPC shape: ", cpc_train_data.shape)
    
    # Filter negative values
    era5_train_data[era5_train_data < 0] = 0
    cpc_train_data[cpc_train_data < 0] = 0
    
    # Choose the quantile mapping method
    point_qm = QuantileMapping(n_quantiles=n_quantiles, method=method_point)

    # Correct bias with quantile mapping
    print("Training quantile mapping (point to point)...")
    start_time = time.time()
    point_qm.fit(cpc_train_data, era5_train_data)
    end_time = time.time()
    print("Trained in {:.2f} seconds".format(end_time - start_time))
    
    start_time = time.time()
    end_time = time.time()
    print("Predicted in {:.2f} seconds".format(end_time - start_time))
    
    # Read test data
    with h5py.File(os.path.join(test_data_dir, "era5_nearest_low-pass.h5"), "r") as f:
        era5_test_data = f["precip"][:,:,:]
    test_times = xr.open_dataset(os.path.join(test_data_dir, "era5_nearest_low-pass.h5")).time.values
        
    # Filter negative values
    era5_test_data[era5_test_data < 0] = 0
        
    # Correct bias with quantile mapping
    point_qm_test_data = point_qm.predict(era5_test_data)
    
    # Save processed data
    write_dataset(
        test_times, lats, lons, point_qm_test_data, 
        os.path.join(sim_data_dir, "qm_nearest_low-pass_point.h5")
    )
    
    
def main():
    config = get_config()
    train_data_dir = config.train_data_dir
    test_data_dir = config.test_data_dir
    sim_data_dir = config.sim_data_dir
    n_quantiles = config.n_quantiles
    method_point = config.method_point
    
    debias(train_data_dir, test_data_dir, sim_data_dir, n_quantiles, method_point)


if __name__ == "__main__":
    main()
