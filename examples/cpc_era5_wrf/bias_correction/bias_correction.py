import os, time, h5py
import xarray as xr

from models.quantile_mapping import QuantileMapping
from models.generalized_gamma import GeneralizedGamma
from evaluation.plots import plot_maps, plot_cdfs, plot_pdfs
from utils import create_folder, write_dataset


def main():
    # Read training data
    train_data_dir = "/work/FAC/FGSE/IDYST/tbeucler/downscaling/mlima/data/train_data"
    max_train_size = 5000
    with h5py.File(os.path.join(train_data_dir, "era5_low-pass.h5"), "r") as f:
        lats = f["latitude"][:]
        lons = f["longitude"][:]
        era5_train_data = f["precip"][:max_train_size,:,:]
    train_times = xr.open_dataset(os.path.join(train_data_dir, "era5_low-pass.h5")).time.values[:max_train_size]
    print("ERA5 shape: ", era5_train_data.shape)
        
    with h5py.File(os.path.join(train_data_dir, "cpc.h5"), "r") as f:
        cpc_train_data = f["precip"][:max_train_size,:,:]
    print("CPC shape: ", cpc_train_data.shape)
    
    # Filter negative values
    era5_train_data[era5_train_data < 0] = 0
    cpc_train_data[cpc_train_data < 0] = 0
    
    # Chose the quantile mapping methods
    all_qm = QuantileMapping(n_quantiles=500, method='all')
    point_qm = QuantileMapping(n_quantiles=500, method='point_to_point')
    # point_gengamma = GeneralizedGamma(method='point_to_point')

    # Correct bias with quantile mapping
    # empirical all
    print("training empirical (all)...")
    start_time = time.time()
    all_qm.fit(cpc_train_data, era5_train_data)
    end_time = time.time()
    print("trained in {:.2f} seconds".format(end_time - start_time))
    start_time = time.time()
    all_qm_train_data = all_qm.predict(era5_train_data)
    end_time = time.time()
    print("predicted in {:.2f} seconds".format(end_time - start_time))
    
    # empirical point-to-point
    print("training empirical (point to point)...")
    start_time = time.time()
    point_qm.fit(cpc_train_data, era5_train_data)
    end_time = time.time()
    print("trained in {:.2f} seconds".format(end_time - start_time))
    start_time = time.time()
    point_qm_train_data = point_qm.predict(era5_train_data)
    end_time = time.time()
    print("predicted in {:.2f} seconds".format(end_time - start_time))
    
    # # gengamma point-to-point
    # print("training gengamma (point to point)...")
    # start_time = time.time()
    # point_gengamma.fit(test_cpc_regridded, era5_train_data)
    # end_time = time.time()
    # print("trained in {:.2f} seconds".format(end_time - start_time))
    # start_time = time.time()
    # point_gengamma_train_data = point_gengamma.predict(era5_train_data)
    # end_time = time.time()
    # print("predicted in {:.2f} seconds".format(end_time - start_time))


    # Plot maps
    t = -1
    extent = (lons[0], lons[-1], lats[0], lats[-1])
    script_dir = os.path.dirname(os.path.abspath(__file__))
    figs_dir = os.path.join(script_dir, "figs")
    create_folder(figs_dir)
    
    # qm
    arrays = (all_qm_train_data[t,:,:], point_qm_train_data[t,:,:], era5_train_data[t,:,:], cpc_train_data[t,:,:])
    extents = (extent, extent, extent, extent)
    titles = ("QM (all)", "QM (point-to-point)", "ERA5", "CPC")
    fig, _ = plot_maps(arrays, titles, extents)
    fig.savefig(os.path.join(figs_dir, "train_maps.png"))

    
    # Plot CDFs and PDFs
    titles = (
        "QM (all)",
        "QM (point-to-point)",
        "ERA5",
        "CPC",
        # "Generalized Gamma (point-to-point)",
    )
    arrays = (
        all_qm_train_data,
        point_qm_train_data,
        era5_train_data,
        cpc_train_data,
        # point_gengamma_train_data,
    )
    
    fig, _ = plot_cdfs(arrays, titles)
    fig.savefig(os.path.join(figs_dir, "cdf_train.png"))
    
    fig, _ = plot_pdfs(arrays, titles)
    fig.savefig(os.path.join(figs_dir, "pdf_train.png"))
    
    
    # Read test data
    test_data_dir = "/work/FAC/FGSE/IDYST/tbeucler/downscaling/mlima/data/test_data"
    with h5py.File(os.path.join(test_data_dir, "era5_low-pass.h5"), "r") as f:
        era5_test_data = f["precip"][:,:,:]
    test_times = xr.open_dataset(os.path.join(test_data_dir, "era5_low-pass.h5")).time.values
        
    with h5py.File(os.path.join(test_data_dir, "cpc.h5"), "r") as f:
        cpc_test_data = f["precip"][:,:,:]
        
    # Filter negative values
    era5_test_data[era5_test_data < 0] = 0
        
    # Correct bias with quantile mapping
    all_qm_test_data = all_qm.predict(era5_test_data)
    point_qm_test_data = point_qm.predict(era5_test_data)
    # point_gengamma_test_data = point_gengamma.predict(era5_test_data)
    
    # Plot maps
    t = -10
    
    # qm
    arrays = (all_qm_test_data[t,:,:], point_qm_test_data[t,:,:], era5_test_data[t,:,:], cpc_test_data[t,:,:])
    fig, _ = plot_maps(arrays, titles, extents)
    fig.savefig(os.path.join(figs_dir, "test_maps.png"))
    
    # Plot CDFs and PDFs
    titles = (
        "QM (all)", 
        "QM (point-to-point)",
        "ERA5",
        "CPC",
        # "Generalized Gamma (point-to-point)"
    )
    arrays = (
        all_qm_test_data,
        point_qm_test_data,
        era5_test_data,
        cpc_test_data,
        # point_gengamma_test_data,
    )
    
    fig, _ = plot_cdfs(arrays, titles)
    fig.savefig(os.path.join(figs_dir, "cdf_test.png"))
    
    fig, _ = plot_pdfs(arrays, titles)
    fig.savefig(os.path.join(figs_dir, "pdf_test.png"))
    
    # Save data
    # train
    write_dataset(train_times, lats, lons, all_qm_train_data, os.path.join(train_data_dir, "era5_qm_all.h5"))
    write_dataset(train_times, lats, lons, point_qm_train_data, os.path.join(train_data_dir, "era5_qm_point.h5"))
    # test
    write_dataset(test_times, lats, lons, all_qm_test_data, os.path.join(test_data_dir, "era5_qm_all.h5"))
    write_dataset(test_times, lats, lons, point_qm_test_data, os.path.join(test_data_dir, "era5_qm_point.h5"))
    
    
if __name__ == "__main__":
    main()
