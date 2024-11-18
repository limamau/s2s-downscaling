import os, ml_collections, numpy as np

def get_config():
    config = ml_collections.ConfigDict()
    base = "/work/FAC/FGSE/IDYST/tbeucler/downscaling"
    raw_data_dir = os.path.join(base, "raw_data/S2S")
    config.test_data_dir = os.path.join(base, "mlima/data/test_data")
    config.storm_dates = (
        (np.datetime64('2018-06-11'), np.datetime64('2018-06-12')),
        # (datetime.datetime(2021, 6, 28), datetime.datetime(2018, 6, 29)),
    )
    # TODO: create tuple of tuples 
    config.storm_files = (
        os.path.join(raw_data_dir, "det_sfc_2018-05-21.nc"),
    )
    
    return config
