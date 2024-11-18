import os, ml_collections

def get_config():
    config = ml_collections.ConfigDict()
    base = "/work/FAC/FGSE/IDYST/tbeucler/downscaling"
    config.test_data_dir = os.path.join(base, "mlima/data/test_data")
    config.train_data_dir = os.path.join(base, "mlima/data/train_data")
    config.validation_data_dir = os.path.join(base, "mlima/data/validation_data")
    config.cpc_preprocessed_file_name = "cpc.h5"
    config.cpc_aggregated_file_name = "cpc_6h.h5"
    
    return config
