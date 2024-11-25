import os, ml_collections

def get_config():
    config = ml_collections.ConfigDict()
    
    base = "/work/FAC/FGSE/IDYST/tbeucler/downscaling/mlima/"
    config.train_data_dir = os.path.join(base, "data/era5/train_data")
    config.validation_data_dir = os.path.join(base, "data/era5/validation_data")
    config.test_data_dir = os.path.join(base, "data/era5/test_data")
    config.simulations_dir = os.path.join(base, "data/era5/simulations")
    
    return config
