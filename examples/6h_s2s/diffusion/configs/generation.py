import os, ml_collections

def get_config():
    config = ml_collections.ConfigDict()
    base_dir = "/work/FAC/FGSE/IDYST/tbeucler/downscaling/mlima"
    config.prior_file_path = os.path.join(
        base_dir, 
        "data/test_data/s2s_nearest_low-pass.h5"
    )
    config.save_dir = os.path.join(base_dir, "data/simulations/s2s/diffusion")
    config.clip_max = 100
    config.num_samples = 4
    
    return config
