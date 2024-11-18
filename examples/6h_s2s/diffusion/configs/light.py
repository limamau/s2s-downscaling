import os, ml_collections

def get_config():
    config = ml_collections.ConfigDict()
    config.experiment_name = "light"
    
    # File paths and keys
    base = "/work/FAC/FGSE/IDYST/tbeucler/downscaling/mlima/"
    config.train_file_path = os.path.join(base, "data/train_data/cpc_6h_resampled.h5")
    config.validation_file_path = os.path.join(base, "data/validation_data/cpc_6h.h5")
    config.workdir = os.path.join(
        base, 
        f"s2s-downscaling/examples/6h_s2s/diffusion/experiments/{config.experiment_name}"
    )
    config.key = "precip"
    
    # Apply log transformation
    config.apply_log = False
    
    # Dataset std
    config.data_std = 0.31
    
    # Resolutions
    config.num_channels = (64, 128, 256)
    config.downsample_ratio = (2, 2, 2)
    config.num_blocks = 6
    
    # Training parameters
    config.num_train_steps = 500_000
    config.train_batch_size = 4
    config.eval_batch_size = 4
    config.initial_lr = 0.0
    config.peak_lr = 1e-4
    config.warmup_steps = 1000
    config.end_lr = 1e-6
    config.ema_decay = 0.99
    config.ckpt_interval = 10_000
    config.max_ckpt_to_keep = 5
    
    return config
