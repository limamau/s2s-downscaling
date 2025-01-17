import os, ml_collections

def get_config():
    config = ml_collections.ConfigDict()
    config.experiment_name = "heavy"
    
    # File paths and keys
    config.train_file_name = "cpc.h5"
    config.workdir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        f"../experiments/{config.experiment_name}"
    )
    config.key = "precip"
    
    # Dataset std
    config.data_std = 0.31
    
    # Apply log transformation
    config.apply_log = False
    
    # Resolutions
    config.num_channels = (32, 64, 128, 256)
    config.downsample_ratio = (2, 2, 2, 2)
    config.num_blocks = 6
    
    # Training parameters
    config.num_train_steps = 1_000_000
    config.train_batch_size = 2
    config.eval_batch_size = 2
    config.initial_lr = 0.0
    config.peak_lr = 1e-4
    config.warmup_steps = 1000
    config.end_lr = 1e-6
    config.ema_decay = 0.99
    config.ckpt_interval = 100_000
    config.max_ckpt_to_keep = 5
    
    return config
