import os, tomllib
import numpy as np

from data.surface_data import ForecastSurfaceData, ForecastEnsembleSurfaceData

import configs
from gen_utils import generate


def preprocess_prior_ds(prior_file_path):
        # Read test surface data
        prior_sfc_data = ForecastSurfaceData.load_from_h5(prior_file_path, ["precip"])
        
        # Transform ForecastSurfaceData to jnp array and expand dims (numbers and channels)
        precip = np.expand_dims(prior_sfc_data.precip, axis=1)
        
        # Tranform ForecastSurfaceData in ForecastEnsembleSurfaceData with one ensemble
        prior_sfc_data = ForecastEnsembleSurfaceData(
            lead_time=prior_sfc_data.lead_time,
            number=np.array([0]),
            time=prior_sfc_data.time,
            latitude=prior_sfc_data.latitude,
            longitude=prior_sfc_data.longitude,
            precip=precip,
        )
        
        return prior_sfc_data
    

def main():
    # directory paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(script_dir, "../dirs.toml"), "rb") as f:
        dirs = tomllib.load(f)
    base = dirs["main"]["base"]
    train_data_dir = os.path.join(base, dirs["subs"]["train"])
    validation_data_dir = os.path.join(base, dirs["subs"]["validation"])
    test_data_dir = os.path.join(base, dirs["subs"]["test"])
    simulations_dir = os.path.join(base, dirs["subs"]["simulations"])
    
    # extra configurations
    model_config = configs.light_longer.get_config(train_data_dir, validation_data_dir)
    train_file_path = os.path.join(train_data_dir, model_config.train_file_name)
    prior_file_path = os.path.join(test_data_dir, "det_s2s_nearest_low-pass.h5")
    clip_max = 100
    num_samples = 50
    save_file_path = os.path.join(
        simulations_dir,
        "diffusion",
        f"det_{model_config.experiment_name}_cli{clip_max}_ens{num_samples}.h5",
    )
    
    # main call
    prior_sfc_data = preprocess_prior_ds(prior_file_path)
    gen_sfc_data = generate(
        model_config, 
        train_file_path, 
        prior_sfc_data, 
        clip_max, 
        num_samples,
        num_chunks=10,
    )
    gen_sfc_data.save_to_h5(save_file_path)


if __name__ == "__main__":
    main()
