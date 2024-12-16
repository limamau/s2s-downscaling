import os, tomllib
import numpy as np

from data.surface_data import ForecastEnsembleSurfaceData

import configs
from gen_utils import generate


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
    prior_file_path = os.path.join(test_data_dir, "ens_s2s_nearest_low-pass.h5")
    clip_max = 100
    num_samples = 1
    save_file_path = os.path.join(
        simulations_dir,
        "diffusion",
        f"ens_{model_config.experiment_name}_cli{clip_max}_ens{num_samples*50}.h5",
    )
    
    # main call
    prior_sfc_data = ForecastEnsembleSurfaceData.load_from_h5(prior_file_path, ["precip"])
    gen_sfc_data = generate(
        model_config, 
        train_file_path, 
        prior_sfc_data, 
        clip_max, 
        num_samples,
    )
    gen_sfc_data.save_to_h5(save_file_path)


if __name__ == "__main__":
    main()
