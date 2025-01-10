import os, tomllib
import numpy as np

from data.surface_data import SurfaceData


def main():
    # directory paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(script_dir, "../dirs.toml"), "rb") as f:
        dirs = tomllib.load(f)
    base = dirs["main"]["base"]
    test_data_dir = os.path.join(base, dirs["subs"]["test"])
    validation_data_dir = os.path.join(base, dirs["subs"]["validation"])
    train_data_dir = os.path.join(base, dirs["subs"]["train"])
    
    cpc = SurfaceData.load_from_h5(
        os.path.join(train_data_dir, "cpc.h5"),
        ["precip"],
    )
    
    # check for negative values
    check = (cpc.precip >= 0).all()
    if not check:
        print("There are negative values in the data.")
    
    # check for NaN values
    check = np.isnan(np.sum(cpc.precip))
    if check:
        print("There are NaN values in the data.")
    

if __name__ == '__main__':
    main()
