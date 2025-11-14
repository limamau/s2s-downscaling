import os

import numpy as np
import tomllib

from data.surface_data import SurfaceData

EVENT_TIME_LENGTH = 8


def main():
    # directory paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(script_dir, "../dirs.toml"), "rb") as f:
        dirs = tomllib.load(f)
    base = dirs["main"]["base"]
    train_data_dir = os.path.join(base, dirs["subs"]["train"])
    val_data_dir = os.path.join(base, dirs["subs"]["validation"])
    test_data_dir = os.path.join(base, dirs["subs"]["test"])

    # file paths
    train_cpc_path = os.path.join(train_data_dir, "cpc.h5")
    val_cpc_path = os.path.join(val_data_dir, "cpc.h5")
    test_cpc_path = os.path.join(test_data_dir, "cpc.h5")

    # climatology
    cpc_train = SurfaceData.load_from_h5(train_cpc_path, ["precip"])
    cpc_val = SurfaceData.load_from_h5(val_cpc_path, ["precip"])
    cpc_test = SurfaceData.load_from_h5(test_cpc_path, ["precip"])
    times = cpc_test.time
    lats = cpc_test.latitude
    lons = cpc_test.longitude
    climatology = np.repeat(
        np.expand_dims(
            np.mean(
                np.concatenate([cpc_train.precip, cpc_val.precip], axis=0),
                axis=0,
            ),
            axis=0,
        ),
        len(times),
        axis=0,
    )

    # save
    sfc_data = SurfaceData(times, lats, lons, precip=climatology)
    sfc_data.save_to_h5(os.path.join(test_data_dir, "climatology.h5"))


if __name__ == "__main__":
    main()
