import os

import ml_collections
import numpy as np


def get_config(raw_data_dir, test_data_dir):
    config = ml_collections.ConfigDict()
    config.storm_dates = (
        (np.datetime64("2018-06-11"), np.datetime64("2018-06-12")),
        (np.datetime64("2021-06-28"), np.datetime64("2021-06-29")),
    )
    config.oneday_files = {
        "det": (
            os.path.join(raw_data_dir, "hind_cf_tp_2018-06-11.grib"),
            os.path.join(raw_data_dir, "hind_cf_tp_2021-06-28.grib"),
        ),
        "ens": (
            os.path.join(raw_data_dir, "hind_pf_tp_2018-06-11.grib"),
            os.path.join(raw_data_dir, "hind_pf_tp_2021-06-28.grib"),
        ),
    }
    config.cpc_file = os.path.join(test_data_dir, "cpc.h5")

    return config
