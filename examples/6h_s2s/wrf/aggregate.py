import os

import numpy as np
import tomllib

# from configs.single import get_config
from data.surface_data import ForecastEnsembleSurfaceData, SurfaceData

# date: [ensemble-members, ...]
AGGREGATION_LIST = [
    {
        "2018-06-04": [25, 38],
        "2018-05-28": [46, 45],
        "2018-05-21": [1, 14],
    },
    {
        "2021-06-21": [0, 11],
        "2021-06-14": [5, 26],
        "2021-06-07": [0, 17],
    }
]
LINES_UNTIL_EVENT_CHANGE = 3
LEAD_TIMES = ["1-week", "2-week", "3-week"]
NUMBERS = np.array([0, 1])


def aggregate_event(single_dir, aggregate_dir, aggregation_dict):
    first_number_flag = True
    first_lead_time_flag = True

    # lead-time aggregation loop
    for (date, ens_idxs) in aggregation_dict.items():
        # ensemble number aggregation loop
        for ens_idx in ens_idxs:
            single_sfc_data = SurfaceData.load_from_h5(
                os.path.join(single_dir, f"{date}_{ens_idx}.h5"),
                ["precip"],
            )
            if first_number_flag:
                first_number_flag = False
                time = single_sfc_data.time
                # add dimension for ensemble number
                number_precip = np.expand_dims(single_sfc_data.precip, axis=0)
            else:
                number_precip = np.concatenate(
                    (number_precip, np.expand_dims(single_sfc_data.precip, axis=0)),
                    axis=0,
                )
        if first_lead_time_flag:
            first_lead_time_flag = False
            # add dimension for forecast lead-time
            leadtime_number_precip = np.expand_dims(number_precip, axis=0)
        else:
            leadtime_number_precip = np.concatenate(
                (leadtime_number_precip, np.expand_dims(number_precip, axis=0)),
                axis=0,
            )
        first_number_flag = True

    # time, lat, lon, precip
    return time, single_sfc_data.latitude, single_sfc_data.longitude, leadtime_number_precip


def aggregate(single_dir, aggregate_dir):
    # aggregation for event 1
    time1, lat, lon, precip1 = aggregate_event(single_dir, aggregate_dir, AGGREGATION_LIST[0])
    # aggregation for event 2
    time2, _, _, precip2 = aggregate_event(single_dir, aggregate_dir, AGGREGATION_LIST[1])

    # concat time and precip
    time = np.concatenate([time1, time2])
    precip = np.concatenate([precip1, precip2], axis=2)

    # save
    wrf = ForecastEnsembleSurfaceData(
        lead_time=LEAD_TIMES,
        number=NUMBERS,
        time=time,
        latitude=lat,
        longitude=lon,
        precip=precip,
    )
    wrf.save_to_h5(os.path.join(aggregate_dir, "wrf.h5"))


def main():
    # directory paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(script_dir, "../dirs.toml"), "rb") as f:
        dirs = tomllib.load(f)
    base = dirs["main"]["base"]
    simulations_dir = os.path.join(base, dirs["subs"]["simulations"])
    # test_data_dir = os.path.join(base, dirs["subs"]["test"])

    # extra configurations
    # config = get_config()
    # cpc_file = os.path.join(test_data_dir, config.cpc_file)
    # cpc = SurfaceData.load_from_h5(cpc_file, ["precip"])
    # s2s_file = os.path.join(test_data_dir, config.s2s_file)
    # s2s = SurfaceData.load_from_h5(s2s_file, ["precip"])
    single_dir = os.path.join(simulations_dir, "wrf/single")
    aggregate_dir = os.path.join(simulations_dir, "wrf")

    # main calls
    aggregate(single_dir, aggregate_dir)
    print("done!")


if __name__ == "__main__":
    main()
