import os

import numpy as np
import tomllib

from data.surface_data import (
    ForecastEnsembleSurfaceData,
    ForecastSurfaceData,
    SurfaceData,
)
from evaluation.metrics import crps, psd_distance, wasserstein_distance

EVENT_TIME_LENGTH = 8
NUM_EVENTS = 2


def get_ss(skill, base_skill):
    return np.round(1 - skill / base_skill, 2)


def print_metric(
    climatology,
    s2s_det,
    s2s_ens,
    diff_det,
    diff_ens,
    wrf,
    cpc,
    name,
    metric,
    *args,
):
    print(f"Metric: {name}")
    for i, lead_time in enumerate(s2s_det.lead_time):
        print(f"Lead time: {lead_time}", flush=True)
        for e in range(NUM_EVENTS):
            print(f"  {3 * e + 2018} event:")
            time_idxs = slice(e * EVENT_TIME_LENGTH, (e + 1) * EVENT_TIME_LENGTH)
            climatology_arr = climatology.precip[time_idxs]
            cpc_arr = cpc.precip[time_idxs]
            base_skill = metric(cpc_arr, climatology_arr, *args)

            s2s_det_arr = s2s_det.precip[i, time_idxs]
            s2s_ens_arr = s2s_ens.precip[i, :, time_idxs]
            diff_det_arr = diff_det.precip[i, :, time_idxs]
            diff_ens_arr = diff_ens.precip[i, :, time_idxs]
            wrf_arr = wrf.precip[i, :, time_idxs]

            s2s_det_skill = metric(cpc_arr, s2s_det_arr, *args)
            print(f"    S2S-det: {get_ss(s2s_det_skill, base_skill)}", flush=True)
            s2s_ens_skill = metric(cpc_arr, s2s_ens_arr, *args)
            print(f"    S2S-ens: {get_ss(s2s_ens_skill, base_skill)}", flush=True)
            diff_det_skill = metric(cpc_arr, diff_det_arr, *args)
            print(f"    diff-det: {get_ss(diff_det_skill, base_skill)}", flush=True)
            diff_ens_skill = metric(cpc_arr, diff_ens_arr, *args)
            print(f"    diff-ens: {get_ss(diff_ens_skill, base_skill)}", flush=True)
            wrf_skill = metric(cpc_arr, wrf_arr, *args)
            print(f"    WRF: {get_ss(wrf_skill, base_skill)}", flush=True)
    print()


def evaluate(climatology, s2s_det, s2s_ens, diff_det, diff_ens, wrf, cpc):
    # crps
    print_metric(
        climatology,
        s2s_det,
        s2s_ens,
        diff_det,
        diff_ens,
        wrf,
        cpc,
        "CRPS",
        crps,
    )

    # wasserstein_distance
    print_metric(
        climatology,
        s2s_det,
        s2s_ens,
        diff_det,
        diff_ens,
        wrf,
        cpc,
        "Wasserstein distance",
        wasserstein_distance,
    )

    # psd distance
    spatial_lengths = s2s_det.get_spatial_lengths()
    print_metric(
        climatology,
        s2s_det,
        s2s_ens,
        diff_det,
        diff_ens,
        wrf,
        cpc,
        "PSD distance",
        psd_distance,
        *spatial_lengths,
    )


def main():
    # directory paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(script_dir, "../dirs.toml"), "rb") as f:
        dirs = tomllib.load(f)
    base = dirs["main"]["base"]
    simulations_dir = os.path.join(base, dirs["subs"]["simulations"])
    test_data_dir = os.path.join(base, dirs["subs"]["test"])

    # file paths
    s2s_det_path = os.path.join(test_data_dir, "det_s2s_nearest.h5")
    s2s_ens_path = os.path.join(test_data_dir, "ens_s2s_nearest.h5")
    cli = 50
    num_members = 50
    diff_det_path = os.path.join(
        simulations_dir,
        # changed to new
        f"diffusion/det_heavy_cli{cli}_ens{num_members}_new.h5",
    )
    diff_ens_path = os.path.join(
        simulations_dir,
        # changed to new
        f"diffusion/ens_heavy_cli{cli}_ens{num_members}_new.h5",
    )
    wrf_path = os.path.join(simulations_dir, "wrf", "wrf.h5")
    test_cpc_path = os.path.join(test_data_dir, "cpc.h5")
    climatology_path = os.path.join(test_data_dir, "climatology.h5")

    # surface data
    climatology = SurfaceData.load_from_h5(climatology_path, ["precip"])
    s2s_det = ForecastSurfaceData.load_from_h5(s2s_det_path, ["precip"])
    s2s_ens = ForecastEnsembleSurfaceData.load_from_h5(s2s_ens_path, ["precip"])
    diff_det = ForecastEnsembleSurfaceData.load_from_h5(diff_det_path, ["precip"])
    diff_ens = ForecastEnsembleSurfaceData.load_from_h5(diff_ens_path, ["precip"])
    wrf = ForecastEnsembleSurfaceData.load_from_h5(wrf_path, ["precip"])
    cpc = SurfaceData.load_from_h5(test_cpc_path, ["precip"])

    # print shapes:
    print("climatology:", climatology.precip.shape)
    print("s2s det:", s2s_det.precip.shape)
    print("s2s ens:", s2s_ens.precip.shape)
    print("diff det:", diff_det.precip.shape)
    print("diff ens:", diff_ens.precip.shape)
    print("wrf:", wrf.precip.shape)
    print("cpc:", cpc.precip.shape)

    evaluate(climatology, s2s_det, s2s_ens, diff_det, diff_ens, wrf, cpc)


if __name__ == "__main__":
    main()
