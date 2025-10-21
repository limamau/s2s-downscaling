import os

import tomllib

from data.surface_data import (
    ForecastEnsembleSurfaceData,
    ForecastSurfaceData,
    SurfaceData,
)
from evaluation.metrics import crps, psd_distance, wasserstein_distance

EVENT_TIME_LENGTH = 8


def print_metric(s2s_det, s2s_ens, diff_det, diff_ens, wrf, cpc, name, metric, *args):
    print(f"Metric: {name}")
    for i, lead_time in enumerate(s2s_det.lead_time):
        print(f"Lead time: {lead_time}", flush=True)
        print(
            f"  S2S-det: {metric(cpc.precip, s2s_det.precip[i], *args)}",
            flush=True,
        )
        print(
            f"  S2S-ens: {metric(cpc.precip, s2s_ens.precip[i], *args)}",
            flush=True,
        )
        print(
            f"  diff-det: {metric(cpc.precip, diff_det.precip[i], *args)}",
            flush=True,
        )
        print(
            f"  diff-ens: {metric(cpc.precip, diff_ens.precip[i], *args)}",
            flush=True,
        )
        print(
            f"  WRF: {metric(cpc.precip, wrf.precip[i], *args)}",
            flush=True,
        )
    print()


def evaluate(s2s_det, s2s_ens, diff_det, diff_ens, wrf, cpc):
    # crps
    print_metric(
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
        s2s_det,
        s2s_ens,
        diff_det,
        diff_ens,
        wrf,
        cpc,
        "Wasserstein distance",
        wasserstein_distance,
        EVENT_TIME_LENGTH,
    )

    # psd distance
    spatial_lengths = s2s_det.get_spatial_lengths()
    print_metric(
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
    test_data_dir = os.path.join(base, dirs["subs"]["test"])
    simulations_dir = os.path.join(base, dirs["subs"]["simulations"])

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
    cpc_path = os.path.join(test_data_dir, "cpc.h5")

    # surface data
    s2s_det = ForecastSurfaceData.load_from_h5(s2s_det_path, ["precip"])
    s2s_ens = ForecastEnsembleSurfaceData.load_from_h5(s2s_ens_path, ["precip"])
    diff_det = ForecastEnsembleSurfaceData.load_from_h5(diff_det_path, ["precip"])
    diff_ens = ForecastEnsembleSurfaceData.load_from_h5(diff_ens_path, ["precip"])
    wrf = ForecastEnsembleSurfaceData.load_from_h5(wrf_path, ["precip"])
    cpc = SurfaceData.load_from_h5(cpc_path, ["precip"])

    # print shapes:
    print("s2s det:", s2s_det.precip.shape)
    print("s2s ens:", s2s_ens.precip.shape)
    print("diff det:", diff_det.precip.shape)
    print("diff ens:", diff_ens.precip.shape)
    print("wrf:", wrf.precip.shape)
    print("cpc:", cpc.precip.shape)

    evaluate(s2s_det, s2s_ens, diff_det, diff_ens, wrf, cpc)


if __name__ == "__main__":
    main()
