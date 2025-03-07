import os, tomllib

from data.surface_data import SurfaceData, ForecastSurfaceData, ForecastEnsembleSurfaceData
from evaluation.metrics import crps, wasserstein_distance, psd_distance, rank_histogram
    
    
def plot_metric(s2s_det, s2s_ens, diff_det, diff_ens, cpc, name, metric):
    for i, lead_time in enumerate(s2s_det.lead_time):
        print(f"Lead time: {lead_time}")
        print(f"  S2S-det: {metric(cpc.precip, s2s_det.precip[i])}")
        print(f"  S2S-ens: {metric(cpc.precip, s2s_ens.precip[i])}")
        print(f"  diff-det: {metric(cpc.precip, diff_det.precip[i])}")
        print(f"  diff-ens: {metric(cpc.precip, diff_ens.precip[i])}")
    print()


def print_metric(s2s_det, s2s_ens, diff_det, diff_ens, cpc, name, metric, *args):
    print(f"Metric: {name}")
    for i, lead_time in enumerate(s2s_det.lead_time):
        print(f"Lead time: {lead_time}")
        print(f"  S2S-det: {metric(cpc.precip, s2s_det.precip[i], *args)}")
        print(f"  S2S-ens: {metric(cpc.precip, s2s_ens.precip[i], *args)}")
        print(f"  diff-det: {metric(cpc.precip, diff_det.precip[i], *args)}")
        print(f"  diff-ens: {metric(cpc.precip, diff_ens.precip[i], *args)}")
    print()
    
    
def plot_rank_histogram(s2s_ens, diff_det, diff_ens, cpc):
    for i, lead_time in enumerate(s2s_ens.lead_time):
        pass


def evaluate(s2s_det, s2s_ens, diff_det, diff_ens, cpc):
    # crps
    print_metric(
        s2s_det, s2s_ens, diff_det, diff_ens, cpc,
        "CRPS", crps,
    )
    
    # wasserstein_distance
    print_metric(
        s2s_det, s2s_ens, diff_det, diff_ens, cpc,
        "Wasserstein distance", wasserstein_distance,
    )
    
    # psd distance
    spatial_lengths = s2s_det.get_spatial_lengths()
    print_metric(
        s2s_det, s2s_ens, diff_det, diff_ens, cpc,
        "PSD distance", psd_distance, *spatial_lengths,
    )
    
    # rank histogram
    plot_rank_histogram(
        s2s_ens, diff_det, diff_ens, cpc,
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
    diff_det_path = os.path.join(
        simulations_dir, f"diffusion/det_heavy_cli{cli}_ens50.h5",
    )
    diff_ens_path = os.path.join(
        simulations_dir, f"diffusion/ens_heavy_cli{cli}_ens50.h5",
    )
    cpc_path = os.path.join(test_data_dir,"cpc.h5")
    
    # surface data
    cpc = SurfaceData.load_from_h5(cpc_path, ["precip"])
    s2s_det = ForecastSurfaceData.load_from_h5(s2s_det_path, ["precip"])
    s2s_ens = ForecastEnsembleSurfaceData.load_from_h5(s2s_ens_path, ["precip"])
    diff_det = ForecastEnsembleSurfaceData.load_from_h5(diff_det_path, ["precip"])
    diff_ens = ForecastEnsembleSurfaceData.load_from_h5(diff_ens_path, ["precip"])
    
    evaluate(s2s_det, s2s_ens, diff_det, diff_ens, cpc)


if __name__ == '__main__':
    main()
