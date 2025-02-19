import os, tomllib

from data.surface_data import SurfaceData
from evaluation.plots import plot_maps

from configs.config import get_config


def run_analysis(wrf, cpc):
    for time_idx in range(0, len(wrf.time)):
        arrays = (
            wrf.precip[time_idx],
            cpc.precip[time_idx],
        )
        titles = (
            "WRF",
            "CombiPrecip"
        )
        cpc_extent = cpc.get_extent()
        extents = (cpc_extent,) * 2
        fig, _ = plot_maps(arrays, titles, extents)
        script_dir = os.path.dirname(os.path.realpath(__file__))
        figs_dir = os.path.join(script_dir, "figs", "analysis")
        os.makedirs(figs_dir, exist_ok=True)
        fig.savefig(os.path.join(figs_dir, f"maps_{time_idx}.png"))


def main():
    # directory paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(script_dir, "../dirs.toml"), "rb") as f:
        dirs = tomllib.load(f)
    base = dirs["main"]["base"]
    simulations_dir = os.path.join(base, dirs["subs"]["simulations"])
    test_data_dir = os.path.join(base, dirs["subs"]["test"])
    
    # extra configurations
    config = get_config()
    cpc_file = os.path.join(test_data_dir, config.cpc_file)
    cpc = SurfaceData.load_from_h5(cpc_file, ["precip"])
    wrf_simulations_dir = os.path.join(simulations_dir, config.wrf_simulations_dir)
    wrf = SurfaceData.load_from_h5(os.path.join(wrf_simulations_dir, "wrf.h5"), ["precip"])

    # main calls
    run_analysis(wrf, cpc)

if __name__ == '__main__':
    main()
