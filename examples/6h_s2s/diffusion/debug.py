import os, tomllib
import matplotlib.pyplot as plt

from data.surface_data import ForecastSurfaceData, ForecastEnsembleSurfaceData


def plot_means(det_s2s, ens_s2s, det_diff, ens_diff, figs_dir):
    num_members = ens_s2s.precip.shape[1]
    det_s2s_mean = det_s2s.precip.mean()
    det_s2s_means = [det_s2s_mean for _ in range(num_members)]
    ens_s2s_means = ens_s2s.precip.mean(axis=(0, 2, 3, 4))
    det_diff_means = det_diff.precip.mean(axis=(0, 2, 3, 4))
    ens_diff_means = ens_diff.precip.mean(axis=(0, 2, 3, 4))
    
    fig, ax = plt.subplots()
    ax.plot(
        range(1, num_members+1), det_s2s_means, 
        label="det-s2s", color="tab:blue", linestyle="--",
    )
    ax.plot(
        range(1, num_members+1), ens_s2s_means,
        label="ens-s2s", color="tab:orange", linestyle="--",
    )
    ax.plot(
        range(1, num_members+1), det_diff_means,
        label="det-diff", color="tab:blue",
    )
    ax.plot(
        range(1, num_members+1), ens_diff_means,
        label="ens-diff", color="tab:orange",
    )
    ax.set_xlabel("Ensemble member")
    ax.set_ylabel("Precipitation mean (mm/h)")
    ax.legend()
    fig.savefig(os.path.join(figs_dir, "means.png"))
    
    
def print_stats(det_s2s, ens_s2s, det_diff, ens_diff):
    # means
    det_s2s_mean = det_s2s.precip.mean()
    ens_s2s_mean = ens_s2s.precip.mean()
    det_diff_mean = det_diff.precip.mean()
    ens_diff_mean = ens_diff.precip.mean()
    print("Means:")
    print(f"det-s2s: {det_s2s_mean}")
    print(f"ens-s2s: {ens_s2s_mean}")
    print(f"det-diff: {det_diff_mean}")
    print(f"ens-diff: {ens_diff_mean}")
    
    # stds
    det_s2s_std = det_s2s.precip.std()
    ens_s2s_std = ens_s2s.precip.std()
    det_diff_std = det_diff.precip.std()
    ens_diff_std = ens_diff.precip.std()
    print("Stds:")
    print(f"det-s2s: {det_s2s_std}")
    print(f"ens-s2s: {ens_s2s_std}")
    print(f"det-diff: {det_diff_std}")
    print(f"ens-diff: {ens_diff_std}")


def run_debug(det_s2s, ens_s2s, det_diff, ens_diff, figs_dir):
    plot_means(det_s2s, ens_s2s, det_diff, ens_diff, figs_dir)
    print_stats(det_s2s, ens_s2s, det_diff, ens_diff)


def main():
    # directory paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(script_dir, "../dirs.toml"), "rb") as f:
        dirs = tomllib.load(f)
    base = dirs["main"]["base"]
    test_data_dir = os.path.join(base, dirs["subs"]["test"])
    simulations_dir = os.path.join(base, dirs["subs"]["simulations"])
    
    # extra configurations
    det_s2s_path = os.path.join(test_data_dir, "det_s2s_nearest_low-pass.h5")
    ens_s2s_path = os.path.join(test_data_dir, "ens_s2s_nearest_low-pass.h5")
    cli = 50
    det_diff_path = os.path.join(
        simulations_dir, f"diffusion/det_light_cli{cli}_ens50.h5",
    )
    ens_diff_path = os.path.join(
        simulations_dir, f"diffusion/ens_light_cli{cli}_ens50.h5",
    )
    # time_idx for snapshots
    time_idxs = [i for i in range(16)]
    # ensemble member for snapshots
    num_idx = 25
    script_dir = os.path.dirname(os.path.abspath(__file__))
    figs_dir = os.path.join(script_dir, "figs/debug")
    os.makedirs(figs_dir, exist_ok=True)

    # surface data
    det_s2s = ForecastSurfaceData.load_from_h5(det_s2s_path, ["precip"])
    ens_s2s = ForecastEnsembleSurfaceData.load_from_h5(ens_s2s_path, ["precip"])
    det_diff = ForecastEnsembleSurfaceData.load_from_h5(det_diff_path, ["precip"])
    ens_diff = ForecastEnsembleSurfaceData.load_from_h5(ens_diff_path, ["precip"])
    
    # main call
    run_debug(det_s2s, ens_s2s, det_diff, ens_diff, figs_dir)


if __name__ == "__main__":
    main()
