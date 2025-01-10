import os, tomllib
import matplotlib.pyplot as plt
import numpy as np

from data.surface_data import SurfaceData, ForecastSurfaceData, ForecastEnsembleSurfaceData
from engineering.spectrum import get_1dpsd
from evaluation.plots import CURVE_CMAP as cmap


def plot_psds(cpc, det, ens, lambda_star, psd_star, figs_dir):
    for lead_time_idx in range(3):
        spatial_lenghts = cpc.get_spatial_lengths()
        k, cpc_psd = get_1dpsd(cpc.precip, *spatial_lenghts)
        _, det_psd = get_1dpsd(det.precip[lead_time_idx], *spatial_lenghts)
        _, ens_psd = get_1dpsd(
            np.mean(ens.precip[lead_time_idx], axis=0), *spatial_lenghts,
        )
        wavelengths = 2*np.pi / k[::-1]
        
        # get nyquist wavelnegths
        nyquist_wavelngths = 2*np.pi / (2 * spatial_lenghts[0]), 2*np.pi / (2 * spatial_lenghts[1])
        
        # mask wavelengths above nyquist
        mask = wavelengths > max(nyquist_wavelngths)
        
        fig, ax = plt.subplots(figsize=(8, 5))
        
        ax.plot(wavelengths[mask], det_psd[mask][::-1], label="det-s2s", color=cmap(5))
        ax.plot(wavelengths[mask], ens_psd[mask][::-1], label="ens-s2s", color=cmap(6))
        ax.plot(wavelengths[mask], cpc_psd[mask][::-1], label="CombiPrecip", color=cmap(0))
        ax.legend()
        
        if lambda_star is not None:
            ax.axvline(lambda_star, color='black', linestyle='--', label=r'$\lambda^*$')
            ax.text(lambda_star*1.1, 1e-5, r'$\lambda^\star$', fontsize='large')
    
        if psd_star is not None:
            ax.axhline(psd_star, color='black', linestyle='--', label=r'$\sigma^*$')
            ax.text(1e1, psd_star*1.1, r'$\sigma^\star$', fontsize='large')
        
        ax.set_xscale("log")
        ax.set_yscale("log")    
        ax.set_xlabel("Wavelengths (km)")
        ax.set_ylabel("Power spectral density")
        
        lead_time_name = det.lead_time[lead_time_idx].replace(" ", "-")
        plt.title(f"lead time = {lead_time_name}")
        fig.savefig(os.path.join(figs_dir, f"psd_{lead_time_name}.png"))
        

def print_target_noise(psd_star, cpc):
    Nx, Ny = cpc.latitude.size, cpc.longitude.size
    sigma_star = np.sqrt(psd_star * Nx * Ny)
    print("Target noise:", sigma_star)


def main():
    # directory paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(script_dir, "../dirs.toml"), "rb") as f:
        dirs = tomllib.load(f)
    base = dirs["main"]["base"]
    test_data_dir = os.path.join(base, dirs["subs"]["test"])
    
    # extra configuration
    cpc_path = os.path.join(test_data_dir, "cpc.h5")
    det_path = os.path.join(test_data_dir, "det_s2s_nearest_low-pass.h5")
    ens_path = os.path.join(test_data_dir, "ens_s2s_nearest_low-pass.h5")
    
    cpc = SurfaceData.load_from_h5(cpc_path, ["precip"])
    det = ForecastSurfaceData.load_from_h5(det_path, ["precip"])
    ens = ForecastEnsembleSurfaceData.load_from_h5(ens_path, ["precip"])
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    figs_dir = os.path.join(script_dir, "tuning_figs")
    os.makedirs(figs_dir, exist_ok=True)
    
    # to be tuned
    lambda_star = 1.45e3
    psd_star = 3.2e-2
    
    # main calls
    plot_psds(cpc, det, ens, lambda_star, psd_star, figs_dir)
    print("psds saved")
    print_target_noise(psd_star, cpc)
    
    
if __name__ == "__main__":
    main()
