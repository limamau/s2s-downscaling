import numpy as np
from utils import normalize_data


def get_psd(data, x_length, y_length, parseval_check=False):
    # Get spatial dimensions
    Nt, Ny, Nx = data.shape
    
    # Normalize data
    data = normalize_data(data)[0]

    # Get radial wavenumber coordinate (from cartesian to polar coordinates)
    kx = np.fft.fftfreq(Nx, d=x_length/Nx)
    ky = np.fft.fftfreq(Ny, d=y_length/Ny)
    ky_grid, kx_grid = np.meshgrid(kx, ky) # yes it's reversed
    k_grid = np.sqrt(kx_grid**2 + ky_grid**2)
    
    # Define 1D wavenumber bins
    Nbins = int(np.sqrt(Nx*Ny)/2)
    # if Nbins > 100:
    #     Nbins -= 60 # to smooth things out
    eps = 1e-5
    k_bins = np.linspace(0, np.max(k_grid)+eps, Nbins)
    k_centers = 0.5 * (k_bins[1:] + k_bins[:-1])

    # Compute 2D Fourier Transform
    fft_data = np.fft.fft2(data, axes=(1,2))
    
    # Compute 1D PSD for each time step t
    ps_t = np.zeros([len(k_centers), Nt])
    for t in range(Nt):
        ps_t[:, t] = np.histogram(k_grid, bins=k_bins, weights=np.abs(fft_data[t, :, :])**2)[0]
    ps_t /= (Nx*Ny)**2
    
    # return: wavelengths, PSD(wavelengths)
    if not parseval_check:
        hist = np.histogram(k_grid, bins=k_bins)[0]
        mask = hist != 0
        psd_t = ps_t[mask,:] / hist[mask,None]
        psd = np.mean(psd_t, axis=1)
        return 2*np.pi/k_centers[::-1], psd[::-1]

    # return: k, PS(k, t)
    else:
        ps_2d_t = np.abs(fft_data)**2 / (Nx*Ny)**2
        return k_centers, ps_t, ps_2d_t
        

def get_k_star(wavelengths, era5_psd, cpc_psd, threshold=1e-4):
    # TODO: write exact method
    # Find indices where PSD values are below the threshold
    mask_era5 = era5_psd < threshold
    mask_cpc = cpc_psd < threshold

    # Select the PSD values and wavelengths that meet the condition
    era5_psd_below_threshold = era5_psd[mask_era5]
    cpc_psd_below_threshold = cpc_psd[mask_cpc]
    wavelengths_below_threshold = wavelengths[mask_era5 & mask_cpc]

    # Find the index where the two PSD curves intersect
    idx = np.argmin(np.abs(era5_psd_below_threshold - cpc_psd_below_threshold))

    # Return the wave number associated with the intersection point
    k_star = 2*np.pi / wavelengths_below_threshold[idx]
    return k_star


def get_std_star(data, psd_star):
    _, Ny, Nx = data.shape
    return np.sqrt(Nx*Ny * psd_star)


def apply_low_pass_filter(data, cutoff_k, x_length, y_length):
    # Get dimensions
    _, Ny, Nx = data.shape
    
    # Perform a 2D FFT on data
    fft_data = np.fft.fft2(data)

    # Calculate the frequencies corresponding to the FFT
    ky = np.fft.fftfreq(data.shape[1], d=x_length/Ny)
    kx = np.fft.fftfreq(data.shape[2], d=y_length/Nx)
    ky_2d, kx_2d = np.meshgrid(ky, kx, indexing='ij')

    # Tranformation to polar coordinates
    k_2d =  np.sqrt(ky_2d**2 + kx_2d**2)

    # Apply the low-pass filter
    filter_mask = k_2d < cutoff_k
    fft_data_filtered = fft_data * filter_mask

    # Perform an inverse 2D FFT to get the filtered data
    filtered_data = np.fft.ifft2(fft_data_filtered).real

    return filtered_data
    