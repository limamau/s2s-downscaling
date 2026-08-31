import os

import matplotlib.pyplot as plt
import numpy as np
import tomllib
import xarray as xr
from scipy.fft import fft2, fftfreq, fftshift


# ----------------------------------------------------------------------
# 1. Load Total Precipitation (TP) from NetCDF
# ----------------------------------------------------------------------
def get_tp(filepath):
    ds = xr.open_dataset(filepath)
    var_name = None
    for v in ds.data_vars:
        if v.lower() == "tp":
            var_name = v
            break
    if var_name is None:
        raise ValueError(f"No 'tp' variable found in {filepath}")
    tp_data = ds[var_name].values
    lat = ds.latitude.values if "latitude" in ds else ds.lat.values
    lon = ds.longitude.values if "longitude" in ds else ds.lon.values
    if "time" in ds.dims:
        time = ds.time.values
    elif "step" in ds.dims:
        time = ds.step.values
    else:
        time = np.arange(tp_data.shape[0])
    tp_ds = xr.Dataset(
        {"tp": (("time", "latitude", "longitude"), tp_data)},
        coords={"time": time, "latitude": lat, "longitude": lon},
    )
    tp_ds.attrs["initial_date"] = os.path.basename(filepath).split("_")[2]
    return tp_ds


# ----------------------------------------------------------------------
# 2. Radial Power Spectral Density
# ----------------------------------------------------------------------
def radial_psd(data_2d, dx_km=None, dy_km=None):
    ny, nx = data_2d.shape
    data_2d = data_2d - np.mean(data_2d)
    f = fftshift(fft2(data_2d))
    power = np.abs(f) ** 2
    kx = fftshift(fftfreq(nx))
    ky = fftshift(fftfreq(ny))
    if dx_km is not None:
        kx = kx / dx_km
        ky = ky / dy_km
    kx_grid, ky_grid = np.meshgrid(kx, ky)
    k_rad = np.sqrt(kx_grid**2 + ky_grid**2)
    k_max = np.max(k_rad)
    nbins = min(nx, ny) // 2 + 1
    bins = np.linspace(0, k_max, nbins)
    bin_centers = (bins[1:] + bins[:-1]) / 2
    psd_rad = []
    for i in range(len(bin_centers)):
        mask = (k_rad >= bins[i]) & (k_rad < bins[i + 1])
        psd_rad.append(np.mean(power[mask]) if np.any(mask) else np.nan)
    return np.array(bin_centers), np.array(psd_rad)


# ----------------------------------------------------------------------
# 3. Plot PSD (original) – kept for reference
# ----------------------------------------------------------------------
def plot_psd(
    ds, event_label, lat_range, lon_range, ax=None, var_name="tp", time_avg=True
):
    ds = ds.sortby("latitude")
    lon = ds.longitude
    if lon.max() > 180:
        lon_converted = xr.where(lon > 180, lon - 360, lon)
        ds = ds.assign_coords(longitude=lon_converted)
        ds = ds.sortby("longitude")
    ds_sub = ds.sel(
        latitude=slice(lat_range[0], lat_range[1]),
        longitude=slice(lon_range[0], lon_range[1]),
    )
    if ds_sub.sizes["latitude"] == 0 or ds_sub.sizes["longitude"] == 0:
        raise ValueError("Empty domain!")
    if time_avg:
        ds_sub = ds_sub.mean(dim="time")
    data = ds_sub[var_name].values.squeeze()
    lat = ds_sub.latitude.values
    lon = ds_sub.longitude.values
    lat_mid = np.mean(lat_range)
    dx_km = 111.32 * np.cos(np.radians(lat_mid)) * np.abs(lon[1] - lon[0])
    dy_km = 111.32 * np.abs(lat[1] - lat[0])
    k, psd = radial_psd(data, dx_km=dx_km, dy_km=dy_km)
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    ax.loglog(k[1:], psd[1:], label=event_label, linewidth=2)
    ax.set_xlabel("Wavenumber (cycles km⁻¹)")
    ax.set_ylabel("Power Spectral Density")
    ax.grid(True, which="both", ls="--", alpha=0.5)
    return ax


# ----------------------------------------------------------------------
# 4. Compute band‑integrated PSD over time
# ----------------------------------------------------------------------
def compute_band_psd(ds, lat_range, lon_range, k_bands, var_name="tp", time_avg=False):
    ds = ds.sortby("latitude")
    lon = ds.longitude
    if lon.max() > 180:
        lon_converted = xr.where(lon > 180, lon - 360, lon)
        ds = ds.assign_coords(longitude=lon_converted)
        ds = ds.sortby("longitude")
    ds_sub = ds.sel(
        latitude=slice(lat_range[0], lat_range[1]),
        longitude=slice(lon_range[0], lon_range[1]),
    )
    if ds_sub.sizes["latitude"] == 0 or ds_sub.sizes["longitude"] == 0:
        raise ValueError("Empty domain!")
    lat = ds_sub.latitude.values
    lon = ds_sub.longitude.values
    lat_mid = np.mean(lat_range)
    dx_km = 111.32 * np.cos(np.radians(lat_mid)) * np.abs(lon[1] - lon[0])
    dy_km = 111.32 * np.abs(lat[1] - lat[0])
    if time_avg:
        data_mean = ds_sub[var_name].mean(dim="time").values.squeeze()
        k, psd = radial_psd(data_mean, dx_km, dy_km)
        result = {}
        for label, (kmin, kmax) in k_bands.items():
            mask = (k >= kmin) & (k <= kmax)
            if np.any(mask):
                result[label] = np.trapz(psd[mask], k[mask])
            else:
                result[label] = np.nan
        return result
    else:
        times = ds_sub.time.values
        band_values = {label: [] for label in k_bands.keys()}
        for t in times:
            data = ds_sub[var_name].sel(time=t).values.squeeze()
            k, psd = radial_psd(data, dx_km, dy_km)
            for label, (kmin, kmax) in k_bands.items():
                mask = (k >= kmin) & (k <= kmax)
                if np.any(mask):
                    integral = np.trapz(psd[mask], k[mask])
                    band_values[label].append(integral)
                else:
                    band_values[label].append(np.nan)
        result = {}
        for label, vals in band_values.items():
            result[label] = xr.DataArray(vals, coords={"time": times}, dims="time")
        return result


def preprocess_ds(ds):
    """Ensure latitude is ascending and longitude is -180..180."""
    ds = ds.sortby("latitude")
    lon = ds.longitude
    if lon.max() > 180:
        lon_converted = xr.where(lon > 180, lon - 360, lon)
        ds = ds.assign_coords(longitude=lon_converted)
        ds = ds.sortby("longitude")
    return ds


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(script_dir, "../dirs.toml"), "rb") as f:
        dirs = tomllib.load(f)
    raw_data_dir = dirs["raw"]["s2s"]

    file_2018 = os.path.join(raw_data_dir, "det_sfc_2018-06-11_tp.nc")
    file_2021 = os.path.join(raw_data_dir, "det_sfc_2021-06-28_tp.nc")

    print("Loading TP data...")
    tp_2018 = get_tp(file_2018)
    tp_2021 = get_tp(file_2021)

    # Preprocess (sort lat, convert lon to -180..180)
    tp_2018 = preprocess_ds(tp_2018)
    tp_2021 = preprocess_ds(tp_2021)

    # Domains
    europe_domain = {"lat": (35, 70), "lon": (-10, 40)}
    swiss_domain = {"lat": (45.8, 47.8), "lon": (5.9, 10.5)}

    # ------------------------------------------------------------------
    # Debug prints: domain sizes
    # ------------------------------------------------------------------
    def check_domain(ds, lat_range, lon_range, label):
        sub = ds.sel(
            latitude=slice(lat_range[0], lat_range[1]),
            longitude=slice(lon_range[0], lon_range[1]),
        )
        nlat = sub.sizes["latitude"]
        nlon = sub.sizes["longitude"]
        print(f"{label}: {nlat} lat × {nlon} lon")
        if nlat == 0 or nlon == 0:
            print(f"  WARNING: {label} domain is EMPTY!")
        else:
            print(
                f"  Lat range: {sub.latitude.values[0]:.2f}° to {sub.latitude.values[-1]:.2f}°"
            )
            print(
                f"  Lon range: {sub.longitude.values[0]:.2f}° to {sub.longitude.values[-1]:.2f}°"
            )
        return sub

    print("\n--- Checking domains after preprocessing ---")
    swiss_2018 = check_domain(
        tp_2018, swiss_domain["lat"], swiss_domain["lon"], "Switzerland 2018"
    )
    swiss_2021 = check_domain(
        tp_2021, swiss_domain["lat"], swiss_domain["lon"], "Switzerland 2021"
    )
    euro_2018 = check_domain(
        tp_2018, europe_domain["lat"], europe_domain["lon"], "Europe 2018"
    )
    euro_2021 = check_domain(
        tp_2021, europe_domain["lat"], europe_domain["lon"], "Europe 2021"
    )

    # If any domain is empty, abort
    if (
        swiss_2018.sizes["latitude"] == 0
        or swiss_2021.sizes["latitude"] == 0
        or euro_2018.sizes["latitude"] == 0
        or euro_2021.sizes["latitude"] == 0
    ):
        raise ValueError(
            "One or more domains are empty – check latitude/longitude ranges."
        )

    # ------------------------------------------------------------------
    # Wavenumber bands
    # ------------------------------------------------------------------
    bands_synoptic = {"synoptic": (0.003, 0.01)}  # 100–300 km
    bands_mesoscale = {"mesoscale": (0.01, 0.05)}  # 20–100 km

    print("\nComputing Synoptic PSD over Europe...")
    syn_2018 = compute_band_psd(
        tp_2018,
        europe_domain["lat"],
        europe_domain["lon"],
        bands_synoptic,
        var_name="tp",
        time_avg=False,
    )["synoptic"]
    syn_2021 = compute_band_psd(
        tp_2021,
        europe_domain["lat"],
        europe_domain["lon"],
        bands_synoptic,
        var_name="tp",
        time_avg=False,
    )["synoptic"]

    print("Computing Mesoscale PSD over Switzerland...")
    meso_2018 = compute_band_psd(
        tp_2018,
        swiss_domain["lat"],
        swiss_domain["lon"],
        bands_mesoscale,
        var_name="tp",
        time_avg=False,
    )["mesoscale"]
    meso_2021 = compute_band_psd(
        tp_2021,
        swiss_domain["lat"],
        swiss_domain["lon"],
        bands_mesoscale,
        var_name="tp",
        time_avg=False,
    )["mesoscale"]

    # ------------------------------------------------------------------
    # Debug prints: PSD values
    # ------------------------------------------------------------------
    print("\n--- PSD values (first 3 time steps) ---")
    print(f"meso_2018: {meso_2018.values[:3]}")
    print(f"meso_2021: {meso_2021.values[:3]}")
    print(f"syn_2018:  {syn_2018.values[:3]}")
    print(f"syn_2021:  {syn_2021.values[:3]}")

    if np.all(np.isnan(meso_2018.values)) or np.all(np.isnan(syn_2018.values)):
        raise ValueError(
            "PSD arrays contain only NaN – check wavenumber bands or domain size."
        )

    # ------------------------------------------------------------------
    # Area‑average TP over Switzerland
    # ------------------------------------------------------------------
    tp_swiss_2018 = (
        tp_2018["tp"]
        .sel(
            latitude=slice(swiss_domain["lat"][0], swiss_domain["lat"][1]),
            longitude=slice(swiss_domain["lon"][0], swiss_domain["lon"][1]),
        )
        .mean(dim=["latitude", "longitude"])
    )

    tp_swiss_2021 = (
        tp_2021["tp"]
        .sel(
            latitude=slice(swiss_domain["lat"][0], swiss_domain["lat"][1]),
            longitude=slice(swiss_domain["lon"][0], swiss_domain["lon"][1]),
        )
        .mean(dim=["latitude", "longitude"])
    )

    print("\n--- TP over Switzerland (first 3 values) ---")
    print(f"tp_2018: {tp_swiss_2018.values[:3]}")
    print(f"tp_2021: {tp_swiss_2021.values[:3]}")

    if np.all(np.isnan(tp_swiss_2018.values)) or np.all(np.isnan(tp_swiss_2021.values)):
        raise ValueError("TP data contains only NaN – check Switzerland selection.")

    # ------------------------------------------------------------------
    # Time axis and plotting
    # ------------------------------------------------------------------
    hours = np.arange(0, len(tp_swiss_2018.time) * 6, 6)

    figs_dir = os.path.join(script_dir, "figs")
    os.makedirs(figs_dir, exist_ok=True)

    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

    ax = axes[0]
    ax.plot(hours, meso_2018.values, "o-", label="2018")
    ax.plot(hours, meso_2021.values, "s-", label="2021")
    ax.set_ylabel("Mesoscale PSD\n(20–100 km, Switzerland)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(hours, syn_2018.values, "o-", label="2018")
    ax.plot(hours, syn_2021.values, "s-", label="2021")
    ax.set_ylabel("Synoptic PSD\n(100–300 km, Europe)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[2]
    ax.plot(hours, tp_swiss_2018.values, "o-", label="2018")
    ax.plot(hours, tp_swiss_2021.values, "s-", label="2021")
    ax.set_xlabel("Time (hours since event start)")
    ax.set_ylabel("TP (mm)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax.set_xticks(hours)
    plt.tight_layout()
    plt.savefig(os.path.join(figs_dir, "tp_swiss_time_series.png"), dpi=150)
    plt.show()
    print(f"\nFigure saved to {figs_dir}/tp_swiss_time_series.png")


if __name__ == "__main__":
    main()
