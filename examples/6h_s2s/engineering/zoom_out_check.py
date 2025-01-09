import imageio, os, tomllib
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib import pyplot as plt
import matplotlib.colors as mcolors

from data.surface_data import SurfaceData

from engineering_utils import aggregate_det_s2s_precip, aggregate_single_ens_s2s_precip
from configs import det_s2s, ens_s2s


CUSTOM_PRECIP_COLORS = [
    '#000000',  # Black
    '#808080',  # Medium gray
    '#A9A9A9',  # Light gray
    '#FFFFFF',  # White
    '#FFFFCC',  # Light Yellow
    '#C7E9B4',  # Light Green
    '#7FCDBB',  # Moderate Blue-green
    '#41B6C4',  # Moderate Blue
    '#1D91C0',  # Blue
    '#225EA8',  # Darker Blue
    '#253494',  # Dark Blue
    '#54278F',  # Purple
    '#7A0177',  # Dark
    '#C51B8A',  # Pink
]
PRECIP_CMAP = mcolors.ListedColormap(CUSTOM_PRECIP_COLORS)
CUSTOM_VALUES = [-5, -2, -1, 0, 0.1, 0.5, 1, 2, 2.5, 5, 10, 20, 30, 50]
CUSTOM_NORM = mcolors.BoundaryNorm(CUSTOM_VALUES, len(CUSTOM_VALUES)-1)


def plot_maps(
    arrays, 
    titles, 
    extents, 
    projections=(ccrs.PlateCarree(),)*4,
    cmap=PRECIP_CMAP,
    norm=CUSTOM_NORM,
    vmin=None,
    vmax=None,
    cbar_label="Precipitation (mm)",
    main_title=None,
    figsize=(8, 6),
):
    fig, axs = plt.subplots(
        2, 2,
        figsize=figsize,
        dpi=300,
        subplot_kw={'projection': ccrs.PlateCarree()},
    )

    for i, ax in enumerate(axs.flat):
        img = ax.imshow(
            arrays[i],
            origin='lower', 
            extent=extents[i], 
            transform=projections[i],
            cmap=cmap,
            norm=norm,
            vmin=vmin, 
            vmax=vmax,
        )
        ax.add_feature(cfeature.BORDERS)
        ax.set_title(titles[i])
        ax.set_frame_on(False)

    fig.subplots_adjust(
        left=0.1,
        right=0.85,
        bottom=0.03,
        top=1.03,
        wspace=0.1,
        hspace=-0.2,
    )
    
    # Add a main title
    if main_title:
        print(main_title)
        fig.suptitle(main_title, y=1.00, fontsize=16)
    
    # Create a single axis for the colorbar
    cbar_ax = fig.add_axes([0.89, 0.5-0.5/2, 0.02, 1/2])
    fig.colorbar(img, cax=cbar_ax, label=cbar_label)

    return fig, axs


def plot_zoomed_out_gifs(
    s2s, cpc,
    time_idxs,
    gifs_dir,
    event_length=8,
):
    os.makedirs(gifs_dir, exist_ok=True)
    image_paths = []

    # save temporary images for each time_idx
    for time_idx in time_idxs:
        arrays = (
            s2s.precip[0, time_idx],
            s2s.precip[1, time_idx],
            s2s.precip[2, time_idx],
            cpc.precip[time_idx],
        )
        titles = (
            "1-week forecast",
            f"2-week forecast",
            f"3-week forecast",
            "CombiPrecip",
        )
        extents = (
            s2s.get_extent(),
            s2s.get_extent(),
            s2s.get_extent(),
            cpc.get_extent(),
        )
        fig, _ = plot_maps(arrays, titles, extents, main_title=f"{cpc.time[time_idx]}")

        image_path = os.path.join(gifs_dir, f"temp_map_t{time_idx}.png")
        fig.savefig(image_path)
        plt.close(fig) # important to avoid memory leak
        image_paths.append(image_path)

    # create the GIFs
    for event_idx in (1, 2):
        gif_path = os.path.join(gifs_dir, f"zoom_out_map_e{event_idx}.gif")
        if os.path.exists(gif_path):
            os.remove(gif_path)
        with imageio.get_writer(gif_path, mode='I', duration=1000) as writer:
            for image_path_idx in range((event_idx-1)*event_length, event_idx*event_length):
                image_path = image_paths[image_path_idx]
                image = imageio.v2.imread(image_path)
                writer.append_data(image)

    # clean up temporary files
    for image_path in image_paths:
        os.remove(image_path)

def run_det_engineering(storm_dates, lead_time_files, extent):
    # load S2S
    s2s = aggregate_det_s2s_precip(lead_time_files, storm_dates)
    # check negative values in aggregation
    print("Checking negative values in aggregation...")
    check_negative_values(s2s.precip)
    s2s.cut_data(extent)
    # check negative values in cut data
    print("Checking negative values in cut data...")
    check_negative_values(s2s.precip)
    s2s.unflip_latlon()
    # check negative values in unflipped data
    print("Checking negative values in unflipped data...")
    check_negative_values(s2s.precip)
    print(f"Cut S2S data shape: {s2s.precip.shape}")
    
    return s2s


def run_single_ens_engineering(storm_dates, lead_time_files, extent, num_idx):
    # Load S2S
    s2s = aggregate_single_ens_s2s_precip(lead_time_files, storm_dates, num_idx)
    # check negative values in aggregation
    print("Checking negative values in aggregation...")
    check_negative_values(s2s.precip)
    s2s.cut_data(extent)
    # check negative values in cut data
    print("Checking negative values in cut data...")
    check_negative_values(s2s.precip)
    s2s.unflip_latlon()
    # check negative values in unflipped data
    print("Checking negative values in unflipped data...")
    check_negative_values(s2s.precip)
    print(f"Cut S2S data shape: {s2s.precip.shape}")
    
    return s2s


def check_negative_values(data):
    if (data < 0).any():
        print("Negative values found.")
    else:
        print("Only positive values.")


def main():
    # directory paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(script_dir, "../dirs.toml"), "rb") as f:
        dirs = tomllib.load(f)
    base = dirs["main"]["base"]
    raw_data_dir = dirs["raw"]["s2s"]
    test_data_dir = os.path.join(base, dirs["subs"]["test"])
    
    # extra configurations
    config = ens_s2s.get_config(raw_data_dir, test_data_dir)
    storm_dates = config.storm_dates
    lead_time_files = config.lead_time_files
    cpc_file = os.path.join(test_data_dir, "cpc.h5")
    extent = (-10, 20, 40, 53)
    cpc = SurfaceData.load_from_h5(cpc_file, ["precip"])
    # os.makedirs("check", exist_ok=True)
    time_idxs = [i for i in range(16)]
    gifs_dir = os.path.join(script_dir, "gifs")
    num_idx = 0
    
    # main calls
    # s2s = run_det_engineering(
    #     storm_dates, lead_time_files, extent,
    # )
    s2s = run_single_ens_engineering(
        storm_dates, lead_time_files, extent, num_idx,
    )
    print("s2s:")
    check_negative_values(s2s.precip)
    # s2s.save_to_h5(os.path.join("check", "zoom_out_check.h5"))
    # print("zoomed out data saved")
    print("cpc:")
    check_negative_values(cpc.precip)
    plot_zoomed_out_gifs(
        s2s, cpc,
        time_idxs,
        gifs_dir,
    )
    print("maps saved")

if __name__ == "__main__":
    main()
