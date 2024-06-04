import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.colors as mcolors

from utils import get_cdf
from engineering.spectrum import get_psd


def _add_swiss_latlon_labels(ax, plons, plats):
    if plons is not None:
        ax.set_xlabel('Longitude')
        ax.set_xticks(plons)
        ax.set_xticklabels([str(c)+"º" for c in plons])
    if plats is not None:
        ax.set_ylabel('Latitude')
        ax.set_yticks(plats)
        ax.set_yticklabels([str(c)+"º" for c in plats])
        
        
def _add_swiss_xy_labels(ax, xp, yp):
    if xp is not None:
        ax.set_xlabel(r'Swiss-X ($10^5$m)')
        ax.set_xticks(xp)
        ax.set_xticklabels([str(int(c/10**5)) for c in xp])
    if yp is not None:
        ax.set_ylabel(r'Swiss-Y ($10^5$m)')
        ax.set_yticks(yp)
        ax.set_yticklabels([str(int(c/10**5)) for c in yp])
        
        
def _write_label(ax, label):
    plons = None
    plats = None
    xp = None
    yp = None
    
    # TODO: generalize this for any region in the world (not only Switzerland)
    if label[0] == "lon":
        plons = [6, 8, 10]
    elif label[0] == "x":
        xp = [2550000.0, 2750000.0]
    if label[1] == "lat":
        plats = [46, 47.5]
    elif label[1] == "y":
        yp = [1110000.0, 1250000.0]
    
    if label[0] == "lon" or label[1] == "lat":
        _add_swiss_latlon_labels(ax, plons=plons, plats=plats)
    
    elif label[0] == "x" or label[1] == "y":
        _add_swiss_xy_labels(ax, xp=xp, yp=yp)


def plot_2maps(
    arrays,
    titles,
    extents,
    projections=None,
    labels=None,
    cmap='YlGnBu',
    vmin=0,
    vmax=3,
    cbar_label='Precipitation (mm/h)',
):
    if projections is None:
        projections = [ccrs.PlateCarree(), ccrs.PlateCarree()]
    if labels is None:
        labels = [("lon", "lat"), ("lon", None)]

    # Create figure and subplots
    fig = plt.figure(figsize=(12, 5))
    axes = [None, None] # dummy axes to be replaced
    
    for i, ax in enumerate(axes):
        ax = fig.add_subplot(1, 2, i+1, projection=projections[i])
        img = ax.imshow(
            arrays[i],
            origin='lower',
            extent=extents[i],
            transform=projections[i],
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
        )
        ax.add_feature(cfeature.BORDERS)
        ax.set_title(titles[i])
        _write_label(ax, labels[i])
        axes[i] = ax
    
    fig.colorbar(img, shrink=0.25, ax=axes, location='bottom', label=cbar_label)
    
    return fig, axes


def plot_4maps(
    arrays, 
    titles, 
    extents, 
    projections=None,
    labels=None,
    cmap='YlGnBu',
    vmin=0,
    vmax=15,
    cbar_label='Precipitation (mm/h)',
):
    fig, axs = plt.subplots(2, 2, figsize=(10, 6), subplot_kw={'projection': ccrs.PlateCarree()})

    for i, ax in enumerate(axs.flat):
        img = ax.imshow(
            arrays[i],
            origin='lower', 
            extent=extents[i], 
            transform=projections[i],
            cmap=cmap,
            vmin=vmin, 
            vmax=vmax,
        )
        ax.add_feature(cfeature.BORDERS)
        ax.set_title(titles[i])
        _write_label(ax, labels[i])

    fig.colorbar(img, shrink=0.3, ax=axs, location='bottom', label=cbar_label)

    return fig, axs


def plot_maps(
    arrays,
    titles,
    extents,
    projections=None,
    axis_labels=None,
    cmap='YlGnBu',
    vmin=0,
    vmax=3,
    cbar_label='Precipitation (mm/h)',
):
    """
    General function to plot either 2 or 4 maps.

    ## Parameters:
    - arrays (tuple): tuple of data arrays. Should be either 2 or 4 elements long.
    - titles (tuple): tuple of titles for the maps. Should match the length of arrays.
    - extents (tuple): tuple of extents for the maps. Should match the length of arrays.
    - projections (tuple, optional): tuple of projections for the maps. Should match the length of arrays. Defaults to PlateCarree.
    - labels (tuple, optional): tuple of labels for the maps. Defaults to None.
    - cmap (str, optional): colormap to use for plotting. Default is 'YlGnBu'.
    - vmin (int, optional): minimum value for colormap. Default is 0.
    - vmax (int, optional): maximum value for colormap. Default is 3. For high precipitation events, use 15 or even 45 for storms.
    - cbar_label (str, optional): label for the colorbar. Default is 'Precipitation (mm/h)'.

    ## Returns:
    - fig (matplotlib figure object)
    - ax (matplotlib axis object or list of axis objects)
    """
    if len(arrays) == 2:
        projections = projections or [ccrs.PlateCarree(), ccrs.PlateCarree()]
        if axis_labels is None:
            axis_labels = (("lon", "lat"), ("lon", None))
        return plot_2maps(
            arrays, titles, extents, projections, axis_labels, cmap, vmin, vmax, cbar_label
        )
    elif len(arrays) == 4:
        projections = projections or [ccrs.PlateCarree()] * 4
        if axis_labels is None:
            axis_labels = ((None, "lat"), (None, None), ("lon", "lat"), ("lon", None))
        return plot_4maps(
            arrays, titles, extents, projections, axis_labels, cmap, vmin, vmax, cbar_label
        )
    else:
        raise ValueError("arrays must contain either 2 or 4 datasets.")


def plot_cdfs(arrays, labels, n_quantiles=2000, colors=None, cmap='Dark2'):
    """
    Plot the cumulative distribution functions (CDFs) of multiple arrays.

    ## Parameters:
    - arrays (tuple of arrays): tuple of arrays to plot CDFs for.
    - labels (tuple of str): tuple of labels for the arrays. Should match the length of arrays.
    - n_quantiles (int, optional): number of quantiles to divide the data into. Default is 2000.
    - colors (tuple of str, optional): tuple of colors for each plot. If provided, should match the length of arrays.
    - cmap (str, optional): colormap to use for plotting (default: 'Dark2')

    ## Returns:
    - fig (matplotlib figure object)
    - ax (matplotlib axis object)
    """
    if len(arrays) != len(labels):
        raise ValueError("The number of arrays and labels must be the same.")
    
    if colors is not None:
        if len(colors) != len(arrays):
            raise ValueError("The number of colors must match the number of data arrays if colors are provided.")
    else:
        cmap = plt.get_cmap(cmap)
        colors = [cmap(i) for i in range(len(arrays))]
    
    cmap = plt.get_cmap(cmap)
    global_max = max(np.nanmax(arr) for arr in arrays)
    global_min = min(np.nanmin(arr) for arr in arrays)
    
    wide = abs(global_max - global_min) / n_quantiles
    xbins = np.arange(global_min, global_max + wide, wide)

    cdfs = [get_cdf(arr, xbins) for arr in arrays]
    
    fig, ax = plt.subplots(figsize=(8, 5))
    for i, (cdf, label) in enumerate(zip(cdfs, labels)):
        ax.plot(xbins, cdf, label=label, color=colors[i])
    
    ax.set_xlabel('log(Precipitation (mm/h))')
    ax.set_xscale('log')
    ax.set_ylabel('CDF')
    ax.legend()
    
    return fig, ax
    
    
def plot_psds(arrays, labels, spatial_lengths, min_threshold=None, max_threshold=None, colors=None,cmap='Dark2'):
    """
    Plot multiple PSDs on the same figure.

    Parameters:
    - arrays (tuple of arrays): tuple of data arrays to compute PSDs for.
    - labels (tuple of str): tuple of labels for the data arrays. Should match the length of arrays.
    - spatial_lengths (tuple of tuples): tuple of (x_length, y_length) tuples for each data array. Should match the length of arrays.
    - colors (tuple of str, optional): tuple of colors for each plot. If provided, should match the length of arrays.
    - min_threshold (float, optional): minimum threshold value to mask out low PSD values
    - max_threshold (float, optional): maximum threshold value to mask out high PSD values
    - cmap (str, optional): colormap to use for plotting (default: 'Dark2')

    Returns:
    - fig (matplotlib figure object)
    - ax (matplotlib axis object)
    """
    if len(arrays) != len(labels) or len(arrays) != len(spatial_lengths):
        raise ValueError("The number of data arrays, labels, and spatial lengths must be the same.")
    
    if colors is not None:
        if len(colors) != len(arrays):
            raise ValueError("The number of colors must match the number of data arrays if colors are provided.")
    else:
        cmap = plt.get_cmap(cmap)
        colors = [cmap(i) for i in range(len(arrays))]
    
    fig, ax = plt.subplots(figsize=(8, 6))

    for i, (data, label, (x_length, y_length)) in enumerate(zip(arrays, labels, spatial_lengths)):
        wavelengths, psd = get_psd(data, x_length, y_length)
        
        if min_threshold is not None:
            mask = (psd >= min_threshold)
        else:
            mask = np.ones_like(psd, dtype=bool)
            
        if max_threshold is not None:
            mask &= (psd <= max_threshold)
        
        ax.loglog(wavelengths[mask], psd[mask], label=label, color=colors[i])

    ax.set_xlabel(r"Wavelength $(km$)")
    ax.set_ylabel("PSD")
    ax.legend(fontsize='large')

    return fig, ax