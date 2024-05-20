import earthpy.plot as ep
from matplotlib import pyplot as plt


def plot_singleband_raster(raster_data, cmap="terrain", title='Raster Data', ax=None, figsize=(15, 9)):
    """plotting routine for colorizing a single band raster.

Parameters
----------
    raster_data : 2D numpy array
    
    cmap : named mpl colormap (default = "terrain")
        Colormap to apply to the single band data

    title : str (default = 'Raster Data')
        Title to apply to the figure
    
    ax : mpl figure axes (default = None)
        If None, a new figure will be created

    figsize : tuple (default = (15, 9))
        If ax is None the size of the new figure

Returns
-------
    fig : figure handle
        Only if ax is None

    ax : axes handle
        Only if ax is None
    """
    # Plot the data
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)
    ep.plot_bands(raster_data,
                  cmap=cmap,
                  title=title,
                  ax=ax)
    plt.tight_layout()
    plt.show()

    if ax is None:
        return fig, ax


def plot_greyband_only(raster_data_dict, nrows, ncols, plotsize=4):
    """Plotting function to plot only the greyband overlays.

Parameters
----------
    raster_data_dict : dict
        Dictionary of raster data to plot

    nrows : int
        number of rows in the figure

    ncols : int
        number of columns in the figure

    plotsize : int (default = 4)
        Parameter defining how big each subplot should be - assumes square subplots
    """
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=[ncols*plotsize, nrows*plotsize])
    try:
        axs = axs.flatten()
    except:
        pass
    ind = -1
    for key, value in raster_data_dict.items():    
        ind += 1
        try:
            ax = axs[ind]
        except:
            ax = axs
        ep.plot_bands(value,
                      ax=ax,
                      cbar=False,
                      title=f"{key}")
    plt.tight_layout()
    plt.show()


def plot_color_raster_with_greyscale_overlay(raster_data, raster_data_dict, nrows, ncols, plotsize=4, cmap='terrain'):
    """Plotting function to plot a semi-transparent greyband overlay on top of a colorized raster.

Parameters
----------
    raster_data : 2D numpy array
        Original raster data

    raster_data_dict : dict
        Dictionary of semi-transparent raster overlays - like a hillshade
    
    nrows : int
        Number of rows in the figure
    
    ncols : int
        Number of columns in the figure

    plotsize : int (default=4)
        Parameter defining how big each subplot should be - assumes square subplots

    cmap : named mpl colormap (default='terrain')
    """
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=[ncols*plotsize, nrows*plotsize])
    try:
        axs = axs.flatten()
    except:
        pass
    ind = -1
    for key, value in raster_data_dict.items():    
        ind += 1
        try:
            ax = axs[ind]
        except:
            ax = axs
        ep.plot_bands(raster_data,
                      ax=ax,
                      cmap=cmap,
                      title=f"{key}")
        ax.imshow(value, cmap="Greys", alpha=0.5)
    
    plt.tight_layout()
    plt.show()
