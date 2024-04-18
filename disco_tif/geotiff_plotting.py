import os
import rasterio
import earthpy as et
import earthpy.spatial as es
import earthpy.plot as ep
import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import datetime
import numpy as np
import pandas as pd
import sklearn.decomposition


def plot_singleband_raster(raster_data, cmap="terrain", title='Raster Data', ax=None, figsize = [15, 9]):
    '''description
Input Parameters:
    - raster_data:
    
    - cmap="terrain":
    
    - title='Raster Data':
    
    - ax=None:
    
    - figsize = [15, 9]
    '''
    # Plot the data
    if ax is None:
        figsize = [15, 9]
        fig,ax = plt.subplots(1,1,figsize=figsize)
    ep.plot_bands(raster_data,
                  cmap = cmap,
                  title=title,
                  ax=ax)
    plt.tight_layout()
    plt.show()
    return fig, ax


def plot_greyband_only(raster_data_dict,
                       nrows,
                       ncols,
                       plotsize=4):
    
    '''description
Input Parameters:
    - raster_data_dict:

    - nrows:

    - ncols:

    - plotsize=4
    '''
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=[ncols*plotsize, nrows*plotsize] )
    try:
        axs=axs.flatten()
    except:
        pass
    ind=-1
    for key, value in raster_data_dict.items():    
        ind+=1
        try:
            ax=axs[ind]
        except:
            ax=axs
        ep.plot_bands(value,
                      ax=ax,
                      cbar=False,
                      title=f"{key}")
    plt.tight_layout()
    plt.show()


def plot_color_raster_with_greyscale_overlay(raster_data, raster_data_dict, nrows, ncols, plotsize=4, cmap='terrain'):
    '''description
Input Parameters:
    - raster_data:
    
    - raster_data_dict:
    
    - nrows:
    
    - ncols:
    
    - plotsize=4:
    
    - cmap='terrain'
    '''
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=[ncols*plotsize, nrows*plotsize] )
    try:
        axs=axs.flatten()
    except:
        pass
    ind=-1
    for key, value in raster_data_dict.items():    
        ind+=1
        try:
            ax=axs[ind]
        except:
            ax=axs
        ep.plot_bands(raster_data,
                      ax=ax,
                      cmap = cmap,
                      title=f"{key}")
        ax.imshow(value, cmap="Greys", alpha=0.5)
    
    plt.tight_layout()
    plt.show()


