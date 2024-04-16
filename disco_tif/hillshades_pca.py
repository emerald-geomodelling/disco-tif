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

#import process_sb_tiff
import disco_tif.geotiff_plotting

#################################################

def build_hs_az_al(start_al, al_inc, num_al_angles, start_az, num_az_angles):
    azimuths = np.linspace(start_az, start_az+360, num_az_angles+1) # degrees 0=360, number of divisions + 1 because start=end
    filt = azimuths>=360
    while filt.sum()>0:
        azimuths[filt] = azimuths[filt]-360
        filt = azimuths>=360
    azimuths = azimuths[:-1].astype(int).tolist()
    azimuths = sorted(azimuths)
    print(f"azimuths = {azimuths}")
    
    altitudes = np.linspace(start_al, (al_inc*num_al_angles)+start_al, num_al_angles+1)
    altitudes = altitudes[:-1].astype(int).tolist()
    print(f"altitudes = {altitudes}")
    return azimuths, altitudes

def write_raster_dict_data_to_geotiff(single_band_tiff_path, origprofile, raster_data_dict, len_hs=None):
    for key, value in raster_data_dict.items():
        if 'component' in key:
            assert len_hs is not None, "'len_hs' cannot be none if passing in a pca_dictionary_object"
            new_tiff_path = f"{single_band_tiff_path.split('.tif')[0]}_hillshade_pca{len_hs}-{key}.tif"
        else:
            new_tiff_path = f"{single_band_tiff_path.split('.tif')[0]}_hillshade_{key}.tif"
        
        newprofile = origprofile.copy()
        newprofile.update(dtype=str((value[0,0]).dtype), nodata=np.nan)
           
        with rasterio.open(new_tiff_path, 'w', **newprofile) as dst:
            dst.write(arr=value, indexes=1, masked=True)
    
        print(f"New RGBA geotiff '{new_tiff_path}' generated successfully!")
        

def MakeHillShadePCA(hillshades, plot_figures=False, raster_data=None, cmap='terrain', n_components=3):
    if plot_figures:
        assert raster_data is not None, "raster data must be supplied if plot_figures is true"
        
    NoNanIndicies = np.argwhere(~np.isnan(hillshades[list(hillshades.keys())[0]].flatten())).flatten()

    n_samples = len(NoNanIndicies)
    n_features = len(hillshades.keys())
    assert n_features >= 4, "There must be at least 4 hillshades to produce a PCA"
    
    flat_hillshades = np.ones([n_samples, n_features]) * np.nan
    
    feature_ind=-1
    for key, value in hillshades.items():
        feature_ind+=1
        value = value.flatten()
        value = value[NoNanIndicies]
        flat_hillshades[:,feature_ind]=value
    
    assert np.isnan(flat_hillshades).sum()==0, "There can be no NaN's when performing a pca"
    
    pca = sklearn.decomposition.PCA(n_components = n_components)
    
    pcaout = pca.fit(flat_hillshades).transform(flat_hillshades)
    pcaout
    
    pcaout.shape
    
    data = hillshades[list(hillshades.keys())[0]]
    
    nrow = data.shape[0]
    ncol = data.shape[1]
    nlay = len(pcaout[0])
    
    dumarray = np.ones([nrow * ncol])*np.nan
    
    pcaComponents={}
    for ilay in range(0,len(pcaout[0])):
        tdat = dumarray.copy()
        nowdat = pcaout[:, ilay]
        tdat[NoNanIndicies] = nowdat
        pcaComponents[f'component_{ilay+1}'] = tdat.reshape([nrow,ncol], order='C')
    
    if plot_figures:
        # plot the pca outputs
        disco_tif.geotiff_plotting.plot_greyband_only(raster_data_dict=pcaComponents,
                                                               nrows=1,
                                                               ncols=n_components)
        disco_tif.geotiff_plotting.plot_color_raster_with_greyscale_overlay(raster_data=raster_data,
                                                                                     cmap=cmap,
                                                                                     raster_data_dict=pcaComponents,
                                                                                     nrows=1,
                                                                                     ncols=n_components)
    
    # pcaComponents['component_1_0_1'] = (           pcaComponents['component_1'] - np.nanmin(pcaComponents['component_1'])) / \
    #                                    (np.nanmax(pcaComponents['component_1']) - np.nanmin(pcaComponents['component_1']))
    
    # pcaComponents['component_2_0_1'] = (           pcaComponents['component_2'] - np.nanmin(pcaComponents['component_2'])) / \
    #                                    (np.nanmax(pcaComponents['component_2']) - np.nanmin(pcaComponents['component_2']))
    
    # pcaComponents['component_3_0_1'] = (           pcaComponents['component_3'] - np.nanmin(pcaComponents['component_3'])) / \
    #                                    (np.nanmax(pcaComponents['component_3']) - np.nanmin(pcaComponents['component_3']))
                                                                                                                   
    return pcaComponents
    

def build_hillshade(single_band_tiff_path, data_min_max,  hs_azimuths, hs_altitudes, cmap='terrain', process_pca=False, plot_figures=False):
    # read geotiff and minimally process for the colormap function
    with rasterio.open(single_band_tiff_path, 'r') as src:
        data = src.read(1)  # Read the first band
        no_data_value = src.nodata  # Get the no-data value from the GeoTIFF
        epsg_code = src.crs.to_epsg() if src.crs else None
        origprofile = src.profile
        width = src.width 
        height = src.height
        srcmask = src.read_masks(1)
    
    # Clip data values to the specified range
    clipped_data = np.clip(data, data_min_max[0], data_min_max[1])
    
    nan_data = data.copy().astype(float)
    nan_clipped_data = clipped_data.copy().astype(float)
    
    if no_data_value is not None:
        nan_data[data==no_data_value]=np.nan
        nan_clipped_data[data==no_data_value]=np.nan

    #
    if plot_figures:
        # Plot the data
        disco_tif.geotiff_plotting.plot_singleband_raster(raster_data=nan_clipped_data,
                                                cmap=cmap,
                                                title=f"{single_band_tiff_path} Without Hillshade",
                                                ax=None)
       
    # calculate the hillshade for  all combination fo azimuths and altitudes
    num_hillshades = len(hs_azimuths) * len(hs_altitudes)
    # it it expensive to generate all of these hillshades and do the pca on them => we limit the number to something reasonable # 8 azimuths at 3 angles will produce 24 hillshades. ### what is reasonable here?
    max_num_hs = 24
    assert num_hillshades<=max_num_hs, f"Only {max_num_hs} azimuth-altidude combinations can be used to calculate a Hillshade PCA but {num_hillshades} were provided"

    hillshades = {}
    for az_ind, my_azimuth in enumerate(hs_azimuths):
        for al_ind, my_altitude in enumerate(hs_altitudes):
            # Create and plot the hillshade with earthpy
            txt_my_azimuth = f"00{my_azimuth}"[-3:]
            txt_my_altitude = f"0{my_altitude}"[-2:]
            hskey = f'az{txt_my_azimuth}_al{txt_my_altitude}'
            hillshades[hskey] = es.hillshade(nan_data, azimuth=my_azimuth, altitude=my_altitude)

    if plot_figures:
        # plot the hillshades 
        disco_tif.geotiff_plotting.plot_greyband_only(raster_data_dict=hillshades,
                                                               nrows=max([len(hs_azimuths), len(hs_altitudes)]),
                                                               ncols=min([len(hs_azimuths), len(hs_altitudes)]))
        # plot geotiff overlain with the hillshades
        disco_tif.geotiff_plotting.plot_color_raster_with_greyscale_overlay(raster_data=nan_clipped_data,
                                                                                     raster_data_dict=hillshades,
                                                                                     nrows=max([len(hs_azimuths), len(hs_altitudes)]),
                                                                                     ncols=min([len(hs_azimuths), len(hs_altitudes)]),
                                                                                     cmap=cmap)

    write_raster_dict_data_to_geotiff(single_band_tiff_path, origprofile, hillshades)

    if not process_pca:
        return hillshades
    elif process_pca:
        assert num_hillshades>=4, f'Need at least 4 azimuth-altitude combinations, only {num_hillshades} were provided'
        pcaComponents = MakeHillShadePCA(hillshades, plot_figures, raster_data=nan_clipped_data, cmap=cmap)
        write_raster_dict_data_to_geotiff(single_band_tiff_path, origprofile, pcaComponents, len_hs=len(hillshades.keys()))
        return hillshades, pcaComponents
