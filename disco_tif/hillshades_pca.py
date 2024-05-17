# import os
import rasterio
# import earthpy as et
import earthpy.spatial as es
# import earthpy.plot as ep
# import matplotlib as mpl
# from matplotlib import pyplot as plt
# from matplotlib.colors import LinearSegmentedColormap
# import datetime
import numpy as np
# import pandas as pd
import sklearn.decomposition

# import process_sb_tiff
import disco_tif.geotiff_plotting

#################################################

def build_hs_az_al( start_az=45, num_az_angles=4, start_al=30, al_inc=30, num_al_angles=2,):
    """Generate valid azimuths and altitude angles for use in generating hillshade rasters.
    
Parameters
----------
    start_az : int or float (default = 45)
        Starting azimuth angle. Valid angles are between 0 and 360

    num_az_angles : int (default = 4)
        Number of azimuths to generate. Must be greater than 1.

    start_al : int or float (default = 30°)
        Starting altitude angle. Valid angles are between 0 and 90

    al_inc : int or float (default = 30°)
        Altitude increment. First altitude angle will be 'start_al'.
        Second altitude angle will be start_al + al_inc, etc.

    num_al_angles : int (default = 2)
        Number of altitude angles to generate. Must be greater than 1.
        start_al + (al_inc * (num_al_angles - 1)) <= 90

Returns
-------
    azimuths : list
        List of azimuth angles

    altitudes : list
        List of altitude angles
    """
    assert start_az >= 0, 'starting azimuth must be greater than or equal to 0°'
    assert start_az <= 360, 'starting azimuth must be less than or equal to 360°'
    assert num_az_angles >= 1, 'Number of azimuths must be greater than or equal to 1'
    assert start_al >= 0, 'starting altitude angle must be greater than or equal to 0°'
    assert start_al <= 90, 'starting altitude angle must be less then or equal to 90°'
    assert num_al_angles >= 1, 'Number of altitude angles must be greater than or equal to 1'
    assert start_al + (al_inc * (num_al_angles-1)) <= 90, 'The combination of altitude angle parameters generates angles larger than 90°'

    azimuths = np.linspace(start_az, start_az+360, num_az_angles+1)
    filt = azimuths >= 360
    while filt.sum() > 0:
        azimuths[filt] = azimuths[filt]-360
        filt = azimuths >= 360
    azimuths = azimuths[:-1].astype(int).tolist()
    azimuths = sorted(azimuths)
    print(f"azimuths = {azimuths}")
    
    altitudes = np.linspace(start_al, (al_inc*(num_al_angles-1))+start_al, num_al_angles).astype(int).tolist()
    print(f"altitudes = {altitudes}")
    return azimuths, altitudes

def write_raster_dict_data_to_geotiff(single_band_tiff_path, orig_profile, raster_data_dict, num_hs=None):
    """Take a dictionary of hillshade or PCA component rasters and writes them disk.

Parameters
----------
    single_band_tiff_path : str
        Path to the original, single-band, geotiff

    orig_profile : dict
        The original profile from the original single band geotiff

    raster_data_dict : dict
        Dictionary containing hillshade rasters where the key is the name of the hillshade and the value is the hillshade

    num_hs : int (default = None)
        Only valid for a raster_data_dict that contains PCA rasters. The number of hillshades that went into making the PCA components

Returns
-------
    new_geotiff_paths : dict
        Dictionary of paths to the new geotiffs
    """
    new_geotiff_paths = {}
    for key, value in raster_data_dict.items():
        if 'component' in key:
            assert num_hs is not None, "'num_hs' cannot be none if passing in a pca_dictionary_object"
            new_tiff_path = f"{single_band_tiff_path.split('.tif')[0]}_hillshade_pca{num_hs}-{key}.tif"
        else:
            new_tiff_path = f"{single_band_tiff_path.split('.tif')[0]}_hillshade_{key}.tif"

        value = ((value.copy() - np.nanmin(value)) / (np.nanmax(value) - np.nanmin(value)))  # scale 0 to 1

        new_profile = orig_profile.copy()

        if np.sum(np.isnan(value)) == 0:
            value = (value * 255).round().astype('uint8')
            new_profile.update(dtype='uint8')
        else:
            value = (value * 254) + 1
            value[np.isnan(value)] = 0
            value = value.round().astype('uint8')
            new_profile.update(dtype='uint8', nodata=0)

        with rasterio.open(new_tiff_path, 'w', **new_profile) as dst:
            dst.write(arr=value, indexes=1, masked=True)
    
        new_geotiff_paths[key] = new_tiff_path
        print(f"New single-channel geotiff generated successfully: '{new_tiff_path}'")
    return new_geotiff_paths
        

def MakeHillShadePCA(hillshades, plot_figures=False, raster_data=None, cmap='terrain', n_components=3):
    """ Function to find the principal components of a bunch of hillshades.
This will help highlight and illuminate features in the dtm.

Parameters
----------
    hillshades : dict
        Dictionary of hillshades, where the key contains azimuth and altitude pairs, and the value is the hillshade raster

    plot_figures : bool (default = False)
        Switch to plot the newly generated PCA components to screen

    raster_data : 2D numpy array (default = None)
        Used only if plot_figures = True
        Used for plotting the new PCA components as an overlay on the original data.

    cmap : mpl-like colormap (default = 'terrain')
        Matplotlib-like color map. Either a colormap can be passed or a named mpl colormap can be passed.

    n_components : int (default = 3)
        The number of PCA components to generate. This number must be less than the number of hillshades

Returns
-------
    pcaComponents : dict
        Dictionary of PCA components, where the keys are the names of the rasters and the values are the components of the rasters.
    """
    assert n_components < len(hillshades), 'The number of PCA components must be less than the number of hillshades being evaluated.'
    if plot_figures:
        assert raster_data is not None, "raster data must be supplied if plot_figures is true"
        
    no_nan_indicies = np.argwhere(~np.isnan(hillshades[list(hillshades.keys())[0]].flatten())).flatten()

    n_samples = len(no_nan_indicies)
    n_features = len(hillshades.keys())
    assert n_features >= 4, "There must be at least 4 hillshades to produce a PCA"
    
    flat_hillshades = np.ones([n_samples, n_features]) * np.nan
    
    feature_ind = -1
    for key, value in hillshades.items():
        feature_ind += 1
        value = value.flatten()
        value = value[no_nan_indicies]
        flat_hillshades[:, feature_ind] = value
    
    assert np.isnan(flat_hillshades).sum() == 0, "There can be no NaN's when performing a pca"
    
    pca = sklearn.decomposition.PCA(n_components=n_components)
    
    pcaout = pca.fit(flat_hillshades).transform(flat_hillshades)

    data = hillshades[list(hillshades.keys())[0]]  # grab the first entry to get the shape of the raster
    
    nrow = data.shape[0]
    ncol = data.shape[1]

    dumarray = np.ones([nrow * ncol])*np.nan
    
    pcaComponents = {}
    for ilay in range(0, len(pcaout[0])):
        tdat = dumarray.copy()
        nowdat = pcaout[:, ilay]
        tdat[no_nan_indicies] = nowdat
        pcaComponents[f'component_{ilay+1}'] = tdat.reshape([nrow, ncol], order='C')
    
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
                                                                                                                   
    return pcaComponents
    

def build_hillshade(single_band_tiff_path, data_min_max, hs_azimuths, hs_altitudes, cmap='terrain', process_pca=False, plot_figures=False, **kwargs):
    """Function to generate hillshades and, if asked for, principal components of the hillshades.

Parameters
----------
    single_band_tiff_path : str
        Path to the original single-band-tiff

    data_min_max : list or tuple
        Must be of length 2
        Minimum and maximum values to clip the raster values to.

    hs_azimuths : list
        List of azimuths to generate hillshades with

    hs_altitudes : list
        List of altitude angles to generate hillshades with

    cmap : mpl-like colormap (default = 'terrain')
        Matplotlib-like color map. Either a colormap can be passed or a named mpl colormap can be passed.

    process_pca : bool (default = False)
        Switch to generate PCA components

    plot_figures : bool (default = False)
        Switch to plot figures to screen

Returns
-------
    hillshades : dict

    hillshade_file_paths : dict

    """
    assert len(data_min_max) == 2, 'len(data_min_max) must be 2'

    # read geotiff and minimally process for the colormap function
    with rasterio.open(single_band_tiff_path, 'r') as src:
        data = src.read(1)  # Read the first band
        no_data_value = src.nodata  # Get the no-data value from the GeoTIFF
        # epsg_code = src.crs.to_epsg() if src.crs else None
        orig_profile = src.profile
        # width = src.width
        # height = src.height
        # srcmask = src.read_masks(1)

    nan_data = data.copy().astype(float)
    if no_data_value is not None:
        nan_data[data == no_data_value] = np.nan

    num_hillshades = len(hs_azimuths) * len(hs_altitudes)

    max_num_hs = kwargs.get('max_num_hs', 24)
    assert num_hillshades < max_num_hs, f"Only {max_num_hs} azimuth-altitude combinations can be used to calculate a Hillshade PCA but {num_hillshades} were provided"

    hillshades = {}
    for az_ind, my_azimuth in enumerate(hs_azimuths):
        for al_ind, my_altitude in enumerate(hs_altitudes):
            txt_my_azimuth = f"00{my_azimuth}"[-3:]
            txt_my_altitude = f"0{my_altitude}"[-2:]
            hskey = f'az{txt_my_azimuth}_al{txt_my_altitude}'
            hillshades[hskey] = es.hillshade(nan_data, azimuth=my_azimuth, altitude=my_altitude)

    nan_clipped_data = None  # start with None and modify if plotting results to screen
    if plot_figures:
        nan_clipped_data = np.clip(data.copy(), data_min_max[0], data_min_max[1]).astype(float)
        if no_data_value is not None:
            nan_clipped_data[data == no_data_value] = np.nan

        disco_tif.geotiff_plotting.plot_singleband_raster(raster_data=nan_clipped_data,
                                                          cmap=cmap,
                                                          title=f"{single_band_tiff_path} Without Hillshade",
                                                          ax=None)
        disco_tif.geotiff_plotting.plot_greyband_only(raster_data_dict=hillshades,
                                                      nrows=max([len(hs_azimuths), len(hs_altitudes)]),
                                                      ncols=min([len(hs_azimuths), len(hs_altitudes)]))
        # plot geotiff overlain with the hillshades
        disco_tif.geotiff_plotting.plot_color_raster_with_greyscale_overlay(raster_data=nan_clipped_data,
                                                                            raster_data_dict=hillshades,
                                                                            nrows=max([len(hs_azimuths), len(hs_altitudes)]),
                                                                            ncols=min([len(hs_azimuths), len(hs_altitudes)]),
                                                                            cmap=cmap)

    hillshade_file_paths = write_raster_dict_data_to_geotiff(single_band_tiff_path, orig_profile, hillshades)

    if process_pca:
        assert num_hillshades >= 4, f'Need at least 4 azimuth-altitude combinations, only {num_hillshades} were provided'
        pcaComponents = MakeHillShadePCA(hillshades, plot_figures, raster_data=nan_clipped_data, cmap=cmap)
        pca_file_paths = write_raster_dict_data_to_geotiff(single_band_tiff_path, orig_profile, pcaComponents, num_hs=len(hillshades.keys()))
        hillshades.update(pcaComponents)
        hillshade_file_paths.update(pca_file_paths)

    return hillshades, hillshade_file_paths
