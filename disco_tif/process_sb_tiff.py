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

######################################

def nowTime():
    return datetime.datetime.now().strftime("%H:%M:%S")
def now():
    return datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
def snow():
    return datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
def today():
    return datetime.datetime.now().strftime("%Y-%m-%d")
def stoday():
    return datetime.datetime.now().strftime("%Y%m%d")

######################################

def hex_to_rgb(hexcolor):
    if '#' in hexcolor:
        hexcolor = hexcolor.split('#')[1]
    rgb = []
    for i in (0, 2, 4):
        decimal = int(hexcolor[i:i+2], 16)
        rgb.append(decimal)
    return rgb[0], rgb[1], rgb[2]
def rgb_to_hex(r, g, b):
    return '#{:02x}{:02x}{:02x}'.format(r, g, b)
           
######################################

# define custom color steps  - Order matters
EMerald_custom_colors_hexcolorcodes = ['#0000ff', # Blue
                                       '#01ffff', # Cyan
                                       '#3fca3f', # Green
                                       '#f19898', # Salmon
                                       '#deb201', # Tan
                                       '#896651', # Brown
                                       '#f1bfff', # Light_Purple
                                       '#fffafc', # Near_White
                                      ]

colormap_length = 256

######################################

def build_custom_colormap(breaks_by_percentages, custom_color_hex, new_cmap_name="Custom_Colormap"):
    ''' Function to take a sorted array of percentage-break-points (i.e. breaks_by_percentages) and applies it to the colorlist (i.e. custom_color_hex) colormap. The length of the breaks_by_percentages array should be the same length as custom_color_hex (length=8 for EMerald_custom_colors_hexcolorcodes) and range from 0 to 1

- Parameters:
    - breaks_by_percentages:List of breakpoints in decimal-percentages of data. i.e. [0.0, 0.3, 0.6, 0.9, 1.0]
    
    - custom_color_hex: List of color hex codes to generate the colormap from. i.e. ['#000000', '#aaaaaa', '#dddddd', '#eeeeee', '#ffffff']
    
    - new_cmap_name: String name to use for the generation of the new cmap.
        defualt: "Custom_Colormap"
    '''
    custom_color_rgb=[]
    for hexcode in custom_color_hex:
        temp = hex_to_rgb(hexcode) 
        custom_color_rgb.append([np.round(temp[0]/(colormap_length-1), 3), 
                                          np.round(temp[1]/(colormap_length-1), 3), 
                                          np.round(temp[2]/(colormap_length-1), 3)])
    custom_color_array = np.array(custom_color_rgb)

    if breaks_by_percentages[1]==0: # if second entry is 0 we know that there are no negative numbers and we should ignore blue
        sn=1
    else:
        sn=0
        
    CustomColormap_cdict = {'red':   [(breaks_by_percentages[ijk],  custom_color_array[ijk,0], custom_color_array[ijk,0]) for ijk in range(sn, len(breaks_by_percentages))],
                            'green': [(breaks_by_percentages[ijk],  custom_color_array[ijk,1], custom_color_array[ijk,1]) for ijk in range(sn, len(breaks_by_percentages))],
                            'blue':  [(breaks_by_percentages[ijk],  custom_color_array[ijk,2], custom_color_array[ijk,2]) for ijk in range(sn, len(breaks_by_percentages))],
                           }
    if custom_color_hex == EMerald_custom_colors_hexcolorcodes:
        return CustomColormap_cdict
    else:
        CustomColormap = LinearSegmentedColormap(new_cmap_name, CustomColormap_cdict, N=colormap_length)
        return CustomColormap


def build_EMerald_terrain_colormap(breaks_by_percentages):
    EMeraldCustomColormap_cdict = build_custom_colormap(breaks_by_percentages, EMerald_custom_colors_hexcolorcodes)
    EMeraldCustomColormap = LinearSegmentedColormap("EMerald_Custom_Colormap", EMeraldCustomColormap_cdict, N=colormap_length)
    return EMeraldCustomColormap

######################################

def make_percentile_array(data_min_max, data, no_data_value, cmap_method='pseudo_hist_norm', plot_histograms=False):
    '''Function to build a data driven percentile array based on the cmap method specified.
Parameters:
    - data_min_max: list containing the minimum and maximum values to display. values outside this range will be saturated to the end members.
    
    - data: 2D array of cell values of the raster.
    
    - no_data_value: Value that specifies the no_data_value
    
    - cmap_method: parameter to tell the program how to bin the data. options are 'pseudo_hist_norm', or 'pseudo_linear'
        default: 'pseudo_hist_norm'
    
    - plot_histograms: boolean parameter for plotting the out percentage break points
        default: False
    
    '''
    if data_min_max is not None:
        assert len(data_min_max)==2, 'len of data_min_max must be 2'
        assert data_min_max[0] < data_min_max[1], 'first value must be less than second value'
        assert data_min_max[1] > 0, 'This should really be a bigger number, but at least this will save dividing by a zero...'

    datatype = str(data[0,0].dtype)
    num_color = len(EMerald_custom_colors_hexcolorcodes)
    bz_num_color = 1 #number of colors for below zero
    az_num_color = num_color-bz_num_color-1 # number of intervals

    z_data = data.copy()
    z_data = z_data.flatten()
    bz_data = z_data[z_data<0]
    az_data = z_data[z_data>=0]
    norm_az_data = (az_data.copy() - np.min(az_data)) / (np.max(az_data) - np.min(az_data)) # shift to zero, then normalize by the range

    if cmap_method=='pseudo_linear':
        az_min=np.max([0, data_min_max[0]]) # above_zero_min: if data_min_max[0]<0, then 0; if data_min_max[0]>=0, then data_min_max[0].
        #print(f"az_min = {az_min}")
        az_data_breaks = np.round(np.linspace(az_min, data_min_max[1], az_num_color+1))
        #print(f"az_data_breaks = {az_data_breaks}")
        
    elif cmap_method=='pseudo_hist_norm':
        my_percentiles = np.linspace(0, 100, az_num_color+1)
        my_percentiles = my_percentiles[1:-1]
        #print(f"my_percentiles = {my_percentiles}")
        
        az_data_breaks = np.percentile(a=az_data, q=my_percentiles)
        if data_min_max[0]<0:
            az_data_breaks = np.insert(az_data_breaks, 0, 0) #prepend with: if data_min_max[0]<0, then 0; 
        else:    
            az_data_breaks = np.insert(az_data_breaks, 0, data_min_max[0]) #prepend with: if data_min_max[0]>=0, then data_min_max[0]
        az_data_breaks = np.append(az_data_breaks, data_min_max[1])
        #print(f"az_data_breaks = {az_data_breaks}")
    
    data_breaks = [data_min_max[0]]
    #print(f"data_breaks = {data_breaks}")
    data_breaks.extend(az_data_breaks)
    data_breaks = np.array(data_breaks)
    data_breaks = data_breaks.astype(datatype).tolist()
    #print(f"data_breaks = {data_breaks}")
    
    percentile_breaks = np.round((np.array(data_breaks) - data_min_max[0]) / (data_min_max[1] - data_min_max[0]), 4)
    percentile_breaks = percentile_breaks.tolist()
    #print(f"percentile_breaks = {percentile_breaks}")

    if plot_histograms:
        no_dum_data = data.copy()
        no_dum_data = no_dum_data.flatten()
        no_dum_data = no_dum_data.astype(float)
        if no_data_value is not None:
            no_dum_data[no_dum_data==no_data_value] = np.nan
        norm_no_dum_data = (no_dum_data.copy() - data_min_max[0]) / (data_min_max[1] - data_min_max[0]) # shift to zero, then normalize by the range
        
        fig, axs = plt.subplots(nrows=2, ncols=1, sharey=True)
        
        axs[0].hist(no_dum_data, bins=min(data_min_max[1]-data_min_max[0], 100))
        ylimits = axs[0].get_ylim()
        for db in data_breaks:
            axs[0].plot([db, db], [0, ylimits[1]])
        axs[0].set_title('Breaks by data values')
    
        axs[1].hist(norm_no_dum_data, bins=min(data_min_max[1]-data_min_max[0], 100))
        for pb in percentile_breaks:
            axs[1].plot([pb, pb], [0, ylimits[1]])
        axs[1].set_title('Breaks by percentage of data')
        
        plt.tight_layout(); plt.show()

    return percentile_breaks, data_breaks

######################################

def calc_data_min_max(data, no_data_value, clip_perc=None, min_max_method='data_absolute'):
    '''Function to calculate the min max values of the data
Parameters:
    - data: 2D array of raster cell data
    
    - no_data_value: value the defines the no-data-value
    
    - clip_perc: list of percentile (quartile) values to clip the data to. i.e. [1, 99] for clip to the fist and 99th percentiles of the data (essentiall, exclude erroneous high or low points). this is only relevant if the min_max_method is 'data_absolute'.
        Default: None
        
    - min_max_method: defines how to define the min_max method. Options are: 'data_absolute', or 'percentile'
        default: 'data_absolute'
    '''
    if min_max_method=='data_absolute':
        # using absolute min max from data
        if no_data_value is not None:
            data_min_max = [np.min(data[data!=no_data_value]), np.max(data[data!=no_data_value])]
        else:
            data_min_max = [np.min(data), np.max(data)]
        
    elif min_max_method=='percentile':
        # using percentiles of data (defined in the import statement)
        if no_data_value is not None:
            data_min_max = np.percentile(a=data[data!=no_data_value], q=clip_perc)
        else:
            data_min_max = np.percentile(a=data, q=clip_perc)
        data_min_max = np.round(data_min_max)
        data_min_max = data_min_max.astype(int)
        data_min_max = list(data_min_max)
    return data_min_max

######################################

def build_1_component_color_tables(cmap, data_breaks, data, no_data_value, new_multiband_lut_path):
    EMerald_colors_rgb = pd.DataFrame()
    for ii in range(0, len(cmap)):
        hexcolor = cmap[ii]
        EMerald_colors_rgb.loc[ii, ['r']] = hex_to_rgb(hexcolor)[0]
        EMerald_colors_rgb.loc[ii, ['g']] = hex_to_rgb(hexcolor)[1]
        EMerald_colors_rgb.loc[ii, ['b']] = hex_to_rgb(hexcolor)[2]
    
    if len(EMerald_colors_rgb) == len(data_breaks):
        EMerald_colors_rgb['data_val'] = np.array(data_breaks)
    else:
        print("there's an odd mismatch in length of 'EMerald_colors_rgb' and 'data_breaks'")

    outfilepaths=[]
    for rgb in ['r', 'g', 'b']:
        lut_str = f"{EMerald_colors_rgb.loc[0, 'data_val']}: {EMerald_colors_rgb.loc[0, rgb]}" 
        for row in range(1, len(EMerald_colors_rgb)):
                lut_str = f"{lut_str}, {EMerald_colors_rgb.loc[row, 'data_val']}: {int(np.round(EMerald_colors_rgb.loc[row, rgb]))}"
        tname=f"{new_multiband_lut_path}_{rgb}.lut"
        outfilepaths.append(tname)
        with open(tname, 'w') as lut_file_out:
            lut_file_out.write(lut_str)
    
    lut_str = f"{np.array(no_data_value, dtype=data[0,0].dtype).tolist()}: 0, {data_breaks[0]}: 255, {data_breaks[-1]}:255"
    tname=f"{new_multiband_lut_path}_a.lut"
    outfilepaths.append(tname)    
    with open(tname, 'w') as lut_file_out:
        lut_file_out.write(lut_str)

    return outfilepaths
    
######################################

def build_4_component_color_tables(single_band_tiff_path, cmap, data, no_data_value, percentile_breaks, data_breaks, outfile):
    pathparts = single_band_tiff_path.split(os.path.sep)
    destfolder = ''
    for part in pathparts[:-1]:
        destfolder=f"{destfolder}{part}{os.path.sep}"
    
    index_breaks = np.round([id * 255 for id in percentile_breaks]).astype(int).tolist()
    index_breaks

    ph_colormap_df = pd.DataFrame((cmap._lut * 255).astype('uint8'), columns=['red', 'green', 'blue', 'alpha']).iloc[:256,:]
    ph_colormap_df.loc[index_breaks,'data_breaks'] = data_breaks
    ph_colormap_df['data_breaks'] = ph_colormap_df['data_breaks'].interpolate(method='linear').astype(type(data_breaks[0]))
    
    nan_ph_colormap_df = ph_colormap_df.copy()
    nan_ph_colormap_df.loc[-1] = [0, 0, 0, 0, np.array(no_data_value, dtype=data[0,0].dtype).tolist()]
    nan_ph_colormap_df.index = nan_ph_colormap_df.index + 1  # shifting index
    nan_ph_colormap_df = nan_ph_colormap_df.sort_index()  # sorting by index
    
    #outfile = os.path.join(destfolder, outfile)
    
    qgisfile = f"{outfile}_qgis_color_table.txt"
    ph_colormap_df.to_csv(qgisfile, index=False, header=False, columns=['data_breaks', 'red', 'green', 'blue', 'alpha', 'data_breaks'])
    with open(qgisfile, 'r') as inlut:
        origstuff=inlut.read()
    with open(qgisfile, 'w') as outlut:
        outlut.seek(0)
        outlut.write(f"# EMerald Generated Color Map Export File for {single_band_tiff_path}\n")
        outlut.write("INTERPOLATION:INTERPOLATED\n")
        outlut.write(origstuff)

    rgba_lut_file = f"{outfile}_lut.lut"
    ph_colormap_df.to_csv(rgba_lut_file, index=False, header=False, columns=['data_breaks', 'red', 'green', 'blue', 'alpha'])
    nan_rgba_lut_file = f"{outfile}_NaN_lut.lut"
    nan_ph_colormap_df.to_csv(nan_rgba_lut_file, index=False, header=False, columns=['data_breaks', 'red', 'green', 'blue', 'alpha'])
    
    return [qgisfile, rgba_lut_file]
    
######################################

def make_rgba_tiff_from_single_Band(single_band_tiff_path, 
                                    data_min_max=None, 
                                    min_max_method='percentile', 
                                    clip_perc=[1, 99], 
                                    color_palette_name=None, 
                                    cmap_method='pseudo_hist_norm',
                                    output_tif=False,
                                    generate_lookup_tables=False,
                                    plot_rgba_raster=False,
                                   ):
    '''Function to take a single band geotiff file, apply a colormap to the data, and write a rgba geotiff to file

Input parameters:
 - single_band_tiff_path: 
     complete path to single-band-geotiff

 - data_min_max: 
     default = None
     Can take a list of lenth: 2, ex: [0, 500]
     if not spcified, this function will automatically apply min/max values based on the clip_perc values
 
 - min_max_method:
     default = 'percentile'; only relevant if data_min_max==None.
     Also accepts 'data_absolute'. 
     'percentile' uses the percentiles used in clip_perc. 
     'data_absolute' uses the minimum and maximum values of the data
 
 - clip_perc
     default = [1, 99]; only relevant if data_min_max==None.
     Percentile values to clip the data values to if no data_min_max is specified.
 
 - color_palette_name
     default = None
     Desired color pallet based on matplotlib colormaps. 
     https://matplotlib.org/stable/users/explain/colors/colormaps.html
     If None, the funtion will automatically create a new "EMerald_Custom_Colormap"
 
 - cmap_method:
     defualt = 'pseudo_hist_norm'
     Method to determine where the color breaks should be. Also accepts 'pseudo_linear'
     'pseudo_hist_norm' will produce a linear colormap for data values below 0 and a histogram normalized colormap for the positive values
     'pseudo_linear' will produce a linear colormap for data values below 0 and a separate linear colormap for the positive values.

 - output_tif:
     default = False
     If new geotiff are desired as outputs this must be set to True

 - generate_lookup_tables:
     defaule = False
     If set to True look up tables will be generated and written to file. This includes a look up table that can be applied to the single-band geotiff in QGIS and maybe other GIS software.

 - plot_rgba_raster
     default=False
     If set to True this will generate new matplotlib figures.
    '''

    if data_min_max is not None:
        assert len(data_min_max)==2, "len of data_min_max must be 2"
        assert data_min_max[0] < data_min_max[1], "first value must be less than second value"
        assert data_min_max[1] > 0, "This should really be a bigger number, but at least this will save dividing by a zero..."
    assert len(clip_perc)==2, "len of data_min_max must be 2"
    assert clip_perc[0] < clip_perc[1], "first value must be less than second value"
        
    # 1. Read the Single-Band GeoTIFF:
    # Open the single-band GeoTIFF
    with rasterio.open(single_band_tiff_path, 'r') as src:
        data = src.read(1)  # Read the first band
        origprofile = src.profile
        extent = src.bounds
        size = (src.width, src.height)
        epsg_code = src.crs.to_epsg() if src.crs else None
        no_data_value = src.nodata  # Get the no-data value from the GeoTIFF

    no_data_value = np.array(no_data_value, dtype=data[0,0].dtype).tolist()
    if data_min_max is None:
        data_min_max = calc_data_min_max(data, no_data_value, clip_perc, min_max_method=min_max_method)

    # Clip data values to the specified range
    clipped_data = np.clip(data, data_min_max[0], data_min_max[1])
    
    # Normalize the clipped data to the [0, 1] range
    normalized_data = (clipped_data - data_min_max[0]) / (data_min_max[1] - data_min_max[0]) # shift to zero, divide by the range

    # clip the data to the min-max values specified
    clipped_data_with_dum = clipped_data.copy()
    if no_data_value is not None:
        clipped_data_with_dum[data==no_data_value]=no_data_value # since int is a valid datatype we can reuse the no-data-value (iff it's outside our min-max range)
    
    # make percentile ranges
    percentile_breaks, data_breaks = make_percentile_array(data_min_max, 
                                                           clipped_data_with_dum, 
                                                           no_data_value,
                                                           cmap_method=cmap_method)
    #print(f"percentile_breaks = {percentile_breaks}")
    #print(f"data_breaks = {data_breaks}")

    # 2. Generate a custom colormap (EMeraldCustomColormap):
    if color_palette_name is None:
        color_palette_name = "EMeraldCustomTerrain"
        EMeraldCustomColormap = build_EMerald_terrain_colormap(percentile_breaks)
    else:
        EMeraldCustomColormap = mpl.colormaps.get_cmap(color_palette_name)
    EMeraldCustomColormap(0) #need this line for it to make a lut?!?
    #print(EMeraldCustomColormap._lut)
    
    # define output name
    sbpath, ext = os.path.splitext(single_band_tiff_path)
    suffix=f"{color_palette_name}_{data_min_max[0]}_to_{data_min_max[1]}_{cmap_method}"
    new_multiband_lut_path = f"{sbpath}_{suffix}"
    new_multiband_tiff_path = f"{sbpath}_rgba_{suffix}"
    
    if plot_rgba_raster:
        # clip the data to the min-max values specified
        clipped_data_with_nan = clipped_data.copy().astype(float)
        if no_data_value is not None:
            clipped_data_with_nan[data==no_data_value]=np.nan
    
        figsize = [15, 9]
        fig,ax=plt.subplots(1,1,figsize=figsize)
        ep.plot_bands(clipped_data_with_nan,
                      cmap = EMeraldCustomColormap,
                      title=f"{sbpath.split(os.path.sep)[-1]}\n{suffix.replace('_', ' ')}",
                      ax=ax,
                     )
        plt.tight_layout()
        plt.show()    

    if generate_lookup_tables:
        outfilepaths = build_1_component_color_tables(cmap=EMerald_custom_colors_hexcolorcodes,
                                                      data_breaks=data_breaks,
                                                      data=data,
                                                      no_data_value=no_data_value,
                                                      new_multiband_lut_path=new_multiband_lut_path )
        for fp in outfilepaths:
            print(f"Wrote 1component LUT files to: '{fp}'")
    
        outfilepaths = build_4_component_color_tables(single_band_tiff_path=single_band_tiff_path,
                                                      cmap=EMeraldCustomColormap,
                                                      data=data,
                                                      no_data_value=no_data_value,
                                                      percentile_breaks=percentile_breaks,
                                                      data_breaks=data_breaks,
                                                      outfile=new_multiband_tiff_path)
        for fp in outfilepaths:
            print(f"wrote 4component Lut files to: '{fp}''")
    
    if output_tif:
        # apply EMeraldCustomColormap to data
        rgba_data = EMeraldCustomColormap(normalized_data) * (colormap_length-1)  # Scale to 0-255 range
        
        # 3. Convert to RGB Channels:
        rgb_data = rgba_data[:, :, :3]  # Extract RGB channels
        
        # 4. Generate an Alpha Channel:
        alpha_channel = (data != no_data_value).astype('uint8') * (colormap_length-1)
        rgba_data[:, :, 3] = alpha_channel
        
        # 5. Write the New RGBA GeoTIFF:
        newprofile = origprofile.copy()
        newprofile.update(count=4, dtype='uint8', nodata=None)  # RGBA format
        
        with rasterio.open(f"{new_multiband_tiff_path}.tif", 'w', **newprofile) as dst:
            dst.write(rgba_data[:, :, 0].astype('uint8'), 1) #red
            dst.write(rgba_data[:, :, 1].astype('uint8'), 2) #green
            dst.write(rgba_data[:, :, 2].astype('uint8'), 3) #blue
            dst.write(rgba_data[:, :, 3].astype('uint8'), 4) #alpha
    
        print(f"New RGBA geotiff '{new_multiband_tiff_path}' generated successfully!")
    return EMeraldCustomColormap, data_breaks, percentile_breaks,  new_multiband_tiff_path

######################################
