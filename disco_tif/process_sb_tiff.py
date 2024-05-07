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
import copy

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

# define the number of the colors for the colormap
colormap_length = 256

######################################

def build_custom_colormap(breaks_by_percentages, custom_color_hex, new_cmap_name="Custom_Colormap"):
    ''' Function to take a sorted array of percentage-break-points (i.e. breaks_by_percentages) and applies it to the colorlist (i.e. custom_color_hex) colormap. The length of the breaks_by_percentages array should be the same length as custom_color_hex (length=8 for EMerald_custom_colors_hexcolorcodes) and range from 0 to 1

Input Parameters:
    - breaks_by_percentages: List of breakpoints in decimal-percentages of data. [0.0, 0.3, 0.6, 0.9, 1.0]
        - note - if starting and ending points are not at 0 and 1, respectively, the program will prepend and append the list with 0 and 1.
            The program will also copy the first and last colors in custom_color_hex to the new low and high percentages appended
    
    - custom_color_hex: List of color hex codes to generate the colormap from. i.e. ['#000000', '#aaaaaa', '#dddddd', '#eeeeee', '#ffffff']
    
    - new_cmap_name: String name to use for the generation of the new cmap.
        defualt: "Custom_Colormap"
    '''
    breaks_by_percentages = copy.deepcopy(breaks_by_percentages)
    custom_color_hex = copy.deepcopy(custom_color_hex)

    if len(breaks_by_percentages) != len(custom_color_hex): #it's ok for these to be differnt, but only by 1. we just omit the dark blue color if using EMerald_custom_colormap
        custom_color_hex = custom_color_hex[1:]
    else:
        pass
    
    assert len(breaks_by_percentages) == len(custom_color_hex), "The length of the breaks_by_percentages list but be the same as the length of custom_color_hex"

    if breaks_by_percentages[0] != 0:
        breaks_by_percentages.insert(0, 0)
        custom_color_hex.insert(0, custom_color_hex[0])
    if breaks_by_percentages[-1] != 1:
        breaks_by_percentages.append(1)
        custom_color_hex.append(custom_color_hex[-1])
    
    custom_color_rgb=[]
    for hexcode in custom_color_hex:
        temp = hex_to_rgb(hexcode) 
        custom_color_rgb.append([np.round(temp[0]/(colormap_length-1), 3), 
                                          np.round(temp[1]/(colormap_length-1), 3), 
                                          np.round(temp[2]/(colormap_length-1), 3)])
    custom_color_array = np.array(custom_color_rgb)
    custom_color_array

    if breaks_by_percentages[1]==0: # if second entry is 0 we know that there are no negative numbers and we should ignore blue
        sn=1
    else:
        sn=0

    #print(f"breaks_by_percentages = {breaks_by_percentages}")
    CustomColormap_cdict = {'red':   [(breaks_by_percentages[ijk],  custom_color_array[ijk,0], custom_color_array[ijk,0]) for ijk in range(sn, len(breaks_by_percentages))],
                            'green': [(breaks_by_percentages[ijk],  custom_color_array[ijk,1], custom_color_array[ijk,1]) for ijk in range(sn, len(breaks_by_percentages))],
                            'blue':  [(breaks_by_percentages[ijk],  custom_color_array[ijk,2], custom_color_array[ijk,2]) for ijk in range(sn, len(breaks_by_percentages))],
                           }
    CustomColormap = LinearSegmentedColormap(new_cmap_name, CustomColormap_cdict, N=colormap_length)
    CustomColormap    
    return CustomColormap


def build_EMerald_terrain_colormap(breaks_by_percentages):
    EMeraldCustomColormap = build_custom_colormap(breaks_by_percentages=breaks_by_percentages, 
                                                  custom_color_hex=copy.deepcopy(EMerald_custom_colors_hexcolorcodes),
                                                  new_cmap_name="EMerald_Custom_Colormap")
    return EMeraldCustomColormap

######################################

def make_percentile_array(data_min_max,
                          data,
                          no_data_value,
                          cmap_method='pseudo_hist_norm',
                          color_list=copy.deepcopy(EMerald_custom_colors_hexcolorcodes),
                          plot_histograms=False):
    '''Function to build a data driven percentile array based on the cmap method specified.
Input Parameters:
    - data_min_max: list containing the minimum and maximum values to display. values outside this range will be saturated to the end members.
        ex: [min, max]
    
    - data: 2D array of cell values of the raster.
        ex: ([[x1y1, ..., xny1], [x1y2, ..., xny2], ..., [x1yn, ..., xnyn]])
        
    - no_data_value: Value that specifies the no_data_value
    
    - cmap_method: parameter to tell the program how to bin the data. Options are 'pseudo_hist_norm', or 'pseudo_linear'
        default: 'pseudo_hist_norm'

    - color_list: list of colors, global variable
        Default: EMerald_custom_colors_hexcolorcodes
    
    - plot_histograms: boolean parameter for plotting the percentage break points on top of a histogram of the data
        default: False
    
    '''
    if data_min_max is not None:
        assert len(data_min_max)==2, 'len of data_min_max must be 2'
        assert data_min_max[0] < data_min_max[1], 'first value must be less than second value'
        assert data_min_max[1] > 0, 'This should really be a bigger number, but at least this will save dividing by a zero...'
    
    datatype = str(data[0,0].dtype)
    
    if data_min_max[0]<0:
        bz_num_color = 1 #number of colors for below zero
    else:
        bz_num_color = 0 #number of colors for below zero
        color_list = color_list[1:]

    num_color = len(color_list)
    az_num_color = num_color-bz_num_color # number of intervals

    clip_data = data.copy()
    clip_data = clip_data.flatten()
    #clip_data = clip_data.astype(float)
    if no_data_value is not None:
        clip_data = clip_data[clip_data!=no_data_value]
    clip_data = np.clip(clip_data, data_min_max[0], data_min_max[1])
    az_clip_data = clip_data.copy()
    az_clip_data = az_clip_data[az_clip_data>=0]
    

    if cmap_method=='pseudo_linear':
        az_min=np.max([0, data_min_max[0]]) # above_zero_min: if data_min_max[0]<0, then 0; if data_min_max[0]>=0, then data_min_max[0].
        #print(f"az_min = {az_min}")
        az_data_breaks = np.round(np.linspace(az_min, data_min_max[1], az_num_color))
        #print(f"az_data_breaks = {az_data_breaks}")
        
    elif cmap_method=='pseudo_hist_norm':
        my_percentiles = np.linspace(0, 100, az_num_color)
        my_percentiles = my_percentiles[1:-1]
        #print(f"my_percentiles = {my_percentiles}")
        
        az_data_breaks = np.percentile(a=az_clip_data, q=my_percentiles)
        if data_min_max[0]<0:
            az_data_breaks = np.insert(az_data_breaks, 0, 0) #prepend with: if data_min_max[0]<0, then 0; 
        else:    
            az_data_breaks = np.insert(az_data_breaks, 0, data_min_max[0]) #prepend with: if data_min_max[0]>=0, then data_min_max[0]
        az_data_breaks = np.append(az_data_breaks, data_min_max[1])
        #print(f"az_data_breaks = {az_data_breaks}")
    
    if data_min_max[0]<0:
        data_breaks = [data_min_max[0]]
    else:
        data_breaks = []
    #print(f"data_breaks = {data_breaks}")
    data_breaks.extend(az_data_breaks)
    data_breaks = np.array(data_breaks)
    data_breaks = data_breaks.astype(datatype).tolist()
    #print(f"data_breaks = {data_breaks}")
    
    no_dum_data = data.copy()
    no_dum_data = no_dum_data.flatten()
    no_dum_data = no_dum_data.astype(float)
    if no_data_value is not None:
        no_dum_data[no_dum_data==no_data_value] = np.nan
    abs_min_max = [np.nanmin(no_dum_data), np.nanmax(no_dum_data)]
    #percentile_breaks = np.round((np.array(data_breaks) - abs_min_max[0]) / (abs_min_max[1] - abs_min_max[0]), 4) # relative to the specified min,max
    percentile_breaks = np.round((np.array(data_breaks) - data_min_max[0]) / (data_min_max[1] - data_min_max[0]), 4) # relative to the specified min,max
    percentile_breaks = percentile_breaks.tolist()
    #print(f"percentile_breaks = {percentile_breaks}")

    
    if plot_histograms:
        norm_no_dum_data = (no_dum_data.copy() - abs_min_max[0]) / (abs_min_max[1] - abs_min_max[0]) # shift to zero, then normalize by the range
        
        fig, ax = plt.subplots(nrows=1, ncols=1, sharey=True, figsize=[12, 3])

        ax.hist(no_dum_data, bins=min(abs_min_max[1]-abs_min_max[0], 100))
        ylimits = ax.get_ylim()
        ax.plot([abs_min_max[0], abs_min_max[0]], [0, ylimits[1]], c='k', ls=":", lw=1, label=abs_min_max[0])
        for ii, db in enumerate(data_breaks):
            ax.plot([db, db], [0, ylimits[1]], c='k', lw=2, label=None)
            ax.plot([db, db], [0, ylimits[1]], c=color_list[ii], lw=1, label=db)
        ax.plot([abs_min_max[1], abs_min_max[1]], [0, ylimits[1]], c='k', ls="--", lw=1, label=abs_min_max[1])
        ax.set_ylim(ylimits)
        ax.legend(loc='center right', bbox_to_anchor=(1.11, 0.5))
        ax.set_title('Breaks by data values')
        
        plt.tight_layout(); plt.show()

    return data_breaks

######################################

def calc_data_min_max(data, no_data_value, clip_perc=None, min_max_method='data_absolute'):
    '''Function to calculate the min max values of the data
Input Parameters:
    - data: 2D array of cell values of the raster.
        ex: ([[x1y1, ..., xny1], [x1y2, ..., xny2], ..., [x1yn, ..., xnyn]])
        
    - no_data_value: value the defines the no-data-value
    
    - clip_perc: list of percentile (quartile) values to clip the data to. i.e. [1, 99] for clip to the fist and 99th percentiles of the data (essentialy, exclude erroneous high or low points). This is only relevant if the min_max_method is 'percentile'.
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
Luts to build and write: 
    1) map the data to to rgba, 
    2) map data to uint8, 
    3) map uint8 data to r, g, b, a files, and 
    4) map uint8 data to a rgba file
    5) QGIS file

def build_color_lookup_tables(cmap, data_breaks, dtype, no_data_value, new_multiband_lut_path, single_band_tiff_path=None, write_lut_files=False):
    '''description
Input Parameters:
    - cmap: List of hex color codes. ex: EMerald_custom_colors_hexcolorcodes
    
    - data_breaks: List of data values to map the cmap too
    
    #- data: 2D array of cell values of the raster.
    #    ex: ([[x1y1, ..., xny1], [x1y2, ..., xny2], ..., [x1yn, ..., xnyn]])
    - dtype: data_type of raster data - probably 'int', or 'float'
    
    - no_data_value: Value that specifies the no_data_value
    
    - new_multiband_lut_path: Full path (without extionsion) to the desired files. Color component and extention will appended to the filename.
        ex: red file: new_multiband_lut_path + '_r.lut
    
    See this discussion on how to effectively use LUTs in a mapserver mapfile: https://github.com/emerald-geomodelling/disco-tif/issues/4#issuecomment-2084928873
    '''
    colors_rgb = pd.DataFrame()
    for ii in range(0, len(cmap)):
        hexcolor = cmap[ii]
        colors_rgb.loc[ii, ['r']] = hex_to_rgb(hexcolor)[0]
        colors_rgb.loc[ii, ['g']] = hex_to_rgb(hexcolor)[1]
        colors_rgb.loc[ii, ['b']] = hex_to_rgb(hexcolor)[2]
    colors_rgb.loc[:,'a'] = 255
    if len(colors_rgb) == len(data_breaks):
        colors_rgb['data_val'] = np.array(data_breaks)
    else:
        print("there's an odd mismatch in length of 'cmap' and 'data_breaks'")

    rgba_lut_dict = {}
    for ii in range(0, len(colors_rgb)):
        rgba_lut_dict[colors_rgb.loc[ii,'data_val'].astype(dtype)] = (np.round(colors_rgb.loc[ii,'r']).astype(int), np.round(colors_rgb.loc[ii,'g']).astype(int), np.round(colors_rgb.loc[ii,'b']).astype(int), np.round(colors_rgb.loc[ii,'a']).astype(int))
    #print(f"rgba_lut_dict = \n{rgba_lut_dict}")
    
    if not write_lut_files:
        return rgba_lut_dict 
    else:
        outfilepaths=[]
        
        #############################################
        # write RBGA single band files - do we want the no-data value in the color lut files too?
        for rgb in ['r', 'g', 'b']:
            lut_str_data = f"{colors_rgb.loc[0, 'data_val']}:{int(np.round(colors_rgb.loc[0, rgb]))}"
            for row in range(1, len(colors_rgb)):
                    lut_str_data = f"{lut_str_data},{colors_rgb.loc[row, 'data_val']}:{int(np.round(colors_rgb.loc[row, rgb]))}"
            tname_data = f"{new_multiband_lut_path}_data_{rgb}.lut"
            outfilepaths.append(tname_data)
            with open(tname_data, 'w') as lut_file_out:
                lut_file_out.write(lut_str_data)
        lut_str_data = f"{np.array(no_data_value, dtype=dtype).tolist()}:0,{data_breaks[0]}:255,{data_breaks[-1]}:255"
        tname_data=f"{new_multiband_lut_path}_data_a.lut"
        outfilepaths.append(tname_data)    
        with open(tname_data, 'w') as lut_file_out:
            lut_file_out.write(lut_str_data)
        
        #############################################
        # # write RBGA single band files - Normalized by data range
        # for rgb in ['r', 'g', 'b']:
        #     lut_str_norm = f"{int(np.round(((colors_rgb.loc[0, 'data_val'] - data_breaks[0])/(data_breaks[-1] - data_breaks[0]))*255))}:{int(np.round(colors_rgb.loc[0, rgb]))}"
        #     for row in range(1, len(colors_rgb)):
        #             lut_str_norm = f"{lut_str_norm},{int(np.round(((colors_rgb.loc[row, 'data_val'] - data_breaks[0])/(data_breaks[-1] - data_breaks[0]))*255))}:{int(np.round(colors_rgb.loc[row, rgb]))}"
        #     tname_norm = f"{new_multiband_lut_path}_norm_{rgb}.lut"
        #     outfilepaths.append(tname_norm)
        #     with open(tname_norm, 'w') as lut_file_out:
        #         lut_file_out.write(lut_str_norm)
        # lut_str_norm = f"{np.array(no_data_value, dtype=dtype).tolist()}:0,0:255,1:255"
        # tname_norm=f"{new_multiband_lut_path}_norm_a.lut"
        # outfilepaths.append(tname_norm)    
        # with open(tname_norm, 'w') as lut_file_out:
        #     lut_file_out.write(lut_str_norm)
    
        #############################################
        # Write 4channgel lutfile Short- meaning only primary breaks
        colors_rgb['a']=255
    
        tname=f"{new_multiband_lut_path}_short_rgba.lut"
        outfilepaths.append(tname)    
        with open(tname, 'w') as outlut:
            for ii in range(0, len(colors_rgb)):
                outlut.write(f"{colors_rgb.loc[ii,'data_val'].astype(dtype)},{np.round(colors_rgb.loc[ii,'r']).astype(int)},{np.round(colors_rgb.loc[ii,'g']).astype(int)},{np.round(colors_rgb.loc[ii,'b']).astype(int)},{np.round(colors_rgb.loc[ii,'a']).astype(int)}\n")
        
        #############################################
        # write 4 channel lookup table that is compatible with QGIS
        qgisfile = f"{new_multiband_lut_path}_qgis_color_table_{stoday()}.txt"
        with open(qgisfile, 'w') as outlut:
            if single_band_tiff_path is not None:
                outlut.write(f"# EMerald Generated Color Map Export File for {single_band_tiff_path}\n")
            else:            
                outlut.write(f"# EMerald Generated Color Map Export File for {new_multiband_lut_path}\n")
            outlut.write("INTERPOLATION:INTERPOLATED\n")
            for ii in range(0, len(colors_rgb)):
                outlut.write(f"{colors_rgb.loc[ii,'data_val'].astype(dtype)},{np.round(colors_rgb.loc[ii,'r']).astype(int)},{np.round(colors_rgb.loc[ii,'g']).astype(int)},{np.round(colors_rgb.loc[ii,'b']).astype(int)},{np.round(colors_rgb.loc[ii,'a']).astype(int)},{colors_rgb.loc[ii,'data_val'].astype(dtype)}\n")
        
        #############################################
        # write a 4 channel Lut file that also has nodata values
        colors_rgb.loc[len(colors_rgb),['data_val','r','g','b','a']] = [np.array(no_data_value, dtype=dtype), 128, 128, 128, 0 ]
        colors_rgb.sort_values(by=['data_val'], inplace=True, ignore_index=True)
        tname=f"{new_multiband_lut_path}_NaN_short_rgba.lut"
        outfilepaths.append(tname)    
        with open(tname, 'w') as outlut:
            for ii in range(0, len(colors_rgb)):
                outlut.write(f"{colors_rgb.loc[ii,'data_val'].astype(dtype)},{np.round(colors_rgb.loc[ii,'r']).astype(int)},{np.round(colors_rgb.loc[ii,'g']).astype(int)},{np.round(colors_rgb.loc[ii,'b']).astype(int)},{np.round(colors_rgb.loc[ii,'a']).astype(int)}\n")

        #############################################
        # write a 4 channel Lut file that also has nodata values - Norm by data range
        #colors_rgb["norm_data_val"] = (np.round(((colors_rgb.data_val- data_breaks[0])/(data_breaks[-1] - data_breaks[0]))*254+1)).astype(int)
        #print(f"colors_rgb = {colors_rgb}")
        #colors_rgb.loc[colors_rgb.norm_data_val<1, 'norm_data_val'] = 0
        #print(f"colors_rgb = {colors_rgb}")
        #tname=f"{new_multiband_lut_path}_Norm_NaN_short_rgba.lut"
        #outfilepaths.append(tname)    
        #with open(tname, 'w') as outlut:
        #    for ii in range(0, len(colors_rgb)):
        #        outlut.write(f"{colors_rgb.loc[ii,'norm_data_val'].astype(int)},{np.round(colors_rgb.loc[ii,'r']).astype(int)},{np.round(colors_rgb.loc[ii,'g']).astype(int)},{np.round(colors_rgb.loc[ii,'b']).astype(int)},{np.round(colors_rgb.loc[ii,'a']).astype(int)}\n")
                
        return rgba_lut_dict, outfilepaths
    
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
     if not spcified, this function will automatically calculated min/max values based on the min_max_method.
 
 - min_max_method:
     default = 'percentile'; only relevant if data_min_max==None.
     Also accepts 'data_absolute'. 
     'percentile' uses the percentiles supplied in clip_perc. 
     'data_absolute' uses the minimum and maximum values of the data supplied to the function
 
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

    # make percentile ranges
    data_breaks = make_percentile_array(data_min_max=data_min_max, 
                                        data=data.copy(), 
                                        no_data_value=no_data_value,
                                        color_list=EMerald_custom_colors_hexcolorcodes,
                                        cmap_method=cmap_method, 
                                        plot_histograms=True,
                                       )
        
    # 2. Generate a custom colormap (EMeraldCustomColormap):
    if color_palette_name is None:
        color_palette_name = "EMeraldCustomTerrain"
        percentile_breaks = (data_breaks - data_min_max[0]) / (data_min_max[1] - data_min_max[0])
        EMeraldCustomColormap = build_EMerald_terrain_colormap(percentile_breaks) # only valid for data clipped to this range
    else:
        EMeraldCustomColormap = mpl.colormaps.get_cmap(color_palette_name)
    EMeraldCustomColormap(0) #need this line for it to make a lut?!?
    #print(EMeraldCustomColormap._lut)
    
    # define output name
    sbpath, ext = os.path.splitext(single_band_tiff_path)
    suffix=f"{color_palette_name}_{data_min_max[0]}_to_{data_min_max[1]}_{cmap_method}"
    new_multiband_lut_path = f"{sbpath}_{suffix}"
    
    luts = build_color_lookup_tables(cmap=EMerald_custom_colors_hexcolorcodes,
                                     data_breaks=data_breaks,
                                     dtype=data[0,0].dtype,
                                     no_data_value=no_data_value,
                                     new_multiband_lut_path=new_multiband_lut_path,
                                     single_band_tiff_path=single_band_tiff_path,
                                     write_lut_files=generate_lookup_tables,
                                    )
    if not generate_lookup_tables:
        rgba_lut_dict = luts
        #print(f"rgba_lut_dict = \n{rgba_lut_dict}")
    elif generate_lookup_tables:
        rgba_lut_dict = luts[0]
        outfilepaths = luts[1]
        #print(f"rgba_lut_dict = \n{rgba_lut_dict}")
        for fp in outfilepaths:
            print(f"Wrote LUT files to: '{fp}'")
    
    data_with_nan = data.copy().astype(float)
    if no_data_value is not None:
        data_with_nan[data==no_data_value]=np.nan
    clip_data_with_nan = np.clip(data_with_nan, data_min_max[0], data_min_max[1],)
        
    if plot_rgba_raster:
        figsize = [15, 9]
        fig,ax=plt.subplots(1,1,figsize=figsize)
        ep.plot_bands(clip_data_with_nan,
                      #data_with_nan,
                      cmap = EMeraldCustomColormap,
                      title=f"{sbpath.split(os.path.sep)[-1]}\n{suffix.replace('_', ' ')}",
                      ax=ax,
                     )
        plt.tight_layout()
        plt.show()    
    
    if output_tif:
        write_1_channel_geotifff = True
        write_4_channel_geotifff = False

        if write_1_channel_geotifff:
            # Normalize the clipped data to 1 to 255 range # shift min to zero, divide by the range (reserving 0 for no data)
            norm_clip_data_with_dumb = np.round(((clip_data_with_nan - data_min_max[0]) / (data_min_max[1] - data_min_max[0])) * 254)+1
            filt = data == no_data_value
            norm_clip_data_with_dumb[filt] = 0
            
            norm_rgba_lut_dict = {}
            for key,value in rgba_lut_dict.items():
                new_key = (np.round(((key - data_min_max[0]) / (data_min_max[1] - data_min_max[0])) * 254)+1).astype('uint8')
                norm_rgba_lut_dict[new_key] = value
            norm_rgba_lut_dict[0] = (127, 127, 127, 0)
            norm_rgba_lut_df = pd.DataFrame.from_dict(norm_rgba_lut_dict, orient='index', columns=['r', 'g', 'b', 'a'])
            norm_rgba_lut_df.reset_index(names='pix_val', inplace=True)
            for ii in range(0, 256):
                if ii not in norm_rgba_lut_df.pix_val.values:
                    new_row = {'pix_val':ii, 'r':np.nan, 'g':np.nan, 'b':np.nan, 'a':np.nan} # fill new rows with nans
                    norm_rgba_lut_df = pd.concat([norm_rgba_lut_df, pd.DataFrame([new_row])], ignore_index=True)
            norm_rgba_lut_df.sort_values(by='pix_val', ignore_index=True, inplace=True)
            norm_rgba_lut_df.interpolate(inplace=True) #fill in all nans
            norm_rgba_lut_df = norm_rgba_lut_df.round().astype('uint8')
    
            # build dictionary {key1: (r1, g1, b1, a1), ..., keyn: (rn, gn, bn, an)}
            new_norm_rgba_lut_dict={}
            for pix_val in norm_rgba_lut_df.pix_val.values: 
                new_norm_rgba_lut_dict[pix_val] = (norm_rgba_lut_df.loc[norm_rgba_lut_df.pix_val==pix_val,'r'].values[0],
                                                   norm_rgba_lut_df.loc[norm_rgba_lut_df.pix_val==pix_val,'g'].values[0],
                                                   norm_rgba_lut_df.loc[norm_rgba_lut_df.pix_val==pix_val,'b'].values[0],
                                                   norm_rgba_lut_df.loc[norm_rgba_lut_df.pix_val==pix_val,'a'].values[0],)
    
            norm_clip_data_with_dumb = np.array(norm_clip_data_with_dumb, dtype='uint8')
        
            newprofile = origprofile.copy()
            newprofile.update(dtype='uint8', nodata=0)  # RGBA format

            with rasterio.open(f"{new_multiband_lut_path}.tif", 'w', **newprofile) as dst:
                dst.write(norm_clip_data_with_dumb, indexes=1)
                dst.write_colormap(1, new_norm_rgba_lut_dict)
        
        if write_4_channel_geotifff:
            # Normalize the clipped data to the [0, 1] range
            normalized_data_with_nan = (clip_data_with_nan - data_min_max[0]) / (data_min_max[1] - data_min_max[0])

            # apply EMeraldCustomColormap to data
            rgba_data = EMeraldCustomColormap(normalized_data_with_nan) * (colormap_length-1)  # Scale to 0-255 range
            
            # 3. Generate an Alpha Channel and overwrite:
            alpha_channel = (data != no_data_value).astype('uint8') * (colormap_length-1) # 0 or 255
            rgba_data[:, :, 3] = alpha_channel
            
            # 5. Write the New RGBA GeoTIFF:
            newprofile = origprofile.copy()
            newprofile.update(count=4, dtype='uint8', nodata=None)  # RGBA format
 
            new_multiband_tiff_path = f"{sbpath}_rgba_{suffix}"

            with rasterio.open(f"{new_multiband_tiff_path}.tif", 'w', **newprofile) as dst:
                dst.write(rgba_data[:, :, 0].round().astype('uint8'), 1) #red
                dst.write(rgba_data[:, :, 1].round().astype('uint8'), 2) #green
                dst.write(rgba_data[:, :, 2].round().astype('uint8'), 3) #blue
                dst.write(rgba_data[:, :, 3].round().astype('uint8'), 4) #alpha
        
            print(f"New RGBA geotiff '{new_multiband_tiff_path}' generated successfully!")
    return EMeraldCustomColormap, data_breaks

######################################
