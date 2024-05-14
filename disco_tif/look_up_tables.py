import numpy as np
import pandas as pd
import datetime
import disco_tif.process_sb_tiff


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
def cmap_data_break_to_df(cmap, data_breaks):
    """Function that takes a list of hex color codes and a list of data_breaks and makes a dataframe with rgba from them

Input Parameters:
    cmap: a list of hex color codes
    
    data_breaks: a list of data break relating to the hex colors
    """
    assert len(cmap) == len(data_breaks), f"there's an odd mismatch in length of 'cmap' and 'data_breaks' \n cmap:\n {cmap}\n data_breaks:\n {data_breaks}"
    
    colors_rgba = pd.DataFrame()
    for ii in range(0, len(cmap)):
        hexcolor = cmap[ii]
        colors_rgba.loc[ii, ['r']] = disco_tif.process_sb_tiff.hex_to_rgb(hexcolor)[0]
        colors_rgba.loc[ii, ['g']] = disco_tif.process_sb_tiff.hex_to_rgb(hexcolor)[1]
        colors_rgba.loc[ii, ['b']] = disco_tif.process_sb_tiff.hex_to_rgb(hexcolor)[2]
    colors_rgba.loc[:, 'a'] = 255
    colors_rgba['data_val'] = np.array(data_breaks)
    
    return colors_rgba


def short_color_table_to_long_color_table(colors_rgba, no_data_value):
    """ Function takes a short dataframe of data breaks and colors and linearly expands them to a 256 (or 255 if no_data_value is not None) rows

Input Parameters:
    colors_rgba: Dataframe of data values and rgba values
    
    no_data_value: value that represents no_data in the dataset
    """
    data_breaks = colors_rgba.loc[:, 'data_val'].values

    if no_data_value is None:
        uint8_data_breaks = (np.round((       data_breaks  - np.min(data_breaks)) /
                                      (np.max(data_breaks) - np.min(data_breaks)) * 255)).astype('uint8').tolist()
    # elif no_data_value is not None:
    else:
        uint8_data_breaks = (np.round((       data_breaks  - np.min(data_breaks)) /
                                      (np.max(data_breaks) - np.min(data_breaks)) * 254)+1).astype('uint8').tolist()
    
    t_colors_rgba = pd.DataFrame(np.ones(shape=(256, 5))*np.nan, columns=colors_rgba.columns)
    t_colors_rgba.loc[uint8_data_breaks, ['data_val', 'r', 'g', 'b', 'a']] = colors_rgba[['data_val', 'r', 'g', 'b', 'a']].values
    colors_rgba = t_colors_rgba.interpolate(method='linear', axis=0)
    
    colors_rgba.loc[:, 'data_val'] = colors_rgba.loc[:, 'data_val'].round(3)
    colors_rgba.loc[uint8_data_breaks, 'data_val'] = data_breaks
    if no_data_value is not None:
        colors_rgba = colors_rgba.drop(index=0).reset_index(drop=True)
        print(f"colors_rgba = {colors_rgba}")
    colors_rgba.loc[:, ['r', 'g', 'b', 'a']] = colors_rgba.loc[:, ['r', 'g', 'b', 'a']].round().astype('uint8')
    
    return colors_rgba
    
######################################


def rast_dat_to_QGIS_lut(cmap, data_breaks, dtype, output_GIS_lut_path, path_to_orig_geotiff=None, short_file=True, print_to_screen=False):
    """Function to take a set of colors and data breaks and write to a QGIS (possibly other GIS software programs too) compatible color table to apply to a single band geotiff.
    
Input Parameters:
    - cmap: List of hex color codes. ex: EMerald_custom_colors_hexcolorcodes
    
    - data_breaks: List of data values to map the cmap too
    
    - dtype: data_type of raster data - probably 'int', or 'float'
    
    - output_GIS_lut_path: Full path (without extension) to the desired files. Color component and extension will append to the filename.
        ex: f"{output_GIS_lut_path}_qgis_color_table_{stoday()}.txt"

    - short_file: boolean. If False the resulting file will be 256 entries long.
        Default: True
    """
    
    colors_rgba = cmap_data_break_to_df(cmap, data_breaks)

    if not short_file:
        colors_rgba = short_color_table_to_long_color_table(colors_rgba, no_data_value=None)

    if 'int' in str(dtype):
        colors_rgba.loc[:, 'data_val'] = colors_rgba.loc[:, 'data_val'].round()
    
    # write 4 channel lookup table that is compatible with QGIS
    qgisfile = f"{output_GIS_lut_path}_qgis_color_table_{len(colors_rgba)}-colors_{stoday()}.txt"
    if print_to_screen:
        print("\nQGIS lut file contents:\n")
    with open(qgisfile, 'w') as outlut:
        str_out = f"# EMerald Generated Color Map Export File for {path_to_orig_geotiff}\n"
        outlut.write(str_out)
        if print_to_screen: print(str_out)

        str_out = "INTERPOLATION:INTERPOLATED\n"
        outlut.write(str_out)
        if print_to_screen: print(str_out)

        for ii in range(0, len(colors_rgba)):
            str_out = f"{colors_rgba.loc[ii,'data_val'].astype(dtype)},{np.round(colors_rgba.loc[ii,'r']).astype(int)},{np.round(colors_rgba.loc[ii,'g']).astype(int)},{np.round(colors_rgba.loc[ii,'b']).astype(int)},{np.round(colors_rgba.loc[ii,'a']).astype(int)},{colors_rgba.loc[ii,'data_val'].astype(dtype)}\n"
            outlut.write(str_out)
            if print_to_screen: print(str_out)

    print(f"\nGIS software compatible color look up table successfully written to: \n\t- {qgisfile}")


######################################
def rast_dat_to_single_channel_rgba_luts(cmap, data_breaks, dtype, no_data_value, lut_outpath_base, short_file=True, print_to_screen=False):
    """Function that maps data values to look up tables for eac color in r, g, b, and a.
    
Input Parameters:
    - cmap: List of hex color codes. ex: EMerald_custom_colors_hexcolorcodes
    
    - data_breaks: List of data values to map the cmap too
    
    - dtype: data_type of raster data - probably 'int', or 'float'
    
    - no_data_value: Value that specifies the no_data_value. I
    
    - lut_outpath_base: Full path (without extionsion) to the desired files. Color component and extention will appended to the filename.
        ex: red file: lut_outpath_base + '_r.lut
    
    - short_file: boolean. If False the resulting file will be 256 entries long.
        Default: True

    See this discussion on how to effectively use LUTs in a mapserver mapfile: https://github.com/emerald-geomodelling/disco-tif/issues/4#issuecomment-2084928873
    """
    colors_rgba = cmap_data_break_to_df(cmap, data_breaks)

    if not short_file:
        colors_rgba = short_color_table_to_long_color_table(colors_rgba, no_data_value)

    if 'int' in str(dtype):
        colors_rgba.loc[:, 'data_val'] = colors_rgba.loc[:, 'data_val'].round()

    if no_data_value is not None:
        ndict = pd.DataFrame({'data_val': no_data_value, 'r': 0, 'g': 0, 'b': 0, 'a': 0}, index=[0])
        colors_rgba = pd.concat([colors_rgba, ndict], ignore_index=True)
        colors_rgba.sort_values(by=['data_val'], inplace=True, ignore_index=True)
    
    # write RGBA single band files
    outfilepaths = []
    for rgba in ['r', 'g', 'b', 'a']:
        if print_to_screen:
            print(f"\nsingle lut per channel file contents - {rgba}:\n")
        lut_str_data = f"{colors_rgba.loc[0, 'data_val']}:{int(np.round(colors_rgba.loc[0, rgba]))}"
        for row in range(1, len(colors_rgba)):
            lut_str_data = f"{lut_str_data},{colors_rgba.loc[row, 'data_val']}:{int(np.round(colors_rgba.loc[row, rgba]))}"
        tname = f"{lut_outpath_base}_data_{len(colors_rgba)}-colors_{stoday()}_{rgba}.lut"
        outfilepaths.append(tname)
        with open(tname, 'w') as lut_file_out:
            lut_file_out.write(lut_str_data)
        if print_to_screen: print(lut_str_data)

    print(f"\nSingle-channel LUT files mapping data to color values have been written to: \n\t- {outfilepaths[0]} \n\t- {outfilepaths[1]} \n\t- {outfilepaths[2]} \n\t- {outfilepaths[3]}")


def rast_dat_to_multi_channel_rgba_lut(cmap,
                                       data_breaks,
                                       dtype,
                                       no_data_value,
                                       lut_outpath_base,
                                       short_file=True,
                                       print_to_screen=False):
    """Function that takes raster data and writes a single rgba lut to disk.
    
Input Parameters:
    - cmap: List of hex color codes. ex: EMerald_custom_colors_hexcolorcodes
    
    - data_breaks: List of data values to map the cmap too
    
    - dtype: data_type of raster data - probably 'int', or 'float'
    
    - no_data_value: Value that specifies the no_data_value
    
    - lut_outpath_base: Full path (without extension) to the desired files. Color component and extension will append to the filename.
        ex: red file: lut_outpath_base + '_r.lut
    
    - short_file: boolean. If False the resulting file will be 256 entries long.
        Default: True

    See this discussion on how to effectively use LUTs in a mapserver mapfile: https://github.com/emerald-geomodelling/disco-tif/issues/4#issuecomment-2084928873
    """
    colors_rgba = cmap_data_break_to_df(cmap, data_breaks)

    if not short_file:
        colors_rgba = short_color_table_to_long_color_table(colors_rgba, no_data_value)

    if 'int' in str(dtype):
        colors_rgba.loc[:, 'data_val'] = colors_rgba.loc[:, 'data_val'].round()

    if no_data_value is not None:
        ndict = pd.DataFrame({'data_val': no_data_value, 'r': 0, 'g': 0, 'b': 0, 'a': 0}, index=[0])
        colors_rgba = pd.concat([colors_rgba, ndict], ignore_index=True)
        colors_rgba.sort_values(by=['data_val'], inplace=True, ignore_index=True)    
        
    tname = f"{lut_outpath_base}_data_{len(colors_rgba)}-colors_{stoday()}_rgba.lut"
    if print_to_screen:
        print(f"\nmulti-channel lut file contents:\n")
    with open(tname, 'w') as outlut:
        for ii in range(0, len(colors_rgba)):
            str_out = f"{colors_rgba.loc[ii,'data_val'].astype(dtype)},{np.round(colors_rgba.loc[ii,'r']).astype('uint8')},{np.round(colors_rgba.loc[ii,'g']).astype('uint8')},{np.round(colors_rgba.loc[ii,'b']).astype('uint8')},{np.round(colors_rgba.loc[ii,'a']).astype('uint8')}\n"
            outlut.write(str_out)
            if print_to_screen: print(str_out)

    print(f"\nmulti-channel LUT file mapping data to color values has been written to: \n\t- {tname}")
    

def uint8dat_to_single_channel_rgba_luts(cmap, data_breaks, dtype, no_data_value, lut_outpath_base, short_file=True, print_to_screen=False):
    """Function that writes uint8 data to individual color lut files to disk.
    
Input Parameters:
    - cmap: List of hex color codes. ex: EMerald_custom_colors_hexcolorcodes
    
    - data_breaks: List of data values to map the cmap too
    
    - dtype: data_type of raster data - probably 'int', or 'float'
    
    - no_data_value: Value that specifies the no_data_value
    
    - lut_outpath_base: Full path (without extension) to the desired files. Color component and extension will append to the filename.
        ex: red file: lut_outpath_base + '_r.lut

    - short_file: boolean. If False the resulting file will be 256 entries long.
        Default: True

    See this discussion on how to effectively use LUTs in a mapserver mapfile: https://github.com/emerald-geomodelling/disco-tif/issues/4#issuecomment-2084928873
    """
    colors_rgba = cmap_data_break_to_df(cmap, data_breaks)

    if not short_file:
        colors_rgba = short_color_table_to_long_color_table(colors_rgba, no_data_value)

    if 'int' in str(dtype):
        colors_rgba.loc[:, 'data_val'] = colors_rgba.loc[:, 'data_val'].round()

    datamin = np.min(data_breaks)
    datamax = np.max(data_breaks)    
    if no_data_value is None:
        colors_rgba.loc[:, 'data_uint8_val'] = np.round(((colors_rgba.loc[:, 'data_val']-datamin) / (datamax-datamin))*255).astype('uint8')
    elif no_data_value is not None:
        colors_rgba.loc[:, 'data_uint8_val'] = np.round((((colors_rgba.loc[:, 'data_val']-datamin) / (datamax-datamin))*254)+1).astype('uint8')
        
        ndict = pd.DataFrame({'data_val': no_data_value, 'data_uint8_val': 0, 'r': 0, 'g': 0, 'b': 0, 'a': 0}, index=[0])
        colors_rgba = pd.concat([colors_rgba, ndict], ignore_index=True)
    colors_rgba.sort_values(by=['data_uint8_val'], inplace=True, ignore_index=True)

    # write RBGA single band files
    outfilepaths = []
    for rgba in ['r', 'g', 'b', 'a']:
        if print_to_screen:
            print(f"\nsingle lut per channel file contents - UINT8 - {rgba}:\n")
        lut_str_data = f"{colors_rgba.loc[0, 'data_uint8_val']}:{int(np.round(colors_rgba.loc[0, rgba]))}"
        for row in range(1, len(colors_rgba)):
            lut_str_data = f"{lut_str_data},{colors_rgba.loc[row, 'data_uint8_val']}:{int(np.round(colors_rgba.loc[row, rgba]))}"
        tname = f"{lut_outpath_base}_uint8_{len(colors_rgba)}-colors_{stoday()}_{rgba}.lut"
        outfilepaths.append(tname)
        with open(tname, 'w') as lut_file_out:
            lut_file_out.write(lut_str_data)
        if print_to_screen: print(lut_str_data)

    print(f"\nSingle-channel LUT files mapping uint8-data to color values have been written to: \n\t- {outfilepaths[0]} \n\t- {outfilepaths[1]} \n\t- {outfilepaths[2]} \n\t- {outfilepaths[3]}")


def uint8dat_to_multi_channel_rgba_lut(cmap, data_breaks, dtype, no_data_value, lut_outpath_base, short_file=True, print_to_screen=False):
    """This function will write a look-up-table that maps uint8 data vales to r,g,b,a colors. No-data-values will be mapped to 0, min values will be mapped to 1, max values will be mapped to 255
    
Input Parameters:
    - cmap: List of hex color codes. ex: EMerald_custom_colors_hexcolorcodes
    
    - data_breaks: List of data values to map the cmap too
    
    - dtype: data_type of raster data - probably 'int', or 'float'
    
    - no_data_value: Value that specifies the no_data_value
    
    - lut_outpath_base: Full path (without extionsion) to the desired files. Color component and extention will appended to the filename.
        ex: red file: lut_outpath_base + '_r.lut
    
    - short_file: boolean. If False the resulting file will be 256 entries long.
        Default: True

    See this discussion on how to effectively use LUTs in a mapserver mapfile: https://github.com/emerald-geomodelling/disco-tif/issues/4#issuecomment-2084928873
    """
    colors_rgba = cmap_data_break_to_df(cmap, data_breaks)
    
    if not short_file:
        colors_rgba = short_color_table_to_long_color_table(colors_rgba, no_data_value)

    if 'int' in str(dtype):
        colors_rgba.loc[:, 'data_val'] = colors_rgba.loc[:, 'data_val'].round()
        
    datamin = np.min(data_breaks)
    datamax = np.max(data_breaks)
    if no_data_value is None:
        colors_rgba.loc[:, 'data_uint8_val'] = np.round(((colors_rgba.loc[:, 'data_val']-datamin) / (datamax-datamin))*255).astype('uint8')
    elif no_data_value is not None:
        colors_rgba.loc[:, 'data_uint8_val'] = np.round((((colors_rgba.loc[:, 'data_val']-datamin) / (datamax-datamin))*254)+1).astype('uint8')
    
        ndict = pd.DataFrame({'data_val': no_data_value, 'data_uint8_val': 0, 'r': 0, 'g': 0, 'b': 0, 'a': 0}, index=[0])
        colors_rgba = pd.concat([colors_rgba, ndict], ignore_index=True)
        colors_rgba.sort_values(by=['data_uint8_val'], inplace=True, ignore_index=True)

    if print_to_screen:
        print(f"\nmulti-channel lut file contents - UINT8:\n")
    tname = f"{lut_outpath_base}_uint8_{len(colors_rgba)}-colors_rgba.lut"
    with open(tname, 'w') as outlut:
        for ii in range(0, len(colors_rgba)):
            str_out = f"{colors_rgba.loc[ii,'data_uint8_val'].astype('uint8')},{np.round(colors_rgba.loc[ii,'r']).astype('uint8')},{np.round(colors_rgba.loc[ii,'g']).astype('uint8')},{np.round(colors_rgba.loc[ii,'b']).astype('uint8')},{np.round(colors_rgba.loc[ii,'a']).astype('uint8')}\n"
            outlut.write(str_out)
            if print_to_screen: print(str_out)
    
    print(f"\nmulti-channel LUT file mapping data to color values has been written to: \n\t- {tname}")


def data_to_uint8_lut(data_breaks, no_data_value, lut_outpath_base, print_to_screen=False):
    """Function to write a look-up-table for data to uint8 values. This will map the no-data values to 0, the mindata value to 1, and the max data value to 255.

Input Parameters:
    - data_breaks: List of data values to map the cmap too
    
    - no_data_value: Value that specifies the no_data_value
    
    - lut_outpath_base: Full path (without extionsion) to the desired files. Color component and extention will appended to the filename.
        ex: red file: lut_outpath_base + '_data-uint8.lut'
    
    """
    my_dat_b = data_breaks.copy()
    datamin = np.min(data_breaks)
    datamax = np.max(data_breaks)
    my_uint8_b = []

    for nowdatbreak in my_dat_b:
        if no_data_value is None:
            my_uint8_b.append(np.round(((nowdatbreak - datamin) / (datamax - datamin)) * 255).astype('uint8'))
        elif no_data_value is not None:
            my_uint8_b.append((np.round(((nowdatbreak - datamin) / (datamax - datamin)) * 254)+1).astype('uint8'))
    if no_data_value is not None:
        my_dat_b.append(no_data_value)
        my_uint8_b.append(0)
    data_to_uint8_lut_dict = {'data_breaks': my_dat_b, 'uint8_breaks': my_uint8_b}
    
    data_to_uint8_lut_df = pd.DataFrame(data_to_uint8_lut_dict)
    data_to_uint8_lut_df.sort_values(by=['uint8_breaks'], inplace=True, ignore_index=True)

    if print_to_screen:
        print(f"\ndata value to uint8 lut file contents:\n")
    tname = f"{lut_outpath_base}_data-uint8_{len(data_to_uint8_lut_df)}-breaks_{stoday()}.lut"
    lut_str_data = f"{data_to_uint8_lut_df.loc[0, 'data_breaks']}:{data_to_uint8_lut_df.loc[0, 'uint8_breaks']}"
    for row in range(1, len(data_to_uint8_lut_df)):
        lut_str_data = f"{lut_str_data},{data_to_uint8_lut_df.loc[row, 'data_breaks']}:{data_to_uint8_lut_df.loc[row, 'uint8_breaks']}"
    with open(tname, 'w') as lut_file_out:
        lut_file_out.write(lut_str_data)
    if print_to_screen: print(lut_str_data)

    print(f"\nSingle-channel LUT file mapping data values to uint8 values has been written to: \n\t- {tname}")
    

def rgba_lut_dict_builder(cmap, data_breaks, no_data_value, dtype, short_file=True):
    """build a lut dictionary. The key is the data break, the value is a tuple with rgba values
    
Input Parameters:
    - cmap: List of hex color codes. ex: EMerald_custom_colors_hexcolorcodes
    
    - data_breaks: List of data values to map the cmap too
    
    - no_data_value:

    - dtype: data_type of raster data - probably 'int', or 'float'

    - short_file: boolean. If False the resulting file will be 256 entries long.
        Default: True
    
   See this discussion on how to effectively use LUTs in a mapserver mapfile: https://github.com/emerald-geomodelling/disco-tif/issues/4#issuecomment-2084928873
    """
    colors_rgba = cmap_data_break_to_df(cmap, data_breaks)

    if not short_file:
        colors_rgba = short_color_table_to_long_color_table(colors_rgba, no_data_value)
    
    if 'int' in str(dtype):
        colors_rgba.loc[:, 'data_val'] = colors_rgba.loc[:, 'data_val'].round()
    
    rgba_lut_dict = {}
    for ii in range(0, len(colors_rgba)):
        rgba_lut_dict[colors_rgba.loc[ii, 'data_val'].astype(dtype)] = (np.round(colors_rgba.loc[ii, 'r']).astype(int), np.round(colors_rgba.loc[ii, 'g']).astype(int), np.round(colors_rgba.loc[ii, 'b']).astype(int), np.round(colors_rgba.loc[ii, 'a']).astype(int))
    
    return rgba_lut_dict
    

def short_data_lut_to_long_uint8_lut(rgba_lut_dict, no_data_value):
    """Converts a short rgba_lut_dictionary object that maps data-breaks to colors into a long, 256 colors, uint8
    dictionary. if No_data is None data range in the dictionary will map to 0-255. if no_data is a value then this will
    be inserted into the results and will be mapped to a value of 0 and color of black-transparent

Input parameters:
    rgba_lut_dict: dictionary of datavalues and rgba tuple
    
    no_data_value: value that represents no-data in the dataset.
    """
    datamin = np.min(np.fromiter(rgba_lut_dict.keys(), dtype=float))
    datamax = np.max(np.fromiter(rgba_lut_dict.keys(), dtype=float))

    uint8_rgba_lut_dict = {}
    for key, value in rgba_lut_dict.items():
        if no_data_value is None:
            new_key = (np.round(((key - datamin) / (datamax - datamin)) * 255)).astype('uint8')
        # elif no_data_value is not None:
        else:
            new_key = (np.round(((key - datamin) / (datamax - datamin)) * 254)+1).astype('uint8')
        uint8_rgba_lut_dict[new_key] = value
    
    if no_data_value is not None:
        uint8_rgba_lut_dict[0] = (0, 0, 0, 0)
    
    uint8_rgba_lut_df = pd.DataFrame.from_dict(uint8_rgba_lut_dict, orient='index', columns=['r', 'g', 'b', 'a'])
    uint8_rgba_lut_df.reset_index(names='pix_val', inplace=True)
    for ii in range(0, 256):
        if ii not in uint8_rgba_lut_df.pix_val.values:
            new_row = {'pix_val': ii, 'r': np.nan, 'g': np.nan, 'b': np.nan, 'a': np.nan}  # fill new rows with nans
            uint8_rgba_lut_df = pd.concat([uint8_rgba_lut_df, pd.DataFrame([new_row])], ignore_index=True)
    uint8_rgba_lut_df.sort_values(by='pix_val', ignore_index=True, inplace=True)
    uint8_rgba_lut_df.interpolate(inplace=True)  # fill in all nans
    uint8_rgba_lut_df = uint8_rgba_lut_df.round().astype('uint8')

    # build dictionary {key1: (r1, g1, b1, a1), ..., keyn: (rn, gn, bn, an)}
    new_uint8_rgba_lut_dict = {}
    for pix_val in uint8_rgba_lut_df.pix_val.values: 
        new_uint8_rgba_lut_dict[pix_val] = (uint8_rgba_lut_df.loc[uint8_rgba_lut_df.pix_val == pix_val, 'r'].values[0],
                                            uint8_rgba_lut_df.loc[uint8_rgba_lut_df.pix_val == pix_val, 'g'].values[0],
                                            uint8_rgba_lut_df.loc[uint8_rgba_lut_df.pix_val == pix_val, 'b'].values[0],
                                            uint8_rgba_lut_df.loc[uint8_rgba_lut_df.pix_val == pix_val, 'a'].values[0],)

    return new_uint8_rgba_lut_dict
    