import os
import numpy as np
import rasterio
import datetime

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

def generate_LUT_based_mapfile(single_band_tiff_path, mapfile_path, writeLUTs=True, writeTifs=False, writeHillshades=True):
    '''
    funtion to take two geotiffs (single band and multiband) and generate a multi-layer mapsource mapfile
    single_band_tiff_path: path to single band geotiff
    rgba_tiff_path: path to multiband geotiff
    mapfile_path: path of output mapfile
    - single_band_tiff_path:
    
    - mapfile_path:
    
    - writeLUTs=True:
    
    - writeTifs=False:
    
    - writeHillshades=True
    '''

    #print(f"single_band_tiff_path = {single_band_tiff_path}")
    single_band_tiff_dir = os.path.dirname(single_band_tiff_path)
    #print(f"single_band_tiff_dir  = {single_band_tiff_dir}")
    single_band_file = (single_band_tiff_path.split(single_band_tiff_dir)[1]).split(os.path.sep)[1]
    #print(f"single_band_file  = {single_band_file}")
    
    lut_files_1b = []
    lut_files_4b = []
    for file in os.listdir(single_band_tiff_dir):
        if file.startswith(os.path.splitext(single_band_file)[0]):
            if file.endswith("_r.lut"):
                lut_files_1b.append(file[:-6])
            elif np.logical_and('rgba' in file, file.endswith(".lut")):
                lut_files_4b.append(file)
    #print(f"lut_files = {lut_files}")

    hillshade_files = []
    tif_files = []
    for file in os.listdir(single_band_tiff_dir):
        if file.startswith(os.path.splitext(single_band_file)[0]):
            #if file.endswith("hillshade.tif"):
            if np.logical_and('hillshade' in file, file.endswith(".tif")):
                hillshade_files.append(file)
            elif file.endswith(".tif"):
                tif_files.append(file)
    tif_files = [tiff for tiff in tif_files if tiff!=single_band_file]
    hillshade_files = sorted(hillshade_files)
    tif_files = sorted(tif_files)

    # Open the single-band GeoTIFF
    with rasterio.open(single_band_tiff_path, 'r') as src:
        data = src.read(1)  # Read the first band
        #origprofile = src.profile
        extent = src.bounds
        size = (src.width, src.height)
        epsg_code = src.crs.to_epsg() if src.crs else None
        no_data_value = src.nodata  # Get the no-data value from the GeoTIFF

    no_data_value = np.array(no_data_value, dtype=data[0,0].dtype).tolist()
    data_min_max = np.min(data[data!=no_data_value]), np.max(data)
    #print(data_min_max)
    
    # Define the paths to your GeoTIFF files
    name = single_band_file.split('.')[0]

    # Create a basic Mapfile template
    mapfile_template = f"""## Mapfile written on {today()} at {nowTime()}
MAP
    NAME "{name}"
    EXTENT {np.round(extent.left, 7)} {np.round(extent.bottom, 7)} {np.round(extent.right, 7)} {np.round(extent.top, 7)}
    SIZE {size[0]} {size[1]}
    PROJECTION
        "init=EPSG:{epsg_code}"  # Set the EPSG code
    END

    WEB
        METADATA
          'wms_title'          '{name}'
          'wms_enable_request' '*'
          'wms_srs'            'EPSG:{epsg_code}'
        END 
    END
    
    LAYER
        NAME "{single_band_file.split('.')[0]}"
        DATA "{single_band_file}"
        TYPE RASTER
        STATUS ON
        EXTENT {np.round(extent.left, 7)} {np.round(extent.bottom, 7)} {np.round(extent.right, 7)} {np.round(extent.top, 7)}
        PROJECTION
            "init=EPSG:{epsg_code}"
        END
        PROCESSING "NODATA={no_data_value}"  # Set your desired Nodata value
        PROCESSING "LUT={data_min_max[0]}:0, {data_min_max[1]}:255"  # Define the LUT (elevation:color)
    END
    """
    if writeLUTs:
        for lut in lut_files_1b:
            mapfile_template = f"""{mapfile_template}
    LAYER
        NAME "{lut}"
        DATA "{single_band_file}"
        TYPE RASTER
        STATUS ON
        EXTENT {np.round(extent.left, 7)} {np.round(extent.bottom, 7)} {np.round(extent.right, 7)} {np.round(extent.top, 7)}
        PROJECTION
            "init=EPSG:{epsg_code}"
        END
        PROCESSING "BANDS=1,1,1,1"
        PROCESSING "LUT_1={lut}_r.lut"
        PROCESSING "LUT_2={lut}_g.lut"
        PROCESSING "LUT_3={lut}_b.lut"
        PROCESSING "LUT_4={lut}_a.lut"
    END
"""
        for lut in lut_files_4b:
            mapfile_template = f"""{mapfile_template}
    LAYER
        NAME "{lut}"
        DATA "{single_band_file}"
        TYPE RASTER
        STATUS ON
        EXTENT {np.round(extent.left, 7)} {np.round(extent.bottom, 7)} {np.round(extent.right, 7)} {np.round(extent.top, 7)}
        PROJECTION
            "init=EPSG:{epsg_code}"
        END
        PROCESSING "BANDS=1,1,1,1"
        PROCESSING "LUT={lut}"
    END
"""

    if writeTifs:
        for tiff in tif_files:
            # Open the single-band GeoTIFF
            with rasterio.open(os.path.join(single_band_tiff_dir, tiff), 'r') as src:
                #data = src.read(1)  # Read the first band
                #origprofile = src.profile
                extent = src.bounds
                size = (src.width, src.height)
                epsg_code = src.crs.to_epsg() if src.crs else None
            mapfile_template = f"""{mapfile_template}
    LAYER
        NAME "{tiff.split('.')[0]}"
        DATA "{tiff}"
        TYPE RASTER
        STATUS ON
        EXTENT {np.round(extent.left, 7)} {np.round(extent.bottom, 7)} {np.round(extent.right, 7)} {np.round(extent.top, 7)}
        PROJECTION
            "init=EPSG:{epsg_code}"
        END
    END
"""

    if writeHillshades:
        for hillshade in hillshade_files:
            # Open the single-band GeoTIFF
            with rasterio.open(os.path.join(single_band_tiff_dir, hillshade), 'r') as src:
                #data = src.read(1)  # Read the first band
                #origprofile = src.profile
                extent = src.bounds
                size = (src.width, src.height)
                epsg_code = src.crs.to_epsg() if src.crs else None
            mapfile_template = f"""{mapfile_template}
    LAYER
        NAME "{hillshade.split('.')[0]}"
        DATA "{hillshade}"
        TYPE RASTER
        STATUS ON
        EXTENT {np.round(extent.left, 7)} {np.round(extent.bottom, 7)} {np.round(extent.right, 7)} {np.round(extent.top, 7)}
        PROJECTION
            "init=EPSG:{epsg_code}"
        END
        COMPOSITE
            OPACITY 50
        END
    END
"""

    mapfile_template = f"""{mapfile_template}
END
    """

    # Write the Mapfile to a file
    print(mapfile_template)
    with open(mapfile_path, 'w') as mapfile:
        mapfile.write(mapfile_template)

    print(f"Mapfile '{mapfile_path}' generated successfully!")

######################################

def generate_tiff_based_mapfile(single_band_tiff_path, rgba_tiff_path, mapfile_path):
    '''
    funtion to take two geotiffs (single band and multiband) and generate a multi-layer mapsource mapfile
    single_band_tiff_path: path to single band geotiff
    rgba_tiff_path: path to multiband geotiff
    mapfile_path: path of output mapfile

    - single_band_tiff_path:
    
    - rgba_tiff_path:
    
    - mapfile_path
    '''
    # Open the single-band GeoTIFF
    with rasterio.open(single_band_tiff_path, 'r') as src:
        data = src.read(1)  # Read the first band
        origprofile = src.profile
        extent = src.bounds
        size = (src.width, src.height)
        epsg_code = src.crs.to_epsg() if src.crs else None
    
    # Define the paths to your GeoTIFF files
    single_band_file = single_band_tiff_path.split(os.path.sep)[-1]
    rgba_file        =        rgba_tiff_path.split(os.path.sep)[-1]
    name = single_band_file.split('.')[0]
    
    # Create a basic Mapfile template
    mapfile_template = f"""## Mapfile written on {today()} at {nowTime()}
MAP
    NAME "{name}"
    EXTENT {extent.left} {extent.bottom} {extent.right} {extent.top}
    SIZE {size[0]} {size[1]}
    # IMAGECOLOR 255 255 255 #white background - exclude for transparent background
    PROJECTION
        "init=EPSG:{epsg_code}"  # Set the EPSG code
    END
    
    LAYER
        NAME "SingleBandLayer"
        DATA "{single_band_file}"
        TYPE RASTER
        STATUS ON
        EXTENT {extent.left} {extent.bottom} {extent.right} {extent.top}
        PROJECTION
            "init=EPSG:{epsg_code}"
        END
    END

    LAYER
        NAME "RGBALayer"
        DATA "{rgba_file}"
        TYPE RASTER
        STATUS ON
        EXTENT {extent.left} {extent.bottom} {extent.right} {extent.top}
        PROJECTION
            "init=EPSG:{epsg_code}"
        END
    END
END
    """

    # Write the Mapfile to a file
    with open(mapfile_path, 'w') as mapfile:
        mapfile.write(mapfile_template)

    print(f"Mapfile '{mapfile_path}' generated successfully!")

######################################
