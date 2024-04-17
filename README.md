# disco_tif
some general raster tools for visualization

The primary utility of this library is to take a single-channel geotiff and create an rgba (32bit) geotiff. 
The code can take a colormap name from matplotlib (https://matplotlib.org/stable/users/explain/colors/colormaps.html), however If no colormap is supplied the EMerald_standard_colormap will be applied. This is a data driven colormap.

Basically, the EMerald_standard_colormap consists of 8 colors, as seen below.
![image](https://github.com/emerald-geomodelling/disco-tif/assets/15694972/578de983-d28b-49be-8c0f-49e2f4d92817)
The dark blue color is reserved for negative data values. If the data range supplied does not have any negative data then the minimum color will start at cyan.
Essentially, we have three cases for the minimum value of the data range:
  1) The minimum value of the data range is less than 0:
     - Here negative data will be mapped lineraly from the min value to 0, which will be cyan
  2) The minimum value of the data range is 0:
     - Here the dark blue will be excluded and cyan will be mapped to 0
  3) The minimum value of the data range is greater than 0:
     - Here cyan will be mapped to the minimum value

The rest of the colormap can be created in one of two ways: either 'pseudo-hist-norm' or 'pseudo-linear'.
If 'pseudo-hist-norm' is specified, the positive region of the data range will be normalized by the histogram of the data - i.e. every colorbin will have roughly equal counts. The plot below is for the example data found in the in the 'example_data' folder of this library. You will need to download the original geotiff. Here we are using a min-max datarange = [-407, 5825]
![image](https://github.com/emerald-geomodelling/disco-tif/assets/15694972/19873800-9da2-4c22-9060-673481887003)
  yeliding a colormap that looks like this:
![image](https://github.com/emerald-geomodelling/disco-tif/assets/15694972/bfb1483f-8346-4a3d-a1a1-73557136634b)

if 'pseudo-linear' is specified, the positive region of the data range will be linearly mapped to the colormap. This will yeild a mapping like
![image](https://github.com/emerald-geomodelling/disco-tif/assets/15694972/3b3e4451-dd3f-47ef-9950-93ad2d8a66f9)
  yeliding a colormap that looks like this:
![image](https://github.com/emerald-geomodelling/disco-tif/assets/15694972/9f8388c7-582d-486c-9aec-f48f87002803)

Both of these colormaps are valid and the end user should evaluate which fits their needs better.

In building the colormaps a user can specify data min-max values, but if no data range is specified the program will choose min-max values based on the 1st and 99th percentile (quartile) of the data values.

In addition to generating new colorized geotiffs this library has the ability to generate hillshades. This is just a wrapper around earthpy. For this you can specify a sun_azimuth angle and an sun_altitude angle to generate the hill shade. You can also pass in an array of sun_azimuth angles and/or and array of sun_altitude angles. This program will generate a new hillshade for each of the combiniations of azimuths and altitudes.
One unique feature that is built into this library is the ability to do priciple-component-analysis on a series of hillshades. Currently the first 3 components of the PCA are saved and output. I find that this really helps highlight prominent features in a geotiff, however, if the source raster is of low-quality or generated from different sources you may also notice gridding and/or processing artifacts. For the example data in the geotiff you will imeadieatly notice these gridding and processing artifacts. Currently there is a limitation of the number of hillshades allowed to be used in the pca, but this is arbitrary and is only a function of computer resources. On my M1 MacBook Pro I have successfully used 32 hillshades for a PCA on this example datset which is only 30mb, but beware that larger rasters or higher resolution rasters will be a lot heavier and could cause your kernel to crash or worse. 

Please see the example notebooks in the 'example_notbooks' folder for function calls and how to use them. [This notebook](/example_notebooks/process_tiff_from_functions.ipynb) plots the output of each function. [This notebook](/example_notebooks/streamlined_example_usage.ipynb) demonstrates how to use these tools for automation.

Lastly, If you are using a mapsource mapfile this program will write the file for you. This is the least tested part of this library, so please fork, edit, and make pull-requests as you use this library! 

Thanks for reading and happy disco_tif-ing 
