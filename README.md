# disco_tif
some general raster tools for visualization

The primary utility of this library is to take a single-channel geotiff and create an rgba (32bit) geotiff. 
The code can take a colormap name from matplotlib (https://matplotlib.org/stable/users/explain/colors/colormaps.html), however If no colormap is supplied the EMerald_standard_colormap will be applied. This is a data driven colormap.
Basically
