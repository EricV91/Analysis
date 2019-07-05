# -*- coding: utf-8 -*-
"""
Created on Fri Jun 21 16:59:28 2019

@author: VanBoven
"""

import os
import numpy as np

import cv2

import geopandas as gpd
import rasterio 
from shapely.geometry import mapping
from rasterio.mask import mask
import rasterio

from sklearn import preprocessing

import gdal

    
def remove_outliers_DEM(x):
    x = x[0,:,:]
    x[x < -5] = np.nan
    x[x > 15] = np.nan
    x_masked = np.ma.masked_invalid(x) 
    return x_masked
    
def reproject_raster_and_resample(input_path, output_path, scaling_factor=1, destination_crs='EPSG:28992'):
    """Projects a GeoTIFF to another coordinate system
    Parameters
    ----------
        input_path(str): relative path to the input geotiff
        output_path(str): path where the reprojected geotiff should be saved
        scaling_factor(int/flt): rescaling of the raster size, bigger scaling_factor
                                 makes the resolution lower.
        destination_crs(str): epsg code of the coordinate system the raster is reprojected to
    Returns
    -------
        reprojected and resampled raster saved to disk
    """

    with rasterio.open(input_path) as src:
        transform, width, height = rasterio.warp.calculate_default_transform(
            src.crs, destination_crs, src.width//scaling_factor, src.height//scaling_factor, *src.bounds)
        kwargs = src.meta.copy()
        kwargs.update({
            'crs': destination_crs,
            'transform': transform,
            'width': width,
            'height': height
        })

        with rasterio.open(output_path, 'w', **kwargs) as dst:
            rasterio.warp.reproject(
                source=rasterio.band(src, 1),
                destination=rasterio.band(dst, 1),
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=transform,
                dst_crs=destination_crs,
                resampling=rasterio.warp.Resampling.cubic)


#define variables
output_path = r'D:\700 Georeferencing\NZ66_2\Temp_results' 
img_path = r'D:\700 Georeferencing\NZ66_2\DEM/20190607_DEM_clipped.tif'
src_path = r'D:\700 Georeferencing\NZ66_2\DEM/20190607_DEM_clipped.tif'

out_path = output_path + '/test_resampeld.tif'
#reproject_raster_and_resample(img_path, out_path, scaling_factor=4, destination_crs='EPSG:4326')

src = rasterio.open(img_path)

ds = gdal.Open(img_path)
band = ds.GetRasterBand(1)
xsize = band.XSize
ysize = band.YSize

#read bands as array
dem = ds.GetRasterBand(1).ReadAsArray()
dem[dem < 0] = 0
dem[dem > 10] = 0

kernel = np.ones((8,1),np.float32)/8
#filter dem
filtered_dem = cv2.filter2D(dem,-1,kernel)
for i in range(2):
    filtered_dem = cv2.filter2D(filtered_dem,-1,kernel)

#determine local mean
kernel = np.ones((5,5),np.float32)/25
smooth = cv2.filter2D(dem,-1,kernel)

ridges = filtered_dem-smooth
#ridges = dem - filtered_dem
ridges[ridges > 0] = 255
#ridges[ridges < 0] = 0
#ridges = (ridges/ridges.max()) * 255
ridges = ridges.astype(np.uint8)

kernel = np.ones((3,3))
#ridges = cv2.erode(ridges,kernel,iterations = 2)
#ridges = cv2.morphologyEx(ridges, cv2.MORPH_CLOSE, kernel)

# Register GDAL format drivers and configuration options with a
# context manager.
with rasterio.Env():

    # Write an array as a raster band to a new 8-bit file. For
    # the new file's profile, we start with the profile of the source
    profile = src.profile

    # And then change the band count to 1, set the
    # dtype to uint8, and specify LZW compression.
    profile.update(
        dtype=rasterio.int8,
        count=1,
        compress='lzw',
        nodata = 0)

    with rasterio.open(output_path+'/crop_top.tif', 'w', **profile) as dst:
        dst.write(ridges.astype(rasterio.int8), 1)

# At the end of the ``with rasterio.Env()`` block, context
# manager exits and all drivers are de-registered.
