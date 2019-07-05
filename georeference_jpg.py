# -*- coding: utf-8 -*-
"""
Created on Wed Jul  3 08:51:45 2019

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
from PIL import Image

import gdal

Image.MAX_IMAGE_PIXELS = 30000000000   

img_path = r'D:\700 Georeferencing\Guido Cortvriendt\temp_results/c03_termote-Guido Cortvriendt-201906251146_clipped.jpg'
src_path = r'D:\700 Georeferencing\Guido Cortvriendt\Results/c03_termote-Guido Cortvriendt-201906251146_clipped.tif'

out_path = r'D:\700 Georeferencing\Guido Cortvriendt\Results/20190625_plants.tif'

image = np.array(Image.open(img_path), dtype = np.uint8)
image[image > 0] = 255
kernel = np.ones((9,9))
closing = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
image = closing.copy()

src = rasterio.open(src_path)

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

    with rasterio.open(out_path, 'w', **profile) as dst:
        dst.write(image.astype(rasterio.int8), 1)

