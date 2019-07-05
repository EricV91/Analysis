# -*- coding: utf-8 -*-
"""
Created on Wed Jul  3 08:08:40 2019

@author: VanBoven
"""
import rasterio
from rasterio.features import shapes
import cv2
from PIL import Image
import numpy as np
import numpy.ma as ma
import geopandas as gpd
from shapely.geometry import shape
import pandas as pd
import rasterio
import numpy.ma as ma
import gdal
from shapely.geometry import Polygon, MultiPolygon
import shapely
import time
import os

def vectorize_image(template, img_path):
    #get transform
    src = rasterio.open(img_path)
    #create mask
    mask = ma.masked_values(template, 0)    
    #vectorize
    results = (
        {'properties': {'raster_val': v}, 'geometry': s}
        for i, (s, v) 
        in enumerate(
            shapes(template, mask=mask, connectivity=4, transform=src.transform)))
            
    geoms = list(results)
    geom_list = [shape(geom['geometry']) for geom in geoms]
    value_list = [geom['properties']['raster_val'] for geom in geoms]

    gdf = gpd.GeoDataFrame({'value':value_list}, geometry = geom_list, crs = {'init' :'epsg:4326'})
    return gdf

Image.MAX_IMAGE_PIXELS = 30000000000   

input_path =  r'D:\700 Georeferencing\Guido Cortvriendt\Results/190609_ridges_clipped.tif'
img_path = r'D:\700 Georeferencing\Guido Cortvriendt\Results/190609_ridges_clipped.tif'

input_path =  r'D:\700 Georeferencing\Guido Cortvriendt\temp_results/c03_termote-Guido Cortvriendt-201906091405_clipped.jpg'
img_path =  r'D:\700 Georeferencing\Guido Cortvriendt\Results/c03_termote-Guido Cortvriendt-201906091405_clipped.tif'

input_path =  r'D:\700 Georeferencing\Guido Cortvriendt\temp_results/c03_termote-Guido Cortvriendt-201906181831_clipped.jpg'
img_path =  r'D:\700 Georeferencing\Guido Cortvriendt\Results/c03_termote-Guido Cortvriendt-201906181831_clipped.tif'

input_path =  r'D:\700 Georeferencing\Guido Cortvriendt\temp_results/c03_termote-Guido Cortvriendt-201906251146_clipped.jpg'
img_path =  r'D:\700 Georeferencing\Guido Cortvriendt\Results/c03_termote-Guido Cortvriendt-201906251146_clipped.tif'

output_path = r'D:\700 Georeferencing\Guido Cortvriendt\Results\vector_data'

if input_path.endswith('.tif'):
    ds = gdal.Open(input_path)
    ridges = ds.GetRasterBand(1).ReadAsArray()
    ridges[ridges < 0] = 0
    ridges[ridges > 0] = 255
    image = ridges.astype(np.uint8)
else:
    image = np.array(Image.open(input_path), dtype = np.uint8)
    image[image > 0] = 255
    #optional morphologic operations
    kernel = np.ones((9,9))
    closing = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
    image = closing.copy()

xsize = image.shape[1]
ysize = image.shape[0]
#define size of img blocks
x_block_size = 256
y_block_size = 256
#create iterator to process blocks of imagery one by one. #get 10% subset 
it = list(range(0,100000, 1))

#iterate through img using blocks
blocks = 0
gdf_list = []
for y in range(0, ysize, y_block_size):
    if y > 0:
        y = y 
    if y + y_block_size < ysize:
        rows = y_block_size
    else:
        rows = ysize - y
    for x in range(0, xsize, x_block_size):
        if x > 0:
            x = x 
        blocks += 1
        #if statement for subset
        if blocks in it:
            if x + x_block_size < xsize:
                cols = x_block_size
            else:
                cols = xsize - x
            img = image[y:(y+rows), x:(x+cols)]
            template = np.zeros((image.shape[0], image.shape[1]), dtype = np.uint8)
            template[y:y+rows, x:x+cols] = template[y:y+rows, x:x+cols] + img
            tic = time.time()
            gdf = vectorize_image(template, img_path)
            toc = time.time()
            print('vectorizing took ' + str(toc-tic) + ' seconds')
            gdf_list.append(gdf)

rdf = gpd.GeoDataFrame(pd.concat(gdf_list, ignore_index=True), crs=gdf_list[0].crs)   
  
rdf.to_file(os.path.join(output_path, os.path.basename(input_path)[:-4]+'_vector.shp'))
  

