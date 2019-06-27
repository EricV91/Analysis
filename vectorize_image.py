# -*- coding: utf-8 -*-

"""

Spyder Editor



This is a temporary script file.

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

Image.MAX_IMAGE_PIXELS = 30000000000   

img_path = r'C:\Users\ericv\Desktop\Technodag/c08_biobrass-C49-201905241802.tif'

template = np.array(Image.open(r'C:\Users\ericv\Desktop\Technodag/c08_biobrass-C49-201905241802.jpg'), dtype = np.uint8)[11000:16000,12000:12200]

def vectorize_image(template, img_path):
    #get transform
    src = rasterio.open(img_path)

    #create mask
    mask = ma.masked_values(template, 1)    
    
    #vectorize
    results = (
        {'properties': {'raster_val': v}, 'geometry': s}
        for i, (s, v) 
        in enumerate(
            shapes(template, mask=mask, connectivity=8, transform=src.transform)))
            
    
    geoms = list(results)
    geom_list = [shape(geom['geometry']) for geom in geoms]
    value_list = [geom['properties']['raster_val'] for geom in geoms]

    gdf = gpd.GeoDataFrame({'value':value_list}, geometry = geom_list, crs = {'init' :'epsg:4326'})
    return gdf

def overlay_grid_and_plants(gdf, grid):
    gdf_rd = gdf.to_crs(epsg=28992)
    #create new columns
    grid['id'] = range(len(grid))
    grid['area'] = np.nan
    #intersect data and grid
    intersect = gpd.overlay(grid, points_rd, how='intersection')
    #calculate area per grid cell and write area to column
    for ID in intersect.id:
        temp = intersect[intersect['id'] == ID]
        area = temp.geometry.area.sum()    
        grid.loc[ID, 'area'] = float(area)
    #write grid to file as output
    grid.to_file('/plant_area_grid.shp')

