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
import rasterio
import numpy.ma as ma
import gdal
from shapely.geometry import Polygon, MultiPolygon
import shapely
import time
import os


Image.MAX_IMAGE_PIXELS = 30000000000   

input_path =  r'D:\400 Data analysis\420 Crop rows\Peen\Guido/20190609_resampeld.tif'
img_path =  r'D:\400 Data analysis\420 Crop rows\Peen\Guido/20190609_resampeld.tif'

def create_grid(gdf, angle):
    gdf_rd = gdf.to_crs(epsg=28992)
    
    xmin,ymin,xmax,ymax = gdf_rd.total_bounds
    
    #must be an int
    lenght = 5
    wide = 5
    
    
    cols = list(range(int(np.floor(xmin)), int(np.ceil(xmax)), wide))
    rows = list(range(int(np.floor(ymin)), int(np.ceil(ymax)), lenght))
    rows.reverse()
    
    polygons = []
    for x in cols:
        for y in rows:
            polygon = Polygon([(x,y), ((x+wide), y), ((x+wide), y-(lenght)), (x, y-lenght)])
            polygons.append(polygon)  
    
    #rotate
    polygon_group = MultiPolygon(polygons)
    polygon_group = shapely.affinity.rotate(polygon_group, angle)
    #create grid
    grid = gpd.GeoDataFrame({'geometry':polygon_group[:]})
    #optional, write grid to file
    #grid.to_file(r"C:\Users\ericv\Desktop\Technodag/GRID_rotated.shp")
    return grid

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

def overlay_grid_and_plants(gdf, grid, img_path):
    gdf_rd = gdf.to_crs(epsg=28992)
    #create new columns
    grid['id'] = range(len(grid))
    grid['area'] = np.nan
    #intersect data and grid
    intersect = gpd.overlay(grid, gdf_rd, how='intersection')
    #calculate area per grid cell and write area to column
    for ID in intersect.id:
        temp = intersect[intersect['id'] == ID]
        area = temp.geometry.area.sum()    
        grid.loc[ID, 'area'] = float(area)
    #write grid to file as output
    grid.to_file(os.path.dirname(img_path) + '/plant_area_grid.shp')
    return grid

def overlay_ridges_and_plants(ridges, plants):
    intersect = gpd.overlay(plants, ridges, how='intersection')
    return intersect

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

#optional blockwise processing
"""
xsize = image.shape[1]
ysize = image.shape[0]
#define size of img blocks
x_block_size = 256
y_block_size = 256
#create iterator to process blocks of imagery one by one. #get 10% subset 
it = list(range(0,10000, 1))

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
"""
img = image.copy()
gdf = vectorize_image(img, img_path)
ridges = gdf.copy()
plants = gpd.read_file(os.path.dirname(input_path) + '/ridges_valid.shp')
intersect = overlay_ridges_and_plants(ridges, plants)

test = plants.simplify(tolerance = 0.0000005)

rdf.to_file(os.path.dirname(input_path) + '/simplify.shp')


gdf.to_file(os.path.dirname(img_path) + '/weeds_clip_resampled.shp')
gdf.to_file(os.path.dirname(img_path) + '/0625_plant_area.shp')

angle = 0 # counter-clockwise! set crop row angle         
grid = create_grid(gdf, angle)
tic = time.time()
overlay_grid_and_plants(gdf, grid, img_path)
toc = time.time()
print('overlay took ' + str(toc-tic) + ' seconds')



