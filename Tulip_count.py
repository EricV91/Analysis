# -*- coding: utf-8 -*-
"""
Created on Mon Jun  3 13:46:32 2019

@author: VanBoven
"""

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import pandas as pd
import cv2
import scipy
from sklearn.neighbors import NearestNeighbors
import gdal
import pandas as pd
import time

from fiona.crs import from_epsg
import fiona
import geopandas as gpd
import rasterio 
from rasterio.features import shapes
from shapely.geometry import mapping, Polygon, shape, Point

from functools import partial
from shapely.ops import transform
import pyproj
    

#max number of pixels in image is restricted, in order to open big orthos it has to be modified
Image.MAX_IMAGE_PIXELS = 3000000000      

template = np.array(Image.open(r'E:\400 Data analysis\410 Plant count\test_results/mask_c04_verdegaal-Achter_de_rolkas-20190416_full.png'), dtype = np.uint8)
zeros = np.zeros((template.shape[0], template.shape[1]), dtype = np.uint8)



#i = 0
#srcArray = gdalnumeric.LoadFile(raster)
ds = gdal.Open(img_path)
band = ds.GetRasterBand(1)
xsize = band.XSize
ysize = band.YSize
   
#template = np.zeros([ysize, xsize], np.uint8)
#plant_contours = np.zeros([ysize, xsize], np.uint8)

img_path = r'E:\VanBovenDrive\VanBoven MT\Archive\c04_verdegaal\Achter de rolkas\20190416\Orthomosaic/c04_verdegaal-Achter de rolkas-20190416.tif'

x_block_size = 50000
y_block_size = 50000

blocks = 0

ds = gdal.Open(img_path)
band = ds.GetRasterBand(1)
xsize = band.XSize
ysize = band.YSize
  
it = list(range(0,1000, 1))
  
for y in range(0, ysize, y_block_size):
    if y > 0:
        y = y - 30
    if y + y_block_size < ysize:
        rows = y_block_size
    else:
        rows = ysize - y
    for x in range(0, xsize, x_block_size):
        if x > 0:
            x = x - 30
        blocks += 1
        #if statement for subset
        if blocks in it:
            if x + x_block_size < xsize:
                cols = x_block_size
            else:
                cols = xsize - x

            b = np.array(ds.GetRasterBand(1).ReadAsArray(x, y, cols, rows), dtype = np.uint(8))
            g = np.array(ds.GetRasterBand(2).ReadAsArray(x, y, cols, rows), dtype = np.uint(8))
            r = np.array(ds.GetRasterBand(3).ReadAsArray(x, y, cols, rows), dtype = np.uint(8))
            img = np.zeros([b.shape[0],b.shape[1],3], np.uint8)
            img[:,:,0] = r
            img[:,:,1] = g
            img[:,:,2] = b

b = None
g = None
r = None

red_threshold = 150
blue_treshold = 100
result = pd.DataFrame()
it = 100
for j in range(it):
    tic = time.time()
    edges = np.copy(template[int((j/it) * template.shape[0]):int(((j+1)/it) * template.shape[0]),:])
    df = pd.DataFrame()

#edges = template[28000:33000, 28000:33000]
#output= np.zeros((test.shape[0], test.shape[1], 3), dtype = np.uint8)

#kernel = np.ones((3,3),np.uint8)
#edges = cv2.erode(edges,kernel,iterations = 1)

#edges = cv2.Canny(test, 10, 20)

#cv2.imwrite(r'E:\400 Data analysis\410 Plant count\test_results/temp.png',edges)

    circles = cv2.HoughCircles(edges, cv2.HOUGH_GRADIENT, dp = 1, minDist = 6, param1 = 100, param2 = 4, minRadius = 4, maxRadius = 8)
    
    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        circles[:, 1] = circles[:, 1] + int((j/it) * template.shape[0])
        centroids = (circles[:,0], circles[:,1])
        df['x'] = circles[:, 0]      
        df['y'] = circles[:, 1]
        df['centroid'] = [(centroids[0][i],centroids[1][i]) for i in range(len(circles))]  
        df['r'] = circles[:, 2]
        df['j'] = j
    
        result = result.append(df)
        
    toc = time.time()
    print("processing of block " + str(j) + " took "+ str(toc - tic)+" seconds")
      
    result['red'] = result.centroid.apply(lambda x: img[x[1]-3:x[1]+5, x[0]-5:x[0]+3,0].mean())
    result['blue'] = result.centroid.apply(lambda x: img[x[1]-3:x[1]+5, x[0]-5:x[0]+3,2].mean())
    result['a'] = result.centroid.apply(lambda x: cv2.cvtColor(img[x[1]-3:x[1]+5, x[0]-5:x[0]+3,:], cv2.COLOR_BGR2LAB)[:,:,1].mean())
    result['b'] = result.centroid.apply(lambda x: cv2.cvtColor(img[x[1]-3:x[1]+5, x[0]-5:x[0]+3,:], cv2.COLOR_BGR2LAB)[:,:,2].mean())
   
    result2 = result[(result['a'] - result['b']) > 5] #red_threshold) &( result['blue'] < blue_treshold)]
    
    for k in range(len(result2)):
        x = result2.x.iloc[k]
        y = result2.y.iloc[k]
        r = result2.r.iloc[k]
        cv2.circle(img, (x, y), r, (255, 0, 0), 1)
    
cv2.imwrite(r'E:\400 Data analysis\410 Plant count\test_results/Henk_test3.jpg',img)
   
    
def points_in_array(x):
    for point in x:
        point_list = point[0]
        return cx, cy


def transform_geometry(geometry):
    project = partial(
    pyproj.transform,
    pyproj.Proj(init='epsg:4326'), # source coordinate system
    pyproj.Proj(init='epsg:28992')) # destination coordinate system
    geometry = transform(project, geometry)  # apply projection
    return geometry

def write_plants2shp(img_path, df, shp_dir, shp_name):
    #get transform
    src = rasterio.open(img_path)
    #convert centroids to coords and contours to shape in lat, lon
    df['coords'] = df.centroid.apply(lambda x:rasterio.transform.xy(transform = src.transform, rows = x[1], cols = x[0], offset='ul'))
    #df['geom'] = df.contours.apply(lambda x:rasterio.transform.xy(transform = src.transform, rows = list(x[:,0,1]), cols = list(x[:,0,0]), offset='ul'))

    #create points
    gdf_point = gpd.GeoDataFrame(df, geometry = [Point(x, y) for x, y in df.coords], crs = {'init': 'epsg:4326'})
    gdf_point = gdf_point.drop(['j', 'x', 'y', 'cielab', 'centroid', 'coords'], axis=1)    
    
    gdf_point.to_file(r'F:\400 Data analysis\410 Plant count\c03_termote\Binnendijk_links/cichorei_detection.shp')

    return

    
    
    
    
    output = np.copy(img[28000:33000, 28000:33000, :])
    # ensure at least some circles were found
    	# convert the (x, y) coordinates and radius of the circles to integers
    	circles = np.round(circles[0, :]).astype("int")
    	# loop over the (x, y) coordinates and radius of the circles
    	for (x, y, r) in circles:
    		# draw the circle in the output image, then draw a rectangle
    		# corresponding to the center of the circle
    		cv2.circle(output, (x, y), r, (255, 0, 0), 1)
    		#cv2.rectangle(output, (x - 5, y - 5), (x + 5, y + 5), (0, 128, 255), -1)
    
    cv2.imwrite(r'E:\400 Data analysis\410 Plant count\test_results/temp_circle_img.jpg',output)
    
