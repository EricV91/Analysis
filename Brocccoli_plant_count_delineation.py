# -*- coding: utf-8 -*-
"""
Created on Fri May 24 10:31:20 2019

@author: VanBoven
"""

from tensorflow.keras import models

import os

import pandas as pd

import time
import cv2
import sklearn
import numpy as np
import numpy.ma as ma
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.cluster import SpectralClustering
from sklearn.cluster import MiniBatchKMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn import preprocessing

from sklearn.datasets import make_blobs

from PIL import Image

#import rasterio
import gdal
from osgeo import gdalnumeric
from osgeo import ogr, osr
from fiona.crs import from_epsg
import fiona
import geopandas as gpd
import rasterio 
from rasterio.features import shapes
from shapely.geometry import mapping, Polygon, shape, Point

from functools import partial
from shapely.ops import transform
import pyproj
    
from skimage.util.shape import view_as_blocks

#resize input images
def resize(x):
    new_shape = (50,50,3)
    x_resized = np.array(Image.fromarray(x).resize((50,50)))
    #X_train_new = scipy.misc.imresize(x, new_shape)
    x_rescaled = x_resized/255
    return x_rescaled

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
    df['coords'] = df.centroid.apply(lambda x:rasterio.transform.xy(transform = src.transform, rows = x[0], cols = x[1], offset='ul'))
    df['geom'] = df.contours.apply(lambda x:rasterio.transform.xy(transform = src.transform, rows = list(x[:,0,1]), cols = list(x[:,0,0]), offset='ul'))
    #convert df to gdf
    #for polygon, first reformat into lists of coordinate pairs
    shape_list = []
    for geom in df.geom:
        x_list = geom[0]
        y_list = geom[1]
        coords_list = []
        for i in range(len(x_list)):
            x = x_list[i]
            y = y_list[i]
            coords_list.append([x, y])
        shape_list.append(coords_list)
    df['geom2'] = shape_list

    #create points
    gdf_point = gpd.GeoDataFrame(df, geometry = [Point(x, y) for x, y in df.coords], crs = {'init': 'epsg:4326'})
    gdf_point = gdf_point.drop(['contours', 'moment', 'cx', 'cy', 'bbox', 'output', 'input',
       'centroid', 'coords', 'geom', 'geom2'], axis=1)    
    #create polygons
    gdf_poly = gpd.GeoDataFrame(df, geometry = [Polygon(shape) for shape in df.geom2], crs = {'init': 'epsg:4326'}) 
    gdf_poly = gdf_poly.drop(['contours', 'moment', 'cx', 'cy', 'bbox', 'output', 'input',
       'centroid', 'coords', 'geom', 'geom2'], axis=1)
    
    gdf_point.to_file(r'E:\400 Data analysis\410 Plant count\c01_verdonk\Rijweg stalling 2/AZ74.shp')
    gdf_poly.to_file(r'E:\400 Data analysis\410 Plant count\c01_verdonk\Rijweg stalling 2/AZ74_shape.shp')    

    return

gpdf = gpd.read_file(r'E:\400 Data analysis\410 Plant count\test_results/c07_hollandbean-Hendrik de Heer-201905131422_clipped_poly2.shp')

def multi2single(gpdf):
    gpdf = gpdf.drop(['area'], axis = 1)
    gpdf.geometry = gpdf.buffer(0)
    gpdf_dissolved = gpdf.dissolve('prediction')
    gpdf_singlepoly = gpdf_dissolved[gpdf_dissolved.geometry.type == 'Polygon']
    gpdf_multipoly = gpdf_dissolved[gpdf_dissolved.geometry.type == 'MultiPolygon']

    for i, row in gpdf_multipoly.iterrows():
        Series_geometries = pd.Series(row.geometry)
        df = pd.concat([gpd.GeoDataFrame(row, crs=gpdf_multipoly.crs).T]*len(Series_geometries), ignore_index=True)
        df['geometry']  = Series_geometries
        gpdf_singlepoly = pd.concat([gpdf_singlepoly, df])

    gpdf_singlepoly.reset_index(inplace=True, drop=True)
    return gpdf_singlepoly

gpdf_multipoly.to_file(r'E:\400 Data analysis\410 Plant count\test_results/Hendrik-multipoly.shp')

#initialize cluster centres for kmeans clustering
green_lab = cv2.cvtColor(np.array([[[100,190,150]]]).astype(np.uint8), cv2.COLOR_BGR2LAB) # as sampled from ortho's file
green_init = np.array(green_lab[0,0,1:3])
background_init = cv2.cvtColor(np.array([[[215,198,190]]]).astype(np.uint8), cv2.COLOR_BGR2LAB) # as sampled from ortho's file
background_init = np.array(background_init[0,0,1:3])
#cluster centres
kmeans_init = np.array([background_init, green_init])

#load model for object classification
model = models.load_model(r'C:\Users\VanBoven\Documents\GitHub\DataAnalysis/Broccoli_model1.h5')

# set limits to object size
min_plant_size = 16
max_plant_size = 1600

#use small block size to cluster based on colors in local neighbourhood
x_block_size = 256  
y_block_size = 256

#input img_path
img_path = r'E:\VanBovenDrive\VanBoven MT\Archive\c08_biobrass\AZ74\20190513\1357\Orthomosaic/c08_biobrass-AZ74-201905131357.tif'
#output file directory
out_path = r''

#create iterator to process blocks of imagery one by one. 
it = list(range(0,400000, 1))
#skip = True if you do not want to process each block but you want to process the entire image
process_full_image = True

# Function to read the raster as arrays for the chosen block size.
def cluster_objects(x_block_size, y_block_size, it, img_path, out_path):    
    #time process
    tic = time.time()

    #srcArray = gdalnumeric.LoadFile(raster)
    ds = gdal.Open(img_path)
    band = ds.GetRasterBand(1)
    xsize = band.XSize
    ysize = band.YSize
    
    #create template for img mask resulting from clustering algorithm 
    template = np.zeros([ysize, xsize], np.uint8)

    #define kernel for morhpological closing operation
    kernel = np.ones((7,7), dtype='uint8')
    
    #iterate through img using blocks
    blocks = 0
    for y in range(0, ysize, y_block_size):
        if y > 0:
            y = y - 30 # use -30 pixels overlap to prevent "lines at the edges of blocks in object detection"
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
                #read bands as array
                b = np.array(ds.GetRasterBand(1).ReadAsArray(x, y, cols, rows), dtype = np.uint(8))
                g = np.array(ds.GetRasterBand(2).ReadAsArray(x, y, cols, rows), dtype = np.uint(8))
                r = np.array(ds.GetRasterBand(3).ReadAsArray(x, y, cols, rows), dtype = np.uint(8))
                img = np.zeros([b.shape[0],b.shape[1],3], np.uint8)
                img[:,:,0] = b
                img[:,:,1] = g
                img[:,:,2] = r
                
                #check if block of img has values
                if img.mean() > 0:
                                     
                    #convert to CieLAB colorspace
                    img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
                    #get a and b bands
                    a = np.array(img_lab[:,:,1])
                    b2 = np.array(img_lab[:,:,2])
                    
                    #flatten the bands for kmeans clustering algorithm
                    a_flat = a.flatten()
                    b2_flat = b2.flatten()
                    
                    #scale values to 0-1
                    scaler = preprocessing.MinMaxScaler(feature_range=(0, 255))
                    a_flat = scaler.fit_transform(a_flat)
                    b2_flat = scaler.fit_transform(b2_flat)
                    
                    #stack bands to create final input for clustering algorithm
                    Classificatie_Lab = np.ma.column_stack((a_flat, b2_flat))
                    
                    #perform kmeans clustering
                    kmeans = KMeans(init = kmeans_init, n_jobs = -1, max_iter = 25, n_clusters = 2, verbose = 0)
                    kmeans.fit(Classificatie_Lab)
                    
                    #do something with resulting cluster centres
                    
                    #cluster image block
                    y_kmeans = kmeans.predict(Classificatie_Lab)
                    
                    #Get plants
                    y_kmeans[y_kmeans == 0] = 0
                    y_kmeans[y_kmeans == 1] = 1

                    #convert binary output back to 8bit image                
                    kmeans_img = y_kmeans                
                    kmeans_img = kmeans_img.reshape(img.shape[0:2]).astype(np.uint8)
                    binary_img = kmeans_img * 125
                    
                    #close detected shapes
                    closing = cv2.morphologyEx(binary_img, cv2.MORPH_CLOSE, kernel)
                    
                    #write img block result back on original sized image template         
                    template[y:y+rows, x:x+cols] = template[y:y+rows, x:x+cols] + closing
                    #print('processing of block ' + str(blocks) + ' finished')
    
    #write of file as jpg img to store results if something goes wrong                
    cv2.imwrite(os.path.join(outpath, 'plant_mask', 'AZ74_blocks_256_'+str(i)+'.jpg',template))     
    toc = time.time()
    print("processing of blocks took "+ str(toc - tic)+" seconds")
    return template
    
    
        if process_full_image == True:

        print('Start with classification of objects')
        #initiate output img
        output = np.zeros([ysize,xsize,3], np.uint8)
        b = np.array(ds.GetRasterBand(1).ReadAsArray()).astype(np.uint(8))
        output[:,:,0] = b
        b = None
        g = np.array(ds.GetRasterBand(2).ReadAsArray()).astype(np.uint(8))
        output[:,:,1] = g
        g = None
        r = np.array(ds.GetRasterBand(3).ReadAsArray()).astype(np.uint(8))
        output[:,:,2] = r
        r = None
        
        result_img = np.zeros((template.shape[0],template.shape[1]),dtype=np.uint8)
        
        #Get contours of features
        contours, hierarchy = cv2.findContours(template, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        #create df with relevant data
        df = pd.DataFrame({'contours': contours})
        df['area'] = df.contours.apply(lambda x:cv2.contourArea(x)) 
        df = df[(df['area'] > 81) & (df['area'] < 500)]
        df['moment'] = df.contours.apply(lambda x:cv2.moments(x))
        df['centroid'] = df.moment.apply(lambda x:(int(x['m01']/x['m00']),int(x['m10']/x['m00'])))
        df['cx'] = df.moment.apply(lambda x:int(x['m10']/x['m00']))
        df['cy'] = df.moment.apply(lambda x:int(x['m01']/x['m00']))
        df['bbox'] = df.contours.apply(lambda x:cv2.boundingRect(x))
        #create input images for model
        df['output'] = df.bbox.apply(lambda x:output[x[1]-5: x[1]+x[3]+5, x[0]-5:x[0]+x[2]+5])
        df = df[df.output.apply(lambda x:x.shape[0]*x.shape[1]) > 0]
        #resize data to create input tensor for model
        df['input'] = df.output.apply(lambda x:resize(x))
        
        #remove img from memory
        output = None
        
        #resize images
        #df.input.apply(lambda x:x.resize(50,50,3, refcheck=False))       
        model_input = np.asarray(list(df.input.iloc[:]))
        
        #predict
        tic = time.time()
        prediction = model.predict(model_input)
        #get prediction result
        pred_final = prediction.argmax(axis=1)
        #add to df
        df['prediction'] = pred_final
        toc = time.time()
        print('classification of '+str(len(df))+' objects took '+str(toc - tic) + ' seconds')
        
        write_plants2shp(img_path, df, shp_dir, shp_name)








