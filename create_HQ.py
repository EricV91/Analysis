import Metashape
import os,re,sys
import time
import logging
import shutil
import pandas as pd

#start with cleared console
Metashape.app.console.clear()

## construct the document class
doc = Metashape.app.document

rijweg1_path = r'E:\Metashape\c01_verdonk\Rijweg stalling 1'
rijweg_names = ['20190416_20190416-173256.psz', '20190515_20190516-022711.psx']

for name in rijweg_names:
    #start with cleared console
    Metashape.app.console.clear()

    ## construct the document class
    doc = Metashape.app.document

    #open file
    file = os.path.join(rijweg1_path, name)
    doc.open(file)

    #get chunk
    chunk = doc.chunks[0]

    #set output filenames
    DEM_out = os.path.join(r'E:\200 Projects\203 ACT_project\Rijweg_stalling1', 'Rijweg stalling 1-' + name[:8]+'-DEM.tif')
    Point_out = os.path.join(r'E:\200 Projects\203 ACT_project\Rijweg_stalling1', 'Rijweg stalling 1-' + name[:8]+'-points.laz')
    Ortho_out = os.path.join(r'E:\200 Projects\203 ACT_project\Rijweg_stalling1', 'Rijweg stalling 1-' + name[:8]+'.tif')

    #build output
    chunk.buildDepthMaps(quality=Metashape.MediumQuality, filter=Metashape.MildFiltering)
    chunk.buildDenseCloud(max_neighbors = 100, point_colors = False)
    chunk.buildDem(source=Metashape.DenseCloudData, interpolation=Metashape.EnabledInterpolation, projection=chunk.crs)
    chunk.buildOrthomosaic(surface=Metashape.ElevationData, blending=Metashape.MosaicBlending, projection=chunk.crs)

    #set shape path for rijweg1
    shape_path = r'E:\200 Projects\203 ACT_project\Rijweg_stalling1/rijweg_stalling1_AoI.shp'
    #import shape
    doc.importShapes(shape_path)
    #set shape as export boundary
    shape.boundary_type = Metashape.Shape.BoundaryType.OuterBoundary


    #export output
    chunk.exportDEM(path = DEM_out, tiff_big = True)
    chunk.exportPoints(path = Point_out, source = chunk.dense_cloud, crs = chunk.crs)
    chunk.exportOrthomosaic(path = ortho_out, tiff_big = True)

rolkas_path = r'E:\Metashape\c04_verdegaal\Achter de rolkas'
rolkas_names = ['20190420_20190423-161041.psx', '20190430_20190501-143715.psx', '20190510_20190511-094706.psx']

for name in rolkas_names:
    #start with cleared console
    Metashape.app.console.clear()

    ## construct the document class
    doc = Metashape.app.document

    #open file
    file = os.path.join(rijweg1_path, name)
    doc.open(file)

    #get chunk
    chunk = doc.chunks[0]

    #import shape
    #doc.importShapes(shape_path)

    #set output filenames
    DEM_out = os.path.join(r'E:\200 Projects\203 ACT_project\Rolkas', 'Achter de rolkas-' + name[:8]+'-DEM.tif')
    Point_out = os.path.join(r'E:\200 Projects\203 ACT_project\Rolkas', 'Achter de rolkas-' + name[:8]+'-points.laz')
    Ortho_out = os.path.join(r'E:\200 Projects\203 ACT_project\Rolkas', 'Achter de rolkas-' + name[:8]+'.tif')

    #build output
    #chunk.buildDepthMaps(quality=Metashape.MediumQuality, filter=Metashape.MildFiltering)
    #chunk.buildDenseCloud(max_neighbors = 100, point_colors = False)
    #chunk.buildDem(source=Metashape.DenseCloudData, interpolation=Metashape.EnabledInterpolation, projection=chunk.crs)
    #chunk.buildOrthomosaic(surface=Metashape.ElevationData, blending=Metashape.MosaicBlending, projection=chunk.crs)

    #set shape path for rijweg1
    shape_path = r'E:\200 Projects\203 ACT_project\Rolkas/Rolkas_AoI.shp'
    #import shape
    doc.importShapes(shape_path)
    #set shape as export boundary
    shape.boundary_type = Metashape.Shape.BoundaryType.OuterBoundary

    #export output
    chunk.exportDEM(path = DEM_out, tiff_big = True)
    chunk.exportPoints(path = Point_out, source = chunk.dense_cloud, crs = chunk.crs)
    chunk.exportOrthomosaic(path = ortho_out, tiff_big = True)

AZ74_path = r'E:\Metashape\c08_biobrass\AZ74'
AZ74_names = ['20190513_20190516-191222.psx', '20190517_20190520-172808.psx']

for name in rolkas_names:
    #start with cleared console
    Metashape.app.console.clear()

    ## construct the document class
    doc = Metashape.app.document

    #open file
    file = os.path.join(rijweg1_path, name)
    doc.open(file)

    #get chunk
    chunk = doc.chunks[0]

    #import shape
    #doc.importShapes(shape_path)

    #set shape path for rijweg1
    shape_path = r'E:\200 Projects\203 ACT_project\AZ74/AZ74_AoI.shp'
    #import shape
    doc.importShapes(shape_path)
    #set shape as export boundary
    shape.boundary_type = Metashape.Shape.BoundaryType.OuterBoundary

    #set output filenames
    DEM_out = os.path.join(r'E:\200 Projects\203 ACT_project\AZ74', 'AZ74-' + name[:8]+'-DEM.tif')
    Point_out = os.path.join(r'E:\200 Projects\203 ACT_project\AZ74', 'AZ74-' + name[:8]+'-points.laz')
    Ortho_out = os.path.join(r'E:\200 Projects\203 ACT_project\AZ74', 'AZ74-' + name[:8]+'.tif')

    #build output
    #chunk.buildDepthMaps(quality=Metashape.MediumQuality, filter=Metashape.MildFiltering)
    #chunk.buildDenseCloud(max_neighbors = 100, point_colors = False)
    #chunk.buildDem(source=Metashape.DenseCloudData, interpolation=Metashape.EnabledInterpolation, projection=chunk.crs)
    #chunk.buildOrthomosaic(surface=Metashape.ElevationData, blending=Metashape.MosaicBlending, projection=chunk.crs)

    #export output
    chunk.exportDEM(path = DEM_out, tiff_big = True)
    chunk.exportPoints(path = Point_out, source = chunk.dense_cloud, crs = chunk.crs)
    chunk.exportOrthomosaic(path = ortho_out, tiff_big = True)
