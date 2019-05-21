# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

    #get transform
    src = rasterio.open(img_path)
    #create mask
    mask = ma.masked_values(plant_contours, 0)    

    #vectorize
    results = (
        {'properties': {'raster_val': v}, 'geometry': s}
        for i, (s, v) 
        in enumerate(
            shapes(plant_contours, mask=mask, connectivity=8, transform=src.transform)))
    geoms = list(results)     
    #vectorize raster
    
    # Define a polygon feature geometry with one attribute
    schema = {
        'geometry': 'Polygon',
        'properties': {'LAI': 'float:10.5',
                       'Height': 'int',
                       'Diameter':'float:10.5'},
    }
    schema2 = {
        'geometry': 'Point',
        'properties': {#'LAI': 'float:10.5',
                       #'Height': 'int',
                       'id':'int'},
    }
    
    #create output filename
    outfile = os.path.join(shp_dir, shp_name)
    #outfile = outfile[:-4] + ".shp"
    
    # Write a new Shapefile
    with fiona.open(str(outfile), 'w', 'ESRI Shapefile', schema2, crs = from_epsg(4326)) as c:
        ## If there are multiple geometries, put the "for" loop here
        for i in range(len(geoms)):
            geom = shape(geoms[i]['geometry'])
            coords = geom.centroid
            #geometry = transform_geometry(geom)
            #bounds = geometry.bounds
            c.write({
                'geometry': mapping(coords),
                'properties': {'id':i},
            })    
    
    with fiona.open(str(outfile), 'w', 'ESRI Shapefile', schema, crs = from_epsg(4326)) as c:
        ## If there are multiple geometries, put the "for" loop here
        for i in range(len(geoms)):
            geom = shape(geoms[i]['geometry'])
            geometry = transform_geometry(geom)
            bounds = geometry.bounds
            c.write({
                'geometry': mapping(geom),
                'properties': {'LAI': geometry.area, 
                               'Height': 0,
                               'Diameter': float(np.max([abs(bounds[2]-bounds[0]),abs(bounds[3]-bounds[1])]))},
            })    
