import numpy as np
import pickle as pkl
import os
import ee
from tqdm import tqdm



def mosaic_by_date(imcol):
    
    """
    spatially mosaic adjoining sentinel-2 footprints
    
    parameters
        imcol(EE object): image collection
    """
    
    # convert image collection to list
    imlist = imcol.toList(imcol.size().getInfo())
    
    # get unique list of dates
    unique_dates = imlist.map(lambda im: ee.Image(im).date().format("YYYY-MM-dd")).distinct()

    # filter images containing same dates/1-day offset and mosaic
    def mosaic(d):
        d = ee.Date(d)

        im = imcol.filterDate(d, d.advance(1, "day")).mosaic()
        return im.set({'system:time_start':d.millis(), 'system:id':d.format("YYYY-MM-dd")})
    
    mosaic_imlist = unique_dates.map(mosaic)
    
    return ee.ImageCollection(mosaic_imlist)
    

    
def prepare_collection(start_date, end_date, ignore_idx, aoi, cloud_filter):
    
    """
    filter sentinel-2 collection spatially/temporally
    removes cloudy observations using specified cloud cover %
    additionally remove images using ignore_idx
    harmonized S2 collection used to correct dynamic range shifts.
    
    parameters
        start_date(string): start date of collection e.g YYYY-MM-DD
        end_date(string) end date of collection e.g YYYY-MM-DD
        ignore_idx(list): indices of images to exclude from collection e.g. [1, 20, 22]
        aoi(EE object): area of interest for spatial filter
        cloud_filter(int): cloud cover percentage e.g. 40

    """
    
    collection = ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED").filterBounds(aoi).filterDate(start_date, end_date)
    collection = collection.select(['B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A','B11', 'B12'])
    
    if cloud_filter is not None:
        collection = collection.filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', cloud_filter))
        
    if ignore_idx is not None:
         collection = collection.filter(ee.Filter.inList('system:index', ignore_idx).Not())
        
    # mosaic spatially
    collection = mosaic_by_date(collection)
    return collection



def generate_array(collection, csv, output_directory):
    
    """
    generates for each point an array of TxC
    T - number of time steps
    C - number of channels
    
    parameters
        collection(EE object): image collection
        csv(dataframe object): dataframe containing lat and long columns
        output_directory(string)
    """
    
    # create directory
    os.makedirs(output_directory, exist_ok=True)
    
    # for each row, get lon, lat and ID
    for index, row in tqdm(csv.iterrows()):
        geometry = ee.Geometry.Point([row['Lon'], row['Lat']])
        geo_id = row['ID']
        
        filename = os.path.join(output_directory, geo_id+'.npy')
        
        try:
            assert os.path.exists(filename)

        except:
            collection = collection.map(lambda img: img.set('temporal', ee.Image(img).reduceRegion(reducer = ee.Reducer.mean(), geometry= geometry, scale=10).values()))

            # convert img collection to 2d array of shape TxC
            np_arr = np.array(collection.aggregate_array('temporal').getInfo())

            # save
            np.save(filename, np_arr)
            
        else:
            continue
            


    
