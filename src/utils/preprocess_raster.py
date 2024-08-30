#%%
import rasterio as rio
from rasterio.plot import show
from rasterio.warp import reproject, Resampling
import numpy as np
from collections import Counter
from rasterio.mask import mask
from shapely.geometry import box
#%%
def read_raster(file):
    with rio.open(file) as src_file:
        raster_file = src_file.read()
        transform = src_file.transform
        crs = src_file.crs
    return src_file, raster_file, transform, crs

def intersect_bounding_boxes(bounds1, bounds2):
    left = max(bounds1.left, bounds2.left)
    bottom = max(bounds1.bottom, bounds2.bottom)
    right = min(bounds1.right, bounds2.right)
    top = min(bounds1.top, bounds2.top)
    return (left, bottom, right, top)

def clip_raster(src_path, dst_path, bbox):
    '''
    This function was created with ChatGPT's help 
    '''
    with rio.open(src_path) as src:
        bbox_geom = [box(*bbox)]
        clipped, transform = mask(src, bbox_geom, crop=True)
        kwargs = src.meta.copy()
        kwargs.update({
            'height': clipped.shape[1],
            'width': clipped.shape[2],
            'transform': transform
        })

        with rio.open(dst_path, 'w', **kwargs) as dst:
            dst.write(clipped)

def preprocess(corine_path,nuts_path,clipped_corine_path,clipped_nuts_path):
    src_corine, _, _, _ = read_raster(corine_path)
    src_nuts, _, _, _ = read_raster(nuts_path)
    intersection_bounds = intersect_bounding_boxes(src_corine.bounds, src_nuts.bounds)
    clip_raster(corine_path, clipped_corine_path, intersection_bounds)
    clip_raster(nuts_path, clipped_nuts_path, intersection_bounds)

def filter_raster(raster_image, filter_codes):
    '''
    Filtering based on the agriculture codes,
    Since we wanna take a mean based on this filtered raster,
    I assigned "1" to all the locations in our filter_codes
    and "0" to all regions that fall outside it.
    This makes the mean calculation easy.

    '''
    raster_image = raster_image.squeeze()
    f_raster = np.zeros((raster_image.shape[0],raster_image.shape[1]), dtype=np.float32)
    f_raster = np.where(np.isin(raster_image, filter_codes), 1, 0)
    f_raster=f_raster[np.newaxis,...]
    return f_raster