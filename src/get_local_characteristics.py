#%%
import rasterio as rio
from rasterio.plot import show
import geopandas as gpd
import pandas as pd
import numpy as np
from rasterio.windows import from_bounds
import ee
import geemap
import os
from pathlib import Path
# %%
try:
    main_path = str(Path(Path(os.path.abspath(__file__)).parents[0]))
    data_main_path=open(main_path+"/src/data_main_path.txt").read()[:-1]
except:
    main_path = str(Path(Path(os.path.abspath(__file__)).parents[1]))
    data_main_path=open(main_path+"/src/data_main_path.txt").read()[:-1]


#%%
result_dir = data_main_path+ "/results/"
#os.makedirs(result_dir, exist_ok=True)
raw_dir = data_main_path+"/raw/"
#os.makedirs(raw_dir, exist_ok=True)
preprocessed_dir = data_main_path+"/preprocessed/"
#os.makedirs(preprocessed_dir, exist_ok=True)
path_to_taskfile=data_main_path+"/input_preprocessing_taskfile.xlsx"

#%%
ee.Authenticate()
ee.Initialize()
# %%
reference_raster=rio.open(preprocessed_dir+"rasters/nuts_2003.tif")
transform_list=list(reference_raster.transform)

#%%

""""""
"""FUNCTIONS"""
"""1) helper functions"""

def add_counter(img):
    return img.addBands(ee.Image.constant(0).uint8().rename('counter'))

def drySpells(img, list):
  # get previous image
  prev = ee.Image(ee.List(list).get(-1))
  # find areas gt precipitation threshold (gt==0, lt==1)
  dry = img.select('total_precipitation_sum').lt(0.001)
  # add previous day counter to today's counter
  accum = prev.select('counter').add(dry).rename('counter')
  #create a result image for iteration
  # precip < thresh will equal the accumulation of counters
  # otherwise it will equal zero
  out = img.select('total_precipitation_sum').addBands(
        img.select('counter').where(dry.eq(1),accum)
      ).uint8()
  return ee.List(list).add(out)

def temp_gt_5(image):
   temp=image.select("temperature_2m")
   temp=temp.where(temp.lt(273.15+5),0).rename("gte5")
   temp=temp.where(temp.gte(273.15+5),1).rename("gte5")
   return image.addBands(temp)

"""2) get data functions"""


def get_sand(return_reprojection=True):
    sand_image=ee.Image("OpenLandMap/SOL/SOL_SAND-WFRACTION_USDA-3A1A1A_M/v02").select("b0") #select only sand content at 0 cm
    if return_reprojection:
        return sand_image.reproject(crs="epsg:3035",crsTransform=transform_list[:6])
    else:
        return sand_image       

def get_clay(return_reprojection=True):
    clay_image=ee.Image("OpenLandMap/SOL/SOL_CLAY-WFRACTION_USDA-3A1A1A_M/v02").select("b0") #select only sand content at 0 cm
    if return_reprojection:
        return clay_image.reproject(crs="epsg:3035",crsTransform=transform_list[:6])
    else:
        return clay_image 
    
def get_oc(return_reprojection=True):
    oc_image=ee.Image("OpenLandMap/SOL/SOL_ORGANIC-CARBON_USDA-6A1C_M/v02").select("b0") #select only sand content at 0 cm
    if return_reprojection:
        return oc_image.reproject(crs="epsg:3035",crsTransform=transform_list[:6])
    else:
        return oc_image 
    
def get_bulk_density(return_reprojection=True):
    bd_image=ee.Image("OpenLandMap/SOL/SOL_BULKDENS-FINEEARTH_USDA-4A1H_M/v02").select("b0") #select only sand content at 0 cm
    if return_reprojection:
        return bd_image.reproject(crs="epsg:3035",crsTransform=transform_list[:6])
    else:
        return bd_image 

    
def get_elevation(return_reprojection=True):
    elevation_image=ee.Image("USGS/GMTED2010").select("be75")
    if return_reprojection:
        return elevation_image.reproject(crs="epsg:3035",crsTransform=transform_list[:6])
    else:
        return elevation_image
    
def get_slope(return_reprojection=True):
    elevation_image=ee.Image("USGS/GMTED2010").select("be75")
    if return_reprojection:
        return ee.Terrain.slope(elevation_image).reproject(crs="epsg:3035",crsTransform=transform_list[:6])
    else:
        return ee.Terrain.slope(elevation_image)

def get_mean_temperature(start_date,end_date,return_reprojection=True):
    era5_imageCollection=(
    ee.ImageCollection("ECMWF/ERA5_LAND/DAILY_AGGR")
    .filterDate(start_date, end_date)
    .filterBounds(ee.Geometry.Rectangle(list(reference_raster.bounds),proj="epsg:3035",evenOdd=False))
    )
    monthly_temperature_mean=era5_imageCollection.select("temperature_2m").mean()
    if return_reprojection:
        return monthly_temperature_mean.reproject(crs="epsg:3035",crsTransform=transform_list[:6])
    else:
        return monthly_temperature_mean



def get_vegetation_period_over_year(start_date,end_date,return_reprojection=True):
    era5_imageCollection=(
        ee.ImageCollection("ECMWF/ERA5_LAND/DAILY_AGGR")
        .filterDate(start_date, end_date)
        .filterBounds(ee.Geometry.Rectangle(list(reference_raster.bounds),proj="epsg:3035",evenOdd=False))
    )
    temp=era5_imageCollection.map(temp_gt_5).select("gte5").sum()
    if return_reprojection:
        return temp.reproject(crs="epsg:3035",crsTransform=transform_list[:6])
    else:
        return temp

def get_mean_solar_radiation_downwards_sum(start_date,end_date, return_reprojection=True):
    era5_imageCollection=(
    ee.ImageCollection("ECMWF/ERA5_LAND/DAILY_AGGR")
    .filterDate(start_date, end_date)
    .filterBounds(ee.Geometry.Rectangle(list(reference_raster.bounds),proj="epsg:3035",evenOdd=False))

    )
    mean_sr=era5_imageCollection.select("surface_solar_radiation_downwards_sum").mean().divide(1000000) #in megajoule
    if return_reprojection:
        return mean_sr.reproject(crs="epsg:3035",crsTransform=transform_list[:6])
    else:
        return mean_sr

def get_precipitation_sum(start_date,end_date, return_reprojection=True):
    terraclimate_imageCollection=(
    ee.ImageCollection("IDAHO_EPSCOR/TERRACLIMATE")
    .filterDate(start_date, end_date)
    .filterBounds(ee.Geometry.Rectangle(list(reference_raster.bounds),proj="epsg:3035",evenOdd=False))

    )
    precipitation_sum=terraclimate_imageCollection.select("pr").sum().toDouble()
    if return_reprojection:
        return precipitation_sum.reproject(crs="epsg:3035",crsTransform=transform_list[:6])
    else:
        return precipitation_sum

def get_mean_soil_moisture(start_date,end_date, return_reprojection=True):
    terraclimate_imageCollection=(
    ee.ImageCollection("IDAHO_EPSCOR/TERRACLIMATE")
    .filterDate(start_date, end_date)
    .filterBounds(ee.Geometry.Rectangle(list(reference_raster.bounds),proj="epsg:3035",evenOdd=False))

    )
    mean_soil_moisture=terraclimate_imageCollection.select("soil").mean()
    if return_reprojection:
        return mean_soil_moisture.reproject(crs="epsg:3035",crsTransform=transform_list[:6])
    else:
        return mean_soil_moisture
    
def get_mean_vapor_pressure(start_date,end_date, return_reprojection=True):
    terraclimate_imageCollection=(
    ee.ImageCollection("IDAHO_EPSCOR/TERRACLIMATE")
    .filterDate(start_date, end_date)
    .filterBounds(ee.Geometry.Rectangle(list(reference_raster.bounds),proj="epsg:3035",evenOdd=False))

    )
    mean_vapor_pressure=terraclimate_imageCollection.select("vap").mean()
    if return_reprojection:
        return mean_vapor_pressure.reproject(crs="epsg:3035",crsTransform=transform_list[:6])
    else:
        return mean_vapor_pressure
    
def get_mean_wind_speed(start_date,end_date, return_reprojection=True):
    terraclimate_imageCollection=(
    ee.ImageCollection("IDAHO_EPSCOR/TERRACLIMATE")
    .filterDate(start_date, end_date)
    .filterBounds(ee.Geometry.Rectangle(list(reference_raster.bounds),proj="epsg:3035",evenOdd=False))

    )
    mean_wind_speed=terraclimate_imageCollection.select("vs").mean()
    if return_reprojection:
        return mean_wind_speed.reproject(crs="epsg:3035",crsTransform=transform_list[:6])
    else:
        return mean_wind_speed
    

    
"""definition of frequency options"""
def get_start_and_end_dates(year,frequency):
    if frequency=="annual":
        dates={
        "start_dates":str(year)+"-01-01",
        "end_dates":str(year)+"-12-31"
        }

    else:
        dates=""
    return dates

def get_product_name(taskname,year=None):
    if year is not None:
        return taskname[4:]+"_"+str(year)
    else:
        return taskname[4:]

#%%
if __name__ == "__main__":
    tasks=pd.read_excel(path_to_taskfile,sheet_name="task")
    years=pd.read_excel(path_to_taskfile,sheet_name="all_feature_years")
    lucas_feature_years=pd.read_excel(path_to_taskfile,sheet_name="LUCAS_feature_years")
    LUCAS_dataset = ee.FeatureCollection('JRC/LUCAS_HARMO/THLOC/V1')

    for i in range(len(tasks)):
        taskname=tasks["task"].iloc[i]
        frequency=tasks["frequency"].iloc[i]

        if taskname=="CORINE":
            """CORINE"""
            #CORINE classes:
            mask_string="""
            b('landcover') == 211 ||
            b('landcover') == 212 ||
            b('landcover') == 213 ||
            b('landcover') == 221 ||
            b('landcover') == 222 ||
            b('landcover') == 223 ||
            b('landcover') == 231 ||
            b('landcover') == 241 ||
            b('landcover') == 242 ||
            b('landcover') == 243 ||
            b('landcover') == 244 
            """

            corine_years=[1990,2000,2006,2012,2018]
            for year in corine_years:
                corine_image=ee.Image('COPERNICUS/CORINE/V20/100m/'+str(year)).select('landcover')
                mask = corine_image.expression(mask_string)
                maks_reproj=mask.reduceResolution(
                reducer=ee.Reducer.mean(),maxPixels=100).reproject(
                    crs="epsg:3035",crsTransform=transform_list[:6])
                
                geemap.ee_export_image_to_drive(
                                maks_reproj, 
                                folder="GEE_DGPCM_19902020",
                                description="CORINE_"+str(year), 
                                scale=1000,   
                                region=ee.Geometry.Rectangle(list(reference_raster.bounds),proj="epsg:3035",evenOdd=False)
                            )
        
        elif frequency!="static":
            for j in range(len(years)):
                year=years["year"].iloc[j]
                dates=get_start_and_end_dates(year,frequency)
                print("export " +get_product_name(taskname,year))

                geemap.ee_export_image_to_drive(
                    locals()[taskname](dates["start_dates"],dates["end_dates"]), #here we call the function indicated by taskname
                    folder="GEE_DGPCM_19902020",
                    description=get_product_name(taskname,year), 
                    scale=1000,          
                    region=ee.Geometry.Rectangle(list(reference_raster.bounds),proj="epsg:3035",evenOdd=False)
                )               
          
        else:
            print("export " +get_product_name(taskname))
            #export features for entire EU
            geemap.ee_export_image_to_drive(
                    locals()[taskname](), #here we call the function indicated by taskname
                    folder="GEE_DGPCM_19902020",
                    description=get_product_name(taskname), 
                    scale=1000,          
                    region=ee.Geometry.Rectangle(list(reference_raster.bounds),proj="epsg:3035",evenOdd=False)
                )
            



            
#%%





