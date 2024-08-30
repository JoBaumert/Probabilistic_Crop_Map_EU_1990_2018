#%%
import rasterio as rio
from rasterio.plot import show
import geopandas as gpd
import pandas as pd
import numpy as np
from rasterio.windows import from_bounds
import ee
import geemap


#%%
ee.Authenticate()
ee.Initialize()
#%%

path_to_selected_region_shapefile="/home/baumert/fdiexchange/baumert/project2/Input/NUTS/DEA_shape.tif"
path_to_GEE_assets="projects/jbdetect04/assets/"
path_to_taskfile="/home/baumert/fdiexchange/baumert/project2/Input/input_preprocessing_taskfile.xlsx"
#%%

region_shape=rio.open(path_to_selected_region_shapefile)
#this transform list is needed to specify the scale and crs of the produced data input files
transform_list=list(region_shape.transform)
#we also need the region in GEE for "filter bounds" in later steps
selected_region="DEA" #name of the region
path_to_file_in_GEE="projects/jbdetect04/assets/NUTS_RG_01M_2016_3035"
path_to_file_on_local_system="/home/baumert/fdiexchange/baumert/project1/Data/Raw_Data/NUTS/NUTS_RG_01M_2016_3035.shp.zip!/NUTS_RG_01M_2016_3035.shp"
agricultural_classes_path="/home/baumert/fdiexchange/baumert/project2/User_Specifications/CORINE_specifications.xlsx" #needed if CORINE is used
nuts_region_converter_output_path="/home/baumert/fdiexchange/baumert/project2/Input/NUTS/NUTS_code_number_converter.csv"
NUTS_shapes=ee.FeatureCollection(path_to_file_in_GEE)
selected_region_shape=NUTS_shapes.filter(ee.Filter.eq("FID",selected_region))
#%%

#%%
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

def mask_s2_clouds(image):
  """Masks clouds in a Sentinel-2 image using the QA band.

  Args:
      image (ee.Image): A Sentinel-2 image.

  Returns:
      ee.Image: A cloud-masked Sentinel-2 image.
  """
  qa = image.select('QA60')

  # Bits 10 and 11 are clouds and cirrus, respectively.
  cloud_bit_mask = 1 << 10
  cirrus_bit_mask = 1 << 11

  # Both flags should be set to zero, indicating clear conditions.
  mask = (
      qa.bitwiseAnd(cloud_bit_mask)
      .eq(0)
      .And(qa.bitwiseAnd(cirrus_bit_mask).eq(0))
  )

  return image.updateMask(mask).divide(10000)

def addNDVI(image):
    ndvi = image.normalizedDifference(['B8', 'B4']).rename('ndvi')
    return image.addBands(ndvi)

def addEVI(image):
   #formula here: https://custom-scripts.sentinel-hub.com/custom-scripts/sentinel-2/evi/
   EVI=image.expression(
    '2.5*(NIR-RED)/((NIR+6*RED-7.5*BLUE)+1)',{
    'NIR':image.select('B8'),
    'RED':image.select('B4'),
    'BLUE':image.select('B2')
      }
   ).rename("EVI")
   image=image.addBands(EVI)
   return image

def temp_gt_5(image):
   temp=image.select("temperature_2m")
   temp=temp.where(temp.lt(273.15+5),0).rename("gte5")
   temp=temp.where(temp.gte(273.15+5),1).rename("gte5")
   return image.addBands(temp)

"""2) get data functions"""

def get_INVEKOS(year,return_reprojection=True):
    if year==2023:
       DEA_INVEKOS=ee.FeatureCollection(path_to_GEE_assets+"NRW_2023")
    else:
       DEA_INVEKOS=ee.FeatureCollection(path_to_GEE_assets+"NRW_hist")
    DEA_INVEKOS_selected_year=DEA_INVEKOS.filter(ee.Filter.eq("WJ",int(year)))

    DEA_INVEKOS_selected_year_image=DEA_INVEKOS_selected_year.reduceToImage(
        properties=['CODE'],
    reducer= ee.Reducer.mode()
    )
    if return_reprojection:
        return DEA_INVEKOS_selected_year_image.reproject(crs="epsg:3035",crsTransform=transform_list[:6])
    else:
        return DEA_INVEKOS_selected_year_image

def get_max_NDVI(start_date,end_date,return_reprojection=True,cloudy_pixel_percentage=20):
    sentinel_imageCollection=(
    ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
    .filterDate(start_date, end_date)
    .filterBounds(selected_region_shape)
    .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE',cloudy_pixel_percentage))
    .map(mask_s2_clouds)
    )
    sentinel_imageCollection_with_NDVI=sentinel_imageCollection.map(addNDVI)
    NDVI_max_entire_period=sentinel_imageCollection_with_NDVI.select("ndvi").max()
    sentinel_imageCollection_with_EVI=sentinel_imageCollection.map(addEVI)

    if return_reprojection:
       return (
          NDVI_max_entire_period.reproject(crs="epsg:3035",crsTransform=transform_list[:6])
       )
    else:
       return(
            NDVI_max_entire_period
       )
    
def get_max_EVI(start_date,end_date,return_reprojection=True,cloudy_pixel_percentage=20):
    sentinel_imageCollection=(
    ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
    .filterDate(start_date, end_date)
    .filterBounds(selected_region_shape)
    .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE',cloudy_pixel_percentage))
    .map(mask_s2_clouds)
    )
    sentinel_imageCollection_with_EVI=sentinel_imageCollection.map(addEVI)
    EVI_max_entire_period=sentinel_imageCollection_with_EVI.select("EVI").max()

    if return_reprojection:
       return (
          EVI_max_entire_period.reproject(crs="epsg:3035",crsTransform=transform_list[:6])
       )
    else:
       return(
            EVI_max_entire_period
       )

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

def get_consecutive_dry_days(start_date,end_date,return_reprojection=True):
    #consecutive days with less than 1mm of precipitation
    era5_imageCollection=(
    ee.ImageCollection("ECMWF/ERA5_LAND/DAILY_AGGR")
    .filterDate(start_date, end_date)
    .filterBounds(selected_region_shape)
    )
    datain_t = era5_imageCollection.select("total_precipitation_sum").map(add_counter).sort('system:time_start')

    # create first image for iteration
    first = ee.List([ee.Image(datain_t.first())])

    #apply dry speall iteration function
    maxDrySpell = ee.ImageCollection.fromImages(
        datain_t.iterate(drySpells,first)
    ).max()# get the max value
    maxDrySpell=maxDrySpell.select("counter")
    if return_reprojection:
        return maxDrySpell.reproject(crs="epsg:3035",crsTransform=transform_list[:6])
    else:
        return maxDrySpell

def get_mean_temperature(start_date,end_date,return_reprojection=True):
    era5_imageCollection=(
    ee.ImageCollection("ECMWF/ERA5_LAND/DAILY_AGGR")
    .filterDate(start_date, end_date)
    .filterBounds(selected_region_shape)
    )
    monthly_temperature_mean=era5_imageCollection.select("temperature_2m").mean()
    if return_reprojection:
        return monthly_temperature_mean.reproject(crs="epsg:3035",crsTransform=transform_list[:6])
    else:
        return monthly_temperature_mean

def get_vegetation_period_aug_july(start_date,end_date,return_reprojection=True):
    era5_imageCollection=(
        ee.ImageCollection("ECMWF/ERA5_LAND/DAILY_AGGR")
        .filterDate(start_date, end_date)
        .filterBounds(selected_region_shape)
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
    .filterBounds(selected_region_shape)

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
    .filterBounds(selected_region_shape)

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
    .filterBounds(selected_region_shape)

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
    .filterBounds(selected_region_shape)

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
    .filterBounds(selected_region_shape)

    )
    mean_wind_speed=terraclimate_imageCollection.select("vs").mean()
    if return_reprojection:
        return mean_wind_speed.reproject(crs="epsg:3035",crsTransform=transform_list[:6])
    else:
        return mean_wind_speed
    
def get_CORINE_LCclass(year,agricultural_classes,return_reprojection=True):
    corine_full_dataset= ee.Image('COPERNICUS/CORINE/V20/100m/'+str(year))
    landcover = corine_full_dataset.select('landcover')
    landcover_remapped=landcover.remap(agricultural_classes,
                                    to=list(np.ones(len(agricultural_classes))),
                                    defaultValue=0)
    if return_reprojection:
        return landcover_remapped.reproject(crs="epsg:3035",crsTransform=transform_list[:6])
    else:
        return landcover_remapped
    
"""definition of frequency options"""
def get_start_and_end_dates(year,frequency):
    if frequency=="season":
        dates={
        "start_dates":[str(year-1)+"-07-01",str(year-1)+"-10-01",str(year)+"-01-01",str(year)+"-04-01",str(year)+"-07-01"],
        "end_dates":[str(year-1)+"-09-30",str(year-1)+"-12-31",str(year)+"-03-31",str(year)+"-06-30",str(year)+"-09-30"]
        }
    elif frequency=="vegperiod-monthly":
        dates={
        "start_dates":[str(year)+"-03-01",str(year)+"-04-01",str(year)+"-05-01",str(year)+"-06-01",str(year)+"-07-01",str(year)+"-08-01"],
        "end_dates":[str(year)+"-03-31",str(year)+"-04-30",str(year)+"-05-31",str(year)+"-06-30",str(year)+"-07-31",str(year)+"-08-31"]
        }
    elif frequency=="vegperiod":
        dates={
        "start_dates":[str(year)+"-03-01"],
        "end_dates":[str(year)+"-08-31"]
        }
    elif frequency=="year":
        dates=str(year)

    elif frequency=="static":
        dates=""
    return dates

def get_product_name(taskname,dates,index=None):
    if index is not None:
        return taskname[4:]+"_"+dates["start_dates"][index].replace("-","")+"_"+dates["end_dates"][index].replace("-","")
    elif dates!="":
        return taskname[4:]+"_"+dates
    else:
        return taskname[4:]
#%%
if __name__ == "__main__":
    tasks=pd.read_excel(path_to_taskfile)


    for i in range(len(tasks)):
        taskname=tasks["task"].iloc[i]
        year=tasks["year"].iloc[i]
        frequency=tasks["frequency"].iloc[i]
        dates=get_start_and_end_dates(year,frequency)

        if taskname=="get_INVEKOS":
            print("export "+get_product_name(taskname,dates))
            geemap.ee_export_image_to_drive(
                locals()[taskname](year), #here we call the function indicated by taskname
                folder="GEE",
                description=get_product_name(taskname,dates), 
                scale=100,          
                region=selected_region_shape.geometry()
            )
            
        elif taskname=="get_CORINE_LCclass":
            print("export "+get_product_name(taskname,dates))
            agricultural_classes=pd.read_excel(agricultural_classes_path)
            agricultural_classes=list(agricultural_classes["agricultural_corine_classes"])
            geemap.ee_export_image_to_drive(
                locals()[taskname](year,agricultural_classes), #here we call the function indicated by taskname
                folder="GEE",
                description=get_product_name(taskname,dates), 
                scale=100,          
                region=selected_region_shape.geometry()
            )
        elif frequency=="static":
            print("export "+get_product_name(taskname,dates))
            geemap.ee_export_image_to_drive(
                locals()[taskname](), #here we call the function indicated by taskname
                folder="GEE",
                description=get_product_name(taskname,dates), 
                scale=100,          
                region=selected_region_shape.geometry()
            )
        
        elif type(dates)==dict:
            for j in range(len(dates["start_dates"])):
                print("export " +get_product_name(taskname,dates,index=j))
                geemap.ee_export_image_to_drive(
                    locals()[taskname](dates["start_dates"][j],dates["end_dates"][j]), #here we call the function indicated by taskname
                    folder="GEE",
                    description=get_product_name(taskname,dates,index=j), 
                    scale=100,          
                    region=selected_region_shape.geometry()
            )
        
#%%
nuts_regs=gpd.read_file(path_to_file_on_local_system)
#%%

nuts_regs_selected_region=nuts_regs.iloc[np.where(np.array(nuts_regs["NUTS_ID"]).astype("U3")==selected_region)[0]]
nuts_regs_selected_region["number"]=np.arange(len(nuts_regs_selected_region))
nuts_regs_selected_region=nuts_regs_selected_region[["LEVL_CODE","NUTS_ID","number","geometry"]]

nuts_regs_selected_region.to_file("/home/baumert/fdiexchange/baumert/project2/Input/Temporary/nuts_regs_2016_"+selected_region+".shp")
nuts_regs_selected_regionfc = geemap.shp_to_ee("/home/baumert/fdiexchange/baumert/project2/Input/Temporary/nuts_regs_2016_"+selected_region+".shp")

nuts_regs_selected_regionfc=nuts_regs_selected_regionfc.filter(ee.Filter.eq("LEVL_CODE",3))
nuts_regs_selected_region_image=nuts_regs_selected_regionfc.reduceToImage(
  properties= ['number'],
  reducer= ee.Reducer.mode()
)

nuts_regs_selected_region[["NUTS_ID","number"]].to_csv(nuts_region_converter_output_path)
# %%
Map=geemap.Map()
Map.addLayer(nuts_regs_selected_region_image)
Map.set_center(8,50,8)
Map
# %%

# %%
def get_yield_potential(return_reprojection=True):
    
    yield_potential=ee.Image(path_to_GEE_assets+"sqr1000_250_v10_yieldpotential").select("b1")

    if return_reprojection:
        return yield_potential.reproject(crs="epsg:3035",crsTransform=transform_list[:6])
    else:
        return yield_potential
# %%
Map=geemap.Map()
Map.addLayer(get_yield_potential(),{"min":0,"max":100})
Map.set_center(8,50,8)
Map
# %%
geemap.ee_export_image_to_drive(
                    get_yield_potential(), #here we call the function indicated by taskname
                    folder="GEE",
                    description="yield_potential", 
                    scale=100,          
                    region=selected_region_shape.geometry()
            )
# %%