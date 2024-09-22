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
main_path = str(Path(Path(os.path.abspath(__file__)).parents[1]))
result_dir = os.path.join(main_path, "data/results/")
os.makedirs(result_dir, exist_ok=True)
raw_dir = main_path+"/data/raw/"
os.makedirs(raw_dir, exist_ok=True)
preprocessed_dir = main_path+"/data/preprocessed/"
os.makedirs(preprocessed_dir, exist_ok=True)
path_to_taskfile=main_path+"/data/input_preprocessing_taskfile.xlsx"
#%%
ee.Authenticate()
ee.Initialize()
# %%
reference_raster=rio.open(result_dir+"multi_band_raster/nuts_raster_2003.tif")
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
        if frequency!="static":
            
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

                #if the selected year is among those neeeded for LUCAS, also do the feature merge for LUCAS points
                if year in np.array(lucas_feature_years["feature_year"]):
                    print("export " +get_product_name(taskname,year)+" for LUCAS")
                    lucas_year=int(lucas_feature_years[lucas_feature_years["feature_year"]==year]["LUCAS_year"].iloc[0])
                    selected_LUCAS_data=LUCAS_dataset.filter(ee.Filter.eq("year",lucas_year)).select([
                        "id",
                        "point_id",
                        "year",
                        "nuts3",
                        "lc1"
                        "lc1_label",
                        "gps_lat"
                        
                    ])
                    
                    feature_image=locals()[taskname](dates["start_dates"],dates["end_dates"])
                    sampled_points=feature_image.sampleRegions(
                        collection=selected_LUCAS_data,
                        projection=ee.Projection("epsg:3035",transform=transform_list[:6])
                    )
                    
                    geemap.ee_export_vector_to_drive(
                    collection=sampled_points,
                    folder="GEE_DGPCM_19902020",
                    description="LUCAS_"+str(lucas_year)+"_"+get_product_name(taskname,year),
                    fileFormat='CSV',
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
            
            #export features for LUCAS points, for each LUCAS year individually to keep output data size limited
            feature_image=locals()[taskname]()
            
            for lucas_year in lucas_feature_years["LUCAS_year"].unique():
                lucas_year=int(lucas_year)
                print("export for LUCAS "+str(lucas_year))
                selected_LUCAS_data=LUCAS_dataset.filter(ee.Filter.eq("year",lucas_year)).select([
                        "id",
                        "point_id",
                        "year",
                        "nuts3",
                        "lc1"
                        "lc1_label",
                        "gps_lat"
                        
                ])
                
                sampled_points=feature_image.sampleRegions(
                    collection=selected_LUCAS_data,
                    projection=ee.Projection("epsg:3035",transform=transform_list[:6])
                    )
                
                geemap.ee_export_vector_to_drive(
                    collection=sampled_points,
                    folder="GEE_DGPCM_19902020",
                    description="LUCAS_"+str(lucas_year)+"_"+get_product_name(taskname),
                    fileFormat='CSV',
                )

            
#%%
for lucas_year in lucas_feature_years["LUCAS_year"].unique():
    print(lucas_year)
#%%
lucas_feature_years["LUCAS_year"].unique()[0]
#%%
a=get_mean_temperature("1987-01-01","1987-12-31",return_reprojection=True)
#%%
lucas_year=2006
selected_LUCAS_data=LUCAS_dataset.filter(ee.Filter.eq("year",lucas_year)).select([
                        "id",
                        "point_id",
                        "year",
                        "nuts3",
                        "lc1"
                        "lc1_label",
                        "gps_lat"
                        
                ])
                
sampled_points=feature_image.sampleRegions(
    collection=selected_LUCAS_data,
    projection=ee.Projection("epsg:3035",transform=transform_list[:6])
    )
Map=geemap.Map()
Map.addLayer(sampled_points)
Map.set_center(8,50,8)
Map
# %%
#this works!
geemap.ee_export_image_to_drive(
                get_sand(), #here we call the function indicated by taskname
                folder="GEE",
                description="sand_test_EU_v3", 
                scale=1000,   
                region=ee.Geometry.Rectangle(list(reference_raster.bounds),proj="epsg:3035",evenOdd=False)
            )
# %%
list(reference_raster.bounds)
#%%



# %%
LUCAS_dataset = ee.FeatureCollection('JRC/LUCAS_HARMO/THLOC/V1')
elevation_image=get_elevation()
# %%

selected_LUCAS_data=LUCAS_dataset.filter(ee.Filter.eq("year",2018)).select([
    "id",
    "point_id",
    "year",
    "nuts3",
    "lc1"
    "lc1_label",
    "gps_lat"
    
])
# %%
sampled_points=elevation_image.sampleRegions(
    collection=selected_LUCAS_data,
    projection=ee.Projection("epsg:3035",transform=transform_list[:6])
    
)

# %%
geemap.ee_export_vector_to_drive(
    collection=sampled_points,
    folder="GEE",
    description="elevation_LUCAS_2018",
    fileFormat='CSV',
)
#%%
a=geemap.ee_to_numpy(
                get_sand(), #here we call the function indicated by taskname
                scale=1000,   
                region=ee.Geometry.Rectangle(list(reference_raster.bounds),proj="epsg:3035",evenOdd=False)
            )
# %%
show(a.transpose(2,0,1)[0])
# %%
elevation_LUCAS_test=pd.read_csv(raw_dir+"elevation_LUCAS_2018.csv")
# %%
elevation_LUCAS_test.iloc[:,1]
# %%
comparison_data=pd.read_csv(raw_dir+"elevation.csv")
# %%
comparison_data=comparison_data[comparison_data["year"]==2018]
# %%
comparison_data
# %%
selection=pd.merge(elevation_LUCAS_test[["point_id","be75"]],comparison_data[["point_id","elevation"]],
                   how="right",on="point_id")
# %%
import matplotlib.pyplot as plt
plt.scatter(x=selection.be75,y=selection.elevation,s=0.1)
# %%
"""test"""
correct_raster=rio.open(result_dir+"multi_band_raster/nuts_raster_2003.tif")
# %%
comparison_raster=rio.open(raw_dir+"mean_temperature_1987.tif")
# %%
comparison_raster.transform
# %%
correct_raster.transform
# %%
show(comparison_raster.read()[0])
# %%
"""unrelated tests"""
cropdata=pd.read_csv(result_dir+"csv/filtered_regional_cropdata.csv")
# %%
cropdata[cropdata["3"]=="LEVL"]["2"].unique()
# %%
cropdata[(cropdata["3"]=="LEVL")&(cropdata["4"]==2010)&(cropdata["1"]=="DE")]
# %%
"""CORINE"""
corine_years=[1990,2000,2006,2012,2018]
for year in corine_years:
    corine_image=ee.Image('COPERNICUS/CORINE/V20/100m/'+str(year))
    corine_image=corine_image.reproject(crs="epsg:3035",crsTransform=transform_list[:6])
    geemap.ee_export_image_to_drive(
                    corine_image, #here we call the function indicated by taskname
                    folder="GEE_DGPCM_19902020",
                    description="CORINE_"+str(year), 
                    scale=1000,   
                    region=ee.Geometry.Rectangle(list(reference_raster.bounds),proj="epsg:3035",evenOdd=False)
                )
# %%
