#%%
import rasterio as rio
from rasterio import features
from rasterio.plot import show
from rasterio.transform import from_gcps
from rasterio.control import GroundControlPoint as GCP
from rasterio.windows import from_bounds
import geopandas as gpd
import pandas as pd
from pathlib import Path
import os
import numpy as np
import zipfile
import xarray
import gc
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
#%%
"""
this script allows to visualize the raster files for the entire EU at once
"""
try:
    main_path = str(Path(Path(os.path.abspath(__file__)).parents[0]))
    data_main_path=open(main_path+"/src/data_main_path.txt").read()[:-1]
except:
    main_path = str(Path(Path(os.path.abspath(__file__)).parents[1]))
    data_main_path=open(main_path+"/src/data_main_path.txt").read()[:-1]

raw_dir = data_main_path+"/raw"
preprocessed_dir = data_main_path+"/preprocessed"
preprocessed_csv_dir=preprocessed_dir+"/csv/"
preprocessed_raster_dir=preprocessed_dir+"/rasters/"
os.makedirs(preprocessed_raster_dir, exist_ok=True)
results_dir=data_main_path+"/results"
os.makedirs(results_dir, exist_ok=True)
prior_proba_output_dir=results_dir+"/numpy_arrays/prior_crop_probas/"
os.makedirs(prior_proba_output_dir, exist_ok=True)
posterior_proba_output_dir=results_dir+"/numpy_arrays/posterior_crop_probas/"
os.makedirs(posterior_proba_output_dir,exist_ok=True)
resulting_parameters_dir=results_dir+"/csv/estimation_parameters_and_scalers/"
os.makedirs(resulting_parameters_dir, exist_ok=True)
simulated_cropshares_dir=results_dir+"/multi_band_raster/simulted_crop_shares/"
os.makedirs(simulated_cropshares_dir,exist_ok=True)
parameter_path=data_main_path+"/delineation_and_parameters/"
user_parameter_path=parameter_path+"user_parameters.xlsx"
GEE_data_path=raw_dir+"/GEE/"
visualizations_path=results_dir+"/visualizations/"
os.makedirs(visualizations_path,exist_ok=True)
grid_path=raw_dir+"/Grid/"
shapefile_path=raw_dir+"/nuts_shapedata/"
EU_posterior_map_path=results_dir+"/multi_band_raster/EU_crop_map/"

albania_shapefile_path=shapefile_path+"albania_shapefile.zip!/ALB_adm0.shp"
bosnia_shapefile_path=shapefile_path+"bosnia_shapefile.zip!/BIH_adm0.shp"
kosovo_shapefile_path=shapefile_path+"kosovo_shapefile.zip!/XKO_adm0.shp"
serbia_shapefile_path=shapefile_path+"serbia_shapefile.zip!/SRB_adm0.shp"
#%%
EUmap=rio.open(EU_posterior_map_path+"EU_expected_crop_shares_1991.tif").read()
#%%
bands=pd.read_csv(EU_posterior_map_path+"bands.csv")
#%%
bands
#%%
show(EUmap[6])
#%%
EUmap[6][np.where(EUmap[6]>0)]

#%%
year=2000
EU_raster=rio.open(
    EU_posterior_map_path+"EU_expected_crop_shares_"+str(year)+".tif"
)
transform=EU_raster.transform

EU_raster_read=EU_raster.read()

#%%
#import NUTS data
NUTS=gpd.read_file(shapefile_path+"NUTS_RG_01M_2016_3035.shp/NUTS_RG_01M_2016_3035.shp")
#%%
NUTS0=gpd.GeoDataFrame(NUTS[NUTS["LEVL_CODE"]==0])
NUTS1=gpd.GeoDataFrame(NUTS[NUTS["LEVL_CODE"]==1])
NUTS2=gpd.GeoDataFrame(NUTS[NUTS["LEVL_CODE"]==2])
NUTS3=gpd.GeoDataFrame(NUTS[NUTS["LEVL_CODE"]==3])
# %%

NUTS0_regs=np.unique(NUTS0["NUTS_ID"])
NUTS0_index=np.arange(len(NUTS0_regs))+1
NUTS0.sort_values(by="NUTS_ID",inplace=True)
NUTS0["country_index"]=NUTS0_index
NUTS1_regs=np.unique(NUTS1["NUTS_ID"])
NUTS1_index=np.arange(len(NUTS1_regs))+1
NUTS1.sort_values(by="NUTS_ID",inplace=True)
NUTS1["country_index"]=NUTS1_index
NUTS2_regs=np.unique(NUTS2["NUTS_ID"])
NUTS2_index=np.arange(len(NUTS2_regs))+1
NUTS2.sort_values(by="NUTS_ID",inplace=True)
NUTS2["country_index"]=NUTS2_index
NUTS3_regs=np.unique(NUTS3["NUTS_ID"])
NUTS3_index=np.arange(len(NUTS3_regs))+1
NUTS3.sort_values(by="NUTS_ID",inplace=True)
NUTS3["country_index"]=NUTS3_index


#%%
geom_value = ((geom,value) for geom, value in zip(NUTS0.geometry, NUTS0.country_index))
NUTS0_raster=features.rasterize(
    geom_value,
    out_shape=EU_raster.shape,
    transform=transform,
)

geom_value = ((geom,value) for geom, value in zip(NUTS1.geometry, NUTS1.country_index))
NUTS1_raster=features.rasterize(
    geom_value,
    out_shape=EU_raster.shape,
    transform=transform,
)

geom_value = ((geom,value) for geom, value in zip(NUTS2.geometry, NUTS2.country_index))
NUTS2_raster=features.rasterize(
    geom_value,
    out_shape=EU_raster.shape,
    transform=transform,
)

geom_value = ((geom,value) for geom, value in zip(NUTS3.geometry, NUTS3.country_index))
NUTS3_raster=features.rasterize(
    geom_value,
    out_shape=EU_raster.shape,
    transform=transform,
)
#%%
#add shapefiles for albania, bosnia, kosovo, serbia
albania_boundary=gpd.read_file(albania_shapefile_path)
bosnia_boundary=gpd.read_file(bosnia_shapefile_path)
kosovo_boundary=gpd.read_file(kosovo_shapefile_path)
serbia_boundary=gpd.read_file(serbia_shapefile_path)

albania_boundary_epsg3035=albania_boundary.to_crs("epsg:3035")
bosnia_boundary_epsg3035=bosnia_boundary.to_crs("epsg:3035")
kosovo_boundary_epsg3035=kosovo_boundary.to_crs("epsg:3035")
serbia_boundary_epsg3035=serbia_boundary.to_crs("epsg:3035")

albania_boundary_epsg3035=albania_boundary_epsg3035["geometry"]
bosnia_boundary_epsg3035=bosnia_boundary_epsg3035["geometry"]
kosovo_boundary_epsg3035=kosovo_boundary_epsg3035["geometry"]
serbia_boundary_epsg3035=serbia_boundary_epsg3035["geometry"]

albania_boundary_epsg3035_df=pd.DataFrame(albania_boundary_epsg3035)
albania_boundary_epsg3035_df.insert(0,"CNTR_CODE","AL")
bosnia_boundary_epsg3035_df=pd.DataFrame(bosnia_boundary_epsg3035)
bosnia_boundary_epsg3035_df.insert(0,"CNTR_CODE","BA")
kosovo_boundary_epsg3035_df=pd.DataFrame(kosovo_boundary_epsg3035)
kosovo_boundary_epsg3035_df.insert(0,"CNTR_CODE","XK")
serbia_boundary_epsg3035_df=pd.DataFrame(serbia_boundary_epsg3035)
serbia_boundary_epsg3035_df.insert(0,"CNTR_CODE","RS")

#%%
NUTS0=NUTS0[["CNTR_CODE","geometry"]]
NUTS0=pd.concat((NUTS0,albania_boundary_epsg3035_df))
NUTS0=pd.concat((NUTS0,kosovo_boundary_epsg3035_df))
NUTS0=pd.concat((NUTS0,bosnia_boundary_epsg3035_df))
NUTS0=pd.concat((NUTS0,serbia_boundary_epsg3035_df))
#%%
#get relevasnt grid as shapefile

"""create one large grid shapefile"""

all_grids=pd.DataFrame()
i=0
#test=["DK_1km.zip","DE_1km.zip","AT_1km.zip"]
for directory in os.listdir(grid_path):
    print(directory)
 #   directory=test[i]
  #  i+=1
    if directory[-3:]=="zip":
        for file in zipfile.ZipFile(grid_path+directory).namelist():
            if file[-7:]=="1km.shp":
                print(directory)
                all_grids=pd.concat((
                    all_grids,
                    gpd.read_file(grid_path+directory+"!/"+file)
                ))
        

all_grids.drop_duplicates("CELLCODE",inplace=True)
#%%
#crop bands
CAPREG_data=pd.read_csv(preprocessed_csv_dir+"preprocessed_CAPREG_step3.csv")
all_crops=np.sort(np.unique(CAPREG_data["DGPCM_crop_code"]))

#%%
all_grids=all_grids[(all_grids["EOFORIGIN"]>EU_raster.bounds[0])&
          (all_grids["EOFORIGIN"]<EU_raster.bounds[2])&
          (all_grids["NOFORIGIN"]>EU_raster.bounds[1])&
          (all_grids["NOFORIGIN"]<EU_raster.bounds[3])]

#%%
""""""

#%%



for year in [1991,1992,1993,1994,1996,1997,1998,1999,2001,2002,2003,2004,2006,2007,2008,2009,2011,2012,2013,2014,2015,2016,2017]:
    EU_raster_read=rio.open(
        EU_posterior_map_path+"EU_expected_crop_shares_"+str(year)+".tif"
    ).read()
    for selected_crop in ["GRAS","SWHE","MAIZ"]:
        print(selected_crop+" "+str(year))
        crop_grid=all_grids.copy()
        east=((np.array(crop_grid.EOFORIGIN)-EU_raster.bounds[0])/1000).astype(int)
        #north=(np.abs((np.array(crop_grid.NOFORIGIN)-EU_raster.bounds[3])/1000)).astype(int)
        north=(np.abs((EU_raster.bounds[3]-np.array(crop_grid.NOFORIGIN))/1000)).astype(int)
        crop_grid["on_land"]=np.where(NUTS0_raster[north,east]>0,1,0)
        crop_grid["crop_share"]=EU_raster_read[np.where(all_crops==selected_crop)[0][0]+1][north,east]
        crop_grid["UAA"]=EU_raster_read[0][north,east]
        crop_grid=crop_grid[crop_grid["on_land"]==1]
        crop_grid=crop_grid[crop_grid["UAA"]>0]

        crop_grid=gpd.GeoDataFrame(crop_grid)



        if selected_crop=="GRAS":
            selected_cmap="YlGn"
            max_val=1000#/1000
        elif selected_crop=="SWHE":
            selected_cmap="YlOrRd"
            max_val=600#/1000
        elif selected_crop=="MAIZ":
            selected_cmap="Blues"
            max_val=600#/1000



        plt.figure(figsize=(12, 12))
        fig, ax = plt.subplots(1, 1, figsize=(12, 12))
        NUTS0.plot(ax=ax,facecolor="lightgrey")
        gpd.GeoDataFrame(crop_grid).plot(ax=ax,column="crop_share",
                    legend=True,
                    cmap=selected_cmap,  # YlGn "YlOrRd"
                    vmin=0,
                    vmax=max_val,)
        NUTS0.boundary.plot(ax=ax,edgecolor="darkgrey",linewidth=0.5)
        ax.set_xlim(EU_raster.bounds[0],EU_raster.bounds[2])
        ax.set_ylim(EU_raster.bounds[1],EU_raster.bounds[3])

        plt.axis("off")

        Path(visualizations_path).mkdir(parents=True, exist_ok=True)
        plt.savefig(visualizations_path+"share_of_"+selected_crop+"_"+str(year)+".png")
        plt.close(fig)
#%%
show(EU_raster_read[np.where(all_crops==selected_crop)[0][0]+1])
#%%
np.where(all_crops=="GRAS")
#%%
show(np.where(EU_raster_read[6]>100,1,0))

#%%
EU_raster_read.shape
# %%
EU_raster_crops=EU_raster_read[1:]
order=np.argmax(EU_raster_crops,axis=0)
crops,freq=np.unique(order[np.where(EU_raster_read[0]>0)],return_counts=True)
# %%
all_crops[crops[np.argsort(freq)]][::-1]
# %%
order.shape
#%%
year=2018
EU_raster_read=rio.open(
    EU_posterior_map_path+"EU_expected_crop_shares_"+str(year)+".tif"
).read()
EU_raster_crops=EU_raster_read[1:]
order=np.argmax(EU_raster_crops,axis=0)
crops,freq=np.unique(order[np.where(EU_raster_read[0]>0)],return_counts=True)

crop_grid=all_grids.copy()
east=((np.array(crop_grid.EOFORIGIN)-EU_raster.bounds[0])/1000).astype(int)
#north=(np.abs((np.array(crop_grid.NOFORIGIN)-EU_raster.bounds[3])/1000)).astype(int)
north=(np.abs((EU_raster.bounds[3]-np.array(crop_grid.NOFORIGIN))/1000)).astype(int)
crop_grid["on_land"]=np.where(NUTS0_raster[north,east]>0,1,0)
crop_grid["crop_share"]=order[north,east]
crop_grid["UAA"]=EU_raster_read[0][north,east]
crop_grid=crop_grid[crop_grid["on_land"]==1]
crop_grid=crop_grid[crop_grid["UAA"]>0]

crop_grid=gpd.GeoDataFrame(crop_grid)
#%%
all_crops
# %%
crop_colors={
            "BARL":[250, 156, 5],  # cereals
            "CITR":[194, 16, 132],  # fruits (including citr) and veg
            "DWHE":[250, 156, 5],  # wheat (durum and soft)
            "FARA":[27, 115, 20],  # gras and fara
            "FRUI":[194, 16, 132],  # fruits (including citr) and veg
            "GRAS":[27, 115, 20],  # gras and fara (grass)
            "INDU":[233, 235, 211], #other
            "MAIZ":[16, 72, 194],  # mais
            "OATS":[250, 156, 5], #cereals
            "OCER":[250, 156, 5], #cerals
            "OLIV":[3, 23, 0],
            "PARI":[5, 5, 247],
            "POTA":[247, 5, 5], #pota and sugb
            "PULS":[233, 235, 211], #other
            "RAPE":[250, 250, 5],  # oil
            "ROOF":[233, 235, 211], #other
            "RYEM":[250, 156, 5],  # cerelas
            "SOYA":[250, 250, 5],  # oil
            "SUGB":[247, 5, 5], #pota and sugb
            "SUNF":[250, 250, 5],  # oil
            "SWHE":[250, 156, 5],  # cereals
            "TEXT":[233, 235, 211], #other
            "TOBA":[233, 235, 211], #other
            "VEGE":[194, 16, 132],  # fruits (including citr) and veg
            "VINY":[105, 13, 80]  # viny
        }

custom_cmap_dominant_crops=np.ndarray((len(all_crops),3))
for c,crop in enumerate(all_crops):
    custom_cmap_dominant_crops[c]=crop_colors[crop]
custom_cmap_dominant_crops = custom_cmap_dominant_crops / 256
custom_cmap_dominant_crops = np.insert(custom_cmap_dominant_crops, 3, np.ones(len(custom_cmap_dominant_crops)), axis=1)
custom_cmap_dominant_crops = ListedColormap(custom_cmap_dominant_crops)
#%%
plt.figure(figsize=(12, 12))
fig, ax = plt.subplots(1, 1, figsize=(12, 12))
NUTS0.plot(ax=ax,facecolor="lightgrey")

gpd.GeoDataFrame(crop_grid).plot(ax=ax,column="crop_share",
            legend=True,
            cmap=custom_cmap_dominant_crops)

NUTS0.boundary.plot(ax=ax,edgecolor="darkgrey",linewidth=0.5)
ax.set_xlim(EU_raster.bounds[0],EU_raster.bounds[2])
ax.set_ylim(EU_raster.bounds[1],EU_raster.bounds[3])

plt.axis("off")
# %%
