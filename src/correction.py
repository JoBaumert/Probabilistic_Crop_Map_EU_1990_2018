#%%
import geopandas as gpd
import pandas as pd
import rasterio as rio
from rasterio.plot import show
from rasterio import features
import matplotlib.pyplot as plt
import numpy as np
import os
from pathlib import Path
from shapely.geometry import Point
from rasterio.windows import from_bounds
import gc
# %%
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
# %%


all_years=np.arange(1990,2019)
CAPREG_data=pd.read_csv(preprocessed_csv_dir+"preprocessed_CAPREG_step3.csv")

#%%
considered_crops=np.sort(np.unique(CAPREG_data["DGPCM_crop_code"]))
#%%
n_considered_crops=len(considered_crops)
n_samples_epistemic=10
n_samples_aleatoric=10
#%%
nuts_indices=rio.open(preprocessed_raster_dir+"nuts_indices_relevant_allyears.tif").read()
index_dictionary=pd.read_csv(preprocessed_csv_dir+"uaa_calculated_allyears.csv")
n_of_fields=rio.open(preprocessed_raster_dir+"n_of_fields_raster_allyears.tif").read()
weights=rio.open(preprocessed_raster_dir+"cellweight_raster_allyears.tif").read()
#%%
#get boundaries and transform of Europe map
trans=rio.open(preprocessed_raster_dir+"nuts_indices_relevant_allyears.tif").transform
h=rio.open(preprocessed_raster_dir+"nuts_indices_relevant_allyears.tif").shape[0]
w=rio.open(preprocessed_raster_dir+"nuts_indices_relevant_allyears.tif").shape[1]

west_ref,south_ref,east_ref,north_ref=rio.transform.array_bounds(h,w,trans)

#%%
if __name__ == "__main__":
    for year in all_years[2:]:

        for country in CAPREG_data["country"].unique():

            if (country=="HR")&(year<1995): #no regional data available for croatia before 1995
                continue 
            print(country+" "+str(year))
            result_raster_year=np.ndarray((nuts_indices.shape[1],
                                    nuts_indices.shape[2],
                                    n_samples_epistemic,
                                    n_considered_crops,
                                    ))
            
            country_raster=np.zeros((nuts_indices.shape[1],
                                    nuts_indices.shape[2]))
            regs=CAPREG_data[(CAPREG_data["year"]==year)&(CAPREG_data["country"]==country)]["CAPRI_code"].unique()
            for reg in regs:
                index=index_dictionary[(index_dictionary["CAPRI_code"]==reg)&
                        (index_dictionary["year"]==year)]["index"].iloc[0]
                data=np.load(
                    posterior_proba_output_dir+str(year)+"/"+reg+"_"+str(year)+".npy"
                )
                result_raster_year[np.where(nuts_indices[np.where(all_years==year)[0][0]]==index)]=data.transpose(1,0,2)
                country_raster[np.where(nuts_indices[np.where(all_years==year)[0][0]]==index)]=1

            south_rel=np.where(country_raster==1)[0].max()*1000
            north_rel=np.where(country_raster==1)[0].min()*1000
            west_rel=np.where(country_raster==1)[1].min()*1000
            east_rel=np.where(country_raster==1)[1].max()*1000

            north=north_ref-north_rel
            south=north_ref-south_rel
            west=west_ref+west_rel
            east=west_ref+east_rel
            height=(north-south)/1000+1
            width=(east-west)/1000+1

            transform=rio.transform.from_bounds(west,south,east,north,width,height)

            result_country_raster=np.zeros((int(
                (n_samples_epistemic*n_samples_aleatoric+1)*n_considered_crops+2),int(height),int(width)))

            index_europe_map=np.where(country_raster==1)

            width_values=(index_europe_map[1]-west_rel/1000).astype(int)
            height_values=(index_europe_map[0]-north_rel/1000).astype(int)
            index_country_map=(height_values,width_values)

            band_list=[]

            #write weight as first band
            weights_country=weights[np.where(all_years==year)[0][0]][index_europe_map]
            result_country_raster[0][index_country_map]=weights_country

            #import original file
            orig=rio.open("/home/baumert/fdiexchange/baumert/DGPCM_19902020/Data/data/results/multi_band_raster/simulted_crop_shares_old/"+country+"/"+country+str(year)+"_simulated_cropshare_10reps_int.tif").read()

            orig[0]=(result_country_raster[0]*1000).round()
            
            print("save raster files...")
            Path(simulated_cropshares_dir+country+"/").mkdir(parents=True, exist_ok=True)
            with rio.open(simulated_cropshares_dir+country+"/"+country+str(year)+"_simulated_cropshare_int.tif", 'w',
                        width=int(width),height=int(height),transform=transform,count=orig.shape[0],dtype=rio.int16,crs="EPSG:3035") as dst:
                dst.write(orig.astype(rio.int16))
# %%
