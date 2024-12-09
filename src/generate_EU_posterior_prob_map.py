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
parameter_path=data_main_path+"/delineation_and_parameters/"
user_parameter_path=parameter_path+"user_parameters.xlsx"
GEE_data_path=raw_dir+"/GEE/"
EU_posterior_map_path=results_dir+"/multi_band_raster/EU_crop_map/"
os.makedirs(EU_posterior_map_path,exist_ok=True)

#%%


#%%
if __name__ == "__main__":
    cellweight=rio.open(preprocessed_raster_dir+"cellweight_raster_allyears.tif").read()
    #%%
    all_years=np.arange(1990,2019)
    n_samples=10

    nuts_indices=rio.open(preprocessed_raster_dir+"nuts_indices_relevant_allyears.tif").read()
    index_dictionary=pd.read_csv(preprocessed_csv_dir+"uaa_calculated_allyears.csv")
    CAPREG_data=pd.read_csv(preprocessed_csv_dir+"preprocessed_CAPREG_step3.csv")
    #
    considered_crops=np.sort(np.unique(CAPREG_data["DGPCM_crop_code"]))
    n_considered_crops=len(considered_crops)

    for year in all_years:
        print(str(year)+"...")
        result_raster_year=np.ndarray((nuts_indices.shape[1],
                                    nuts_indices.shape[2],
                                    n_samples,
                                    n_considered_crops,
                                    ))

        namelist=[]
        for filename in os.listdir(posterior_proba_output_dir+str(year)+"/"):
            if filename[-8:-4]==str(year):
                region=filename[:8]
                namelist.append(region)
                index=index_dictionary[(index_dictionary["CAPRI_code"]==region)&
                        (index_dictionary["year"]==year)]["index"].iloc[0]
                data=np.load(
                    posterior_proba_output_dir+str(year)+"/"+filename
                )
                result_raster_year[np.where(nuts_indices[np.where(all_years==year)[0][0]]==index)]=data.transpose(1,0,2)

        result_raster_year=result_raster_year.transpose(3,2,0,1)
        
        cellweight[np.where(all_years==year)[0]][0].shape
      
        #%%
        #crop="SWHE"
        #np.where(considered_crops==crop)[0][0]
        #%%
        expected_crop_share=result_raster_year[:,0,:,:]
        expected_crop_share=np.insert(expected_crop_share,0,cellweight[np.where(all_years==year)[0]][0],axis=0)
        #show(result_raster_year[12][1])
        #%%
        expected_crop_share[0]=np.where(expected_crop_share[0]>=0,expected_crop_share[0]*100*10,expected_crop_share[0])
        #%%
        factor=np.repeat(1000,n_considered_crops)
        factor=np.insert(factor,0,1)
        
        expected_crop_share=expected_crop_share*factor[:,None,None]
        #%%
        plt.hist(expected_crop_share[1][np.where(expected_crop_share[1]>0)],bins=100)
        #%%
        
        transform=rio.open(preprocessed_raster_dir+"nuts_indices_relevant_allyears.tif").transform
        
        with rio.open(EU_posterior_map_path+"EU_expected_crop_shares_"+str(year)+".tif", 'w',
                    width=int(expected_crop_share.shape[2]),height=int(expected_crop_share.shape[1]),transform=transform,
                    count=expected_crop_share.shape[0],dtype=rio.int16,crs="EPSG:3035") as dst:
            dst.write(expected_crop_share.astype(rio.int16))

    
    
    bands=np.insert(considered_crops,0,"weight")
    #pd.DataFrame({"bands":bands}).to_csv(EU_posterior_map_path+"bands.csv",index=None)
    #%%
    
# %%
