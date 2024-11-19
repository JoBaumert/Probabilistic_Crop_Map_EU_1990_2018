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
visualizations_path=results_dir+"/visualizations/"
os.makedirs(visualizations_path,exist_ok=True)
# %%
country="RO"
all_years=np.arange(1990,2019)

n_samples_aleatoric=10

# %%
#%%
year=1990
bands=pd.read_csv(simulated_cropshares_dir+country+"/"+country+str(year)+"simulated_cropshare_"+str(n_samples_aleatoric)+"reps_bands.csv")
crop="SUNF"
np.where(np.char.find(np.array(bands["name"]).astype(str),crop)>=0)


#%%
for year in all_years[:1]:
    year=2012
    print(year)
    test_country=rio.open(simulated_cropshares_dir+country+"/"+country+str(year)+"_simulated_cropshare_"+str(n_samples_aleatoric)+"reps_int.tif").read()
    bins=np.array([0,5,10,15,20,25,30,35,40,45,50,60,1000])
    pota_selected_year=np.digitize(test_country[14],bins=bins)

    pota_selected_year[np.where(test_country[14]==0)]=0
    
    fig, ax = plt.subplots(1, 1, figsize=(7, 7))
    show(pota_selected_year,title=str(year))


# %%
year=2018
test_country=rio.open(simulated_cropshares_dir+country+"/"+country+str(year)+"_simulated_cropshare_"+str(n_samples_aleatoric)+"reps_int.tif").read()
#%%
show(test_country[21])
# %%
d=rio.open(simulated_cropshares_dir+country+"/"+country+str(2005)+"_simulated_cropshare_"+str(n_samples_aleatoric)+"reps_int.tif").read()
# %%
bins=np.array([0,5,10,15,20,25,30,35,40,45,50,60,1000])
pota_1990=np.digitize(c[14],bins=bins)
pota_1995=np.digitize(test_country[14],bins=bins)
pota_2000=np.digitize(b[14],bins=bins)
pota_2005=np.digitize(d[14],bins=bins)
# %%
np.unique(inds)
# %%
show(pota_1990)
# %%
show(pota_1995)
# %%
show(pota_2000)
#%%
show(pota_2005)
# %%
np.quantile(c[14],0.9)

# %%
test=pd.read_csv(preprocessed_csv_dir+"preprocessed_CAPREG_step3.csv")
# %%
test
# %%
uaa=pd.read_csv(preprocessed_csv_dir+"uaa_calculated_allyears.csv")
# %%
nuts_regions_dictionary=pd.read_csv(preprocessed_csv_dir+"nuts_regions_dictionary.csv")
# %%
nuts_regions_dictionary
# %%
uaa
# %%
mapping=pd.read_csv(parameter_path+"CAPRI_Eurostat_NUTS_mapping.csv",header=None)
# %%
mapping.rename(columns={0:"CAPRI_code",1:"NUTS_code"},inplace=True)
# %%
mapping=mapping.drop(2,axis=1)
# %%
mapping.to_csv(parameter_path+"CAPRI_Eurostat_NUTS_mapping2.csv",index=None)
# %%
test["DGPCM_crop_code"].unique()
# %%
