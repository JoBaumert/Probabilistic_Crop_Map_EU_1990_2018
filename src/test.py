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
test=rio.open("/home/baumert/fdiexchange/baumert/DGPCM_19902020/Data/data/results/multi_band_raster/EU_crop_map/EU_expected_crop_shares_2010.tif").read()
# %%
test[0][np.where(test[0]>0.0)].sum()
# %%
plt.hist(test[0][np.where(test[0]>0)].flatten(),bins=100)
#%%
test[0][2000][2030]
# %%
AT=rio.open("/home/baumert/fdiexchange/baumert/DGPCM_19902020/Data/data/results/multi_band_raster/simulted_crop_shares/AT/AT2010_simulated_cropshare_int.tif").read()
#%%
AT.shape
#%%
show(AT[0])
# %%
weights=rio.open(preprocessed_raster_dir+"cellweight_raster_allyears.tif").read()
# %%
nuts_indices=rio.open(preprocessed_raster_dir+"nuts_indices_relevant_allyears.tif").read()
index_dictionary=pd.read_csv(preprocessed_csv_dir+"uaa_calculated_allyears.csv")
# %%
country="AT"
year=2000
indices=np.array(index_dictionary[(index_dictionary["year"]==year)&(index_dictionary["country"]==country)]["index"])
# %%
locs=np.where(np.isin(nuts_indices[np.where(all_years==year)[0][0]],indices))
#%%
plt.hist(weights[np.where(all_years==year)[0][0]][locs])
# %%
plt.hist(AT[7].flatten())
# %%
test[0][np.where(test[0]>=0)].mean()
# %%
lucas=pd.read_csv(preprocessed_csv_dir+"LUCAS_with_covariates.csv")
# %%
c,f=np.unique(lucas["nuts0"],return_counts=True)
# %%
c[np.argsort(f)]
# %%
test_2018=rio.open("/home/baumert/fdiexchange/baumert/DGPCM_19902020/Data/data/results/multi_band_raster/EU_crop_map/EU_expected_crop_shares_2018.tif").read()
# %%
test_2018_old=rio.open("/home/baumert/fdiexchange/baumert/project1/Data/Results/Simulated_consistent_crop_shares/EU/expected_crop_share_entire_EU_2018.tif").read()
# %%
test_2018_old
# %%
show(test_2018_old[6])
# %%
show(test_2018[6])
# %%
c1,c2=21,24
indices=np.where(test_2018[0]>0)
plt.scatter(test_2018[c1][indices].flatten(),y=test_2018_old[c2][indices].flatten(),s=0.001)
# %%
show(test_2018_old[24])
# %%
show(test_2018[21])
# %%
