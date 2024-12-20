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

def generate_random_results(proba,n_of_fields_country,n_samples_aleatoric):
    proba_corrected=np.where(proba==0,0.0000001,proba)
    proba_corrected=(proba_corrected*(1/(proba_corrected.sum(axis=0)+0.000001))).T

    random_results=np.array(
        [
            np.random.multinomial(n_of_fields_country[i],proba_corrected[i],100)
            /n_of_fields_country[i]
            for i in range(proba_corrected.shape[0])
        ]).T

    deviation=abs(1-random_results.sum(axis=2)/(proba_corrected.sum(axis=0).reshape(-1,1)))
    order=np.argsort(deviation.sum(axis=0))

    random_results_selected=random_results.transpose(1,0,2)[order][:n_samples_aleatoric]
    del random_results
    gc.collect()
    return random_results_selected,proba_corrected,deviation.T[order][:n_samples_aleatoric]

#%%

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
    for year in all_years:
        
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
            band_list.append("weight")
            
            #write number of fields per cell as second band
            n_of_fields_country=n_of_fields[np.where(all_years==year)[0][0]][index_europe_map]
            result_country_raster[1][index_country_map]=n_of_fields_country
            band_list.append("n_of_fields")
            
            selected_posterior_probas=result_raster_year[index_europe_map]

            del index_europe_map
            del country_raster
            del result_raster_year
            gc.collect()
            
            for beta in range(n_samples_epistemic):
                print("beta: "+str(beta))
                proba=selected_posterior_probas.transpose(1,0,2)[beta].T
                print("sample...")
                random_results_selected,proba_corrected,deviation=generate_random_results(proba,n_of_fields_country,n_samples_aleatoric)
                
                random_results_selected=random_results_selected.transpose(1,0,2).reshape(n_samples_aleatoric*n_considered_crops,-1)
                
                if beta==0:
                    for c in range(n_considered_crops):
                        result_country_raster[c+2][index_country_map]=proba_corrected.T[c]
                        band_list.append("expected_share_"+considered_crops[c])

                del proba
                gc.collect()
                
                for i in range(n_considered_crops*n_samples_aleatoric):
                    result_country_raster[n_considered_crops+2+beta*(n_considered_crops*n_samples_aleatoric)+i][index_country_map]=random_results_selected[i]
                    band_list.append(str(considered_crops[i//n_samples_aleatoric]+"_"+str(beta)+str(i%n_samples_aleatoric)))

                
                del random_results_selected
                del proba_corrected
                gc.collect


            #crop="GRAS"
            #np.where(np.char.find(np.array(band_list),crop)>=0)
            #show(result_country_raster[831])
            
            del weights_country
            gc.collect()
            
            factor=np.repeat(1000,result_country_raster.shape[0])
            factor[1]=1
            
            refactored_data=factor.reshape(-1,1,1)*result_country_raster
            del result_country_raster
            gc.collect()

            refactored_data=refactored_data.round()
        
            print("save raster files...")
            Path(simulated_cropshares_dir+country+"/").mkdir(parents=True, exist_ok=True)
            with rio.open(simulated_cropshares_dir+country+"/"+country+str(year)+"_simulated_cropshare_"+str(n_samples_aleatoric)+"reps_int.tif", 'w',
                        width=int(width),height=int(height),transform=transform,count=refactored_data.shape[0],dtype=rio.int16,crs="EPSG:3035") as dst:
                dst.write(refactored_data.astype(rio.int16))


            band_df=pd.DataFrame({"band":np.arange(len(band_list)),"name":np.array(band_list)})

            band_df.to_csv(
                simulated_cropshares_dir+country+"/"+country+str(year)+"simulated_cropshare_"+str(n_samples_aleatoric)+"reps_bands.csv"
            )

            del refactored_data
            del factor
            
            del band_df
            gc.collect()

#%%