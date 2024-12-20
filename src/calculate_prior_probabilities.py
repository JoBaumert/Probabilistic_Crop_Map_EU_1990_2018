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
#%%
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
resulting_parameters_dir=results_dir+"/csv/estimation_parameters_and_scalers/"
os.makedirs(resulting_parameters_dir, exist_ok=True)
parameter_path=data_main_path+"/delineation_and_parameters/"
user_parameter_path=parameter_path+"user_parameters.xlsx"
GEE_data_path=raw_dir+"/GEE/"
# %%
def parameter_preparation(
    params,
    covariance_matrix,
    nofcrops,
    n_of_rand=10000,
    mean_on_first_pos=True,
    insert_zeros_refcrop=True,
):
    means_flattened = np.transpose(params.iloc[:, 1:]).to_numpy().flatten()

    randomly_selected_params = np.random.multivariate_normal(
        means_flattened, covariance_matrix.iloc[2:, 3:], n_of_rand
    )
    if mean_on_first_pos:
        # Set mean as position 0
        randomly_selected_params = np.insert(
            randomly_selected_params, 0, means_flattened, axis=0
        )
    if insert_zeros_refcrop:
        # insert 0s for the reference crop 
        randomly_selected_params = randomly_selected_params.reshape(
            (randomly_selected_params.shape[0], nofcrops - 1, -1)
        )
        randomly_selected_params = np.insert(
            randomly_selected_params,
            0,
            np.zeros(randomly_selected_params.shape[2]),
            axis=1,
        )
    return randomly_selected_params


def get_cell_predictions(X_scaled, parameters):
    latentvars = np.matmul(np.transpose(X_scaled), parameters)
    """LIMIT VALUES TO AVOID INFINITY VALUES"""
    latentvars=np.where(latentvars>40,40,latentvars)
    latentvars=np.where(latentvars<-40,-40,latentvars)
    latentvars_exp = np.exp(latentvars)
    latentvars_exp_sum = latentvars_exp.sum(axis=1)
    p = (latentvars_exp.transpose() / latentvars_exp_sum).transpose()
    return p


def probability_calculation(
    param_matrix, X, nofcrops, nofcells, sample_params=False, nofreps=1
):
    if not sample_params:
        nofreps = 1
    all_probas = []
    for p, param_set in enumerate(param_matrix[:nofreps]):
        probas_cell = get_cell_predictions(
            X, param_set.reshape(nofcrops, -1).transpose()
        )
        probas_cells_reshaped = probas_cell.reshape(nofcells, nofcrops)
        all_probas.append(probas_cells_reshaped)

    return np.array(all_probas)

def get_neighbor_average(matrix,radius, row_number, column_number):
     neighbor_values= [[matrix[i][j] if  i >= 0 and i < len(matrix) and j >= 0 and j < len(matrix[0]) else 0
                for j in range(column_number-1-radius, column_number+radius)]
                    for i in range(row_number-1-radius, row_number+radius)]
     return np.nanmean(neighbor_values)
#%%
countries=np.array(pd.read_excel(user_parameter_path,sheet_name="countries")["country_code"])
parameter_training_countries=pd.read_excel(user_parameter_path,sheet_name="parameter_training_countries")
#%%

all_years=np.arange(1990,2019)
n_prev_years=3
crop_min_freq=20

#%%
if __name__ == "__main__":
    #information required for all countries and years
    nuts_indices=rio.open(preprocessed_raster_dir+"nuts_indices_relevant_allyears.tif").read()
    index_dictionary=pd.read_csv(preprocessed_csv_dir+"uaa_calculated_allyears.csv")
    lucas_with_covariates=pd.read_csv(preprocessed_csv_dir+"LUCAS_with_covariates.csv")

    for year in all_years:
        #year=2000
        print(year)

        #get all file names and check if variables are static or not
        varying_var_names=[]
        static_var_names=[]
        for f,filename in enumerate(os.listdir(GEE_data_path)):
            if (filename[:5]!="LUCAS")&(filename[:6]!="CORINE"):
                try:
                    int(filename[-8:-4])
                    varying_var_names.append(filename[:-9])
                    
                except:
                    static_var_names.append(filename[:-4])

        all_vars=[]
        all_var_names=[]
        for varname in static_var_names:
            print(varname)
            all_vars.append(rio.open(GEE_data_path+varname+".tif").read().squeeze())
            all_var_names.append(varname)

        for varname in np.unique(varying_var_names):
            print(varname)
            vars_before_mean=[]
            for y in np.arange(n_prev_years):
                vars_before_mean.append(rio.open(GEE_data_path+varname+"_"+str(year-y-1)+".tif").read())
            all_vars.append(np.array(vars_before_mean).squeeze().mean(axis=0))
            all_var_names.append(varname)
                
        all_vars=np.array(all_vars)

        for country in countries:
            print(country)
            #load estimated parameters
            for filename in os.listdir(resulting_parameters_dir):
                if filename[:-20]== "multinomial_logit_"+country+"_statsmodel_params":
                    parameters=pd.read_excel(resulting_parameters_dir+filename)
                elif filename[:-20]== "multinomial_logit_"+country+"_statsmodel_covariance":
                    covariance=pd.read_excel(resulting_parameters_dir+filename)
                elif filename=="scaler_"+country+".csv":
                    scaler=pd.read_csv(resulting_parameters_dir+filename)


            year_specific_indices=np.array(index_dictionary[(index_dictionary["country"]==country)&(index_dictionary["year"]==year)]["index"])
            capri_codes=np.array(index_dictionary[(index_dictionary["country"]==country)&(index_dictionary["year"]==year)]["CAPRI_code"])
            region_indices=[]
            for region in year_specific_indices:
                region_indices.append(np.where(np.isin(nuts_indices[np.where(all_years==year)[0][0]],region)))
            country_on_map=np.where(np.isin(nuts_indices[np.where(all_years==year)[0][0]],year_specific_indices),1,0)
            #show(country_on_map)
            country_on_map_indices=np.where(country_on_map==1)
         #%%
            """SCALE THE EXPLANATORY VARIABLES FOR COUNTRY"""
            
            #for some variables there are missing values for some cells (mainly climate variables which originally have a resolution of 12km and only exist for land,
            #so coastal areas are sometimes not completely covered). We replace missing values with the average of the neighbors
            corrected_vars=all_vars.copy()
            for v,var in enumerate(all_vars):
                missing_values= np.where(np.isnan(var[country_on_map_indices]))[0].shape[0]
                if missing_values>0:
                    for i in np.where(np.isnan(var[country_on_map_indices]))[0]:
                        new_value=get_neighbor_average(
                        var,50,country_on_map_indices[0][i],country_on_map_indices[1][i]
                        )
                        #sometimes neighbor distance of 50km is not wide enough (e.g., small islands), then increase distance gradually
                        if np.isnan(new_value):
                            invalid=True
                            while invalid:
                                distance=100
                                new_value=get_neighbor_average(
                                var,distance,country_on_map_indices[0][i],country_on_map_indices[1][i]
                                )
                                distance+=50
                                if not np.isnan(new_value):
                                    invalid=False
                        corrected_vars[v][country_on_map_indices[0][i]][country_on_map_indices[1][i]]=new_value
                    print(str(missing_values)+" missing values for "+all_var_names[v]+" replaced...")

            
            for var in corrected_vars:
                print(np.where(np.isnan(var[country_on_map_indices]))[0].shape[0])
            
            #parameter training country may be different from considered country if considered country is very small
            lucas_country=parameter_training_countries[parameter_training_countries["country"]==country]["training_country"].iloc[0]
            crops,counts=np.unique(lucas_with_covariates[lucas_with_covariates["nuts0"]==lucas_country]["DGPCM_code"],return_counts=True)
            considered_crops_country=np.sort(crops[np.where(counts>=crop_min_freq)])

            randomly_selected_params=parameter_preparation(parameters,covariance,nofcrops=len(considered_crops_country))
            #save crops considered in a country and year
            os.makedirs(prior_proba_output_dir+str(year)+"/", exist_ok=True)
            np.save(prior_proba_output_dir+str(year)+"/"+country+"_"+str(year)+"_considered_crops.npy",considered_crops_country)
            for c,capri_code in enumerate(capri_codes):
                
                a=corrected_vars.transpose(1,2,0)[region_indices[c]]
                rescaled_vars_region=((a-np.array(scaler["mean"]))*(1/np.array(scaler["std"])))
                #insert ones as the constant
                rescaled_vars_region=np.insert(rescaled_vars_region,0,1,axis=1)

                all_probas=probability_calculation(
                    randomly_selected_params,
                    rescaled_vars_region.T,
                    len(considered_crops_country),
                    sample_params=True,
                    nofcells=rescaled_vars_region.shape[0],
                    nofreps=10)

                #save as numpy array
                print("save prior probabilities for region "+capri_code)
                np.save(prior_proba_output_dir+str(year)+"/"+capri_code+"_"+str(year)+".npy",all_probas)

# %%

# %%
