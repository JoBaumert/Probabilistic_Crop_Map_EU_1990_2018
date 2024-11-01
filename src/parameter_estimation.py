#%%
import geopandas as gpd
import pandas as pd
import rasterio as rio
from rasterio.plot import show
from rasterio import features
import numpy as np
import os
from pathlib import Path
from shapely.geometry import Point
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
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
resulting_parameters_dir=results_dir+"/csv/estimation_parameters_and_scalers/"
os.makedirs(resulting_parameters_dir, exist_ok=True)
parameter_path=data_main_path+"/delineation_and_parameters/"
user_parameter_path=parameter_path+"user_parameters.xlsx"
GEE_data_path=raw_dir+"/GEE/"

#%%

n_prev_years=3
crop_min_freq=20

if __name__ == "__main__":
    print("load LUCAS data with covariates...")
    LUCAS_relevant=pd.read_csv(preprocessed_csv_dir+"LUCAS_with_covariates.csv")

    """SCALE VARIABLES FOR EACH COUNTRY INDIVIDUALLY"""
    parameter_training_countries=pd.read_excel(user_parameter_path,sheet_name="parameter_training_countries")
    all_countries=np.array(parameter_training_countries["country"])
    
    for country in all_countries:
        print(country)
        lucas_country=parameter_training_countries[parameter_training_countries["country"]==country]["training_country"].iloc[0]
        selected_df=LUCAS_relevant[LUCAS_relevant["nuts0"]==lucas_country].drop(["id","nuts0","year"],axis=1)
        crop,counts=np.unique(selected_df["DGPCM_code"],return_counts=True)
        relevant_crops=crop[np.where(counts>=crop_min_freq)[0]]
        #discard crops for which insufficient observations exist
        selected_df=selected_df[selected_df["DGPCM_code"].isin(relevant_crops)]

        data=np.array(selected_df.iloc[:,2:])
        scaler=StandardScaler()
        scales=scaler.fit(data)
        scaler_dict={"variable":np.array(selected_df.columns)[2:],
                    "mean":scales.mean_,
                    "std":np.sqrt(scales.var_)}

        scaler_df=pd.DataFrame(scaler_dict)
        scaled_X=scales.transform(data)
    
        scaler_df.to_csv(resulting_parameters_dir+"scaler_"+country+".csv")
        
        regression_scaled_df=pd.DataFrame(scaled_X,columns=np.array(selected_df.columns)[2:])
        regression_scaled_df["y"]=np.array(selected_df["DGPCM_code"])
        regression_scaled_df.sort_values("y",inplace=True)

        
        #add a constant 
        regression_scaled_df=sm.add_constant(regression_scaled_df)
        sel_features=np.array(regression_scaled_df.columns)[:-1]
        
        print("logistic regression starts...")
        """RUN LOGISTIC REGRESSION"""
        logit_model=sm.MNLogit(regression_scaled_df['y'],regression_scaled_df[sel_features])
        result=logit_model.fit(method='lbfgs',maxiter=5000)
        
        multinom_params=result.params
        multinom_covm=result.cov_params().reset_index()
        
        multinom_params.columns=sorted(regression_scaled_df['y'].value_counts().keys())[1:]
        
        """EXPORT DATA  """
        print("export results...")
        multinom_params.to_excel(resulting_parameters_dir+"multinomial_logit_"+country+"_statsmodel_params_obsthreshold"+str(crop_min_freq)+".xlsx")
        multinom_covm.to_excel(resulting_parameters_dir+"multinomial_logit_"+country+"_statsmodel_covariance_obsthreshold"+str(crop_min_freq)+".xlsx")

    # %%
   # test=pd.read_excel(resulting_parameters_dir+"multinomial_logit_"+country+"_statsmodel_params_obsthreshold"+str(crop_min_freq)+".xlsx")
    # %%
#test
# %%
