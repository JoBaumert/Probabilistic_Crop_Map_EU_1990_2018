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
main_path = str(Path(Path(os.path.abspath(__file__)).parents[0]))
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
LUCAS_years=np.array([2006,2009,2012,2015,2018])
all_years=np.arange(1990,2019)
n_prev_years=3
crop_min_freq=20
#%%
#load LUCAS data
LUCAS_preprocessed=pd.read_csv(preprocessed_csv_dir+"LUCAS_preprocessed.csv")
#%%
#translate LUCAS crop codes to DGPCM crop codes
crop_delineation=pd.read_excel(parameter_path+"DGPCM_crop_delineation.xlsx")
crop_delineation=crop_delineation[["LUCAS_code","DGPCM_code"]].drop_duplicates()
crop_delineation.rename(columns={"LUCAS_code":"lc1"},inplace=True)
#%%
LUCAS_preprocessed=pd.merge(LUCAS_preprocessed,crop_delineation,how="left",on="lc1")
LUCAS_preprocessed.dropna(subset="DGPCM_code",inplace=True)
#%%
#get transformed coordinates of LUCAS points to look up in raster files
geometry=gpd.points_from_xy(x=LUCAS_preprocessed.th_long,y=LUCAS_preprocessed.th_lat)
geometry_3035_gdf=gpd.GeoDataFrame({"geometry":geometry},crs="epsg:4326").to_crs("epsg:3035")

#%%
LUCAS_relevant=LUCAS_preprocessed[["id","nuts0","DGPCM_code","year"]]

x=np.array(geometry_3035_gdf.geometry.x)
y=np.array(geometry_3035_gdf.geometry.y)
reference_grid=rio.open(preprocessed_raster_dir+"nuts_2003.tif")
reference_bounds=np.array(reference_grid.bounds)
valid_positions=np.where((x>=reference_bounds[0])&(x<=reference_bounds[2])&
         (y>=reference_bounds[1])&(y<=reference_bounds[3]))

LUCAS_relevant=LUCAS_relevant.iloc[valid_positions[0]]
x_valid=x[valid_positions]
y_valid=y[valid_positions]
#%%
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

#%%
"""PREPROCESS VARYING VARIABLES (calculate 3-year average)"""
for var in np.unique(varying_var_names):
    print(var)
    for year in all_years:
        print(year)
        data_list=[]
        for y in np.arange(n_prev_years):
            data_list.append(rio.open(GEE_data_path+var+"_"+str(year-y-1)+".tif").read())
        mean_var_array=np.array(data_list).squeeze().mean(axis=0)

        with rio.open(preprocessed_raster_dir+var+"_"+str(year)+"_mean.tif", 'w',
                    width=int(reference_grid.shape[1]),height=int(reference_grid.shape[0]),
                    transform=reference_grid.transform,count=1,dtype=rio.float32,crs="EPSG:3035") as dst:
            dst.write(np.expand_dims(mean_var_array,axis=0))

# %%
"""GET STATIC VARIABLES AT LUCAS POSITIONS"""
for var in static_var_names:
    print(var)
    values=[]
    with rio.open(GEE_data_path+var+".tif") as src:

        for val in src.sample(list(zip(x_valid,y_valid))): 
            values.append(val)

    LUCAS_relevant[var]=np.array(values).astype(float)
#%%
"""GET VARYING VARIABLES AT LUCAS POSITIONS"""
for var in np.unique(varying_var_names):
    print(var)
    values_array=np.repeat(-999,len(LUCAS_relevant)).astype(float)
    for year in LUCAS_years:
        year_obs_index=np.where(LUCAS_relevant["year"]==year)[0]
        values=[]
        with rio.open(preprocessed_raster_dir+var+"_"+str(year)+"_mean.tif") as src:
            for val in src.sample(list(zip(x_valid[year_obs_index],y_valid[year_obs_index]))):
                values.append(val)
        values_array[year_obs_index]=np.array(values).flatten()
    LUCAS_relevant[var]=values_array
#%%
"""SAVE LUCAS WITH VARIABLES"""
LUCAS_relevant.dropna(inplace=True)
LUCAS_relevant.to_csv(preprocessed_csv_dir+"LUCAS_with_covariates.csv")
#%%
try:
    len(LUCAS_relevant)#if already loaded
except:
    print("load LUCAS data with covariates...")
    LUCAS_relevant=pd.read_csv(preprocessed_csv_dir+"LUCAS_with_covariates.csv")
#%%
"""SCALE VARIABLES FOR EACH COUNTRY INDIVIDUALLY"""
country="DE"
#%%
selected_df=LUCAS_relevant[LUCAS_relevant["nuts0"]==country].drop(["id","nuts0","year"],axis=1)
crop,counts=np.unique(selected_df["DGPCM_code"],return_counts=True)
relevant_crops=crop[np.where(counts>=crop_min_freq)[0]]
#discard crops for which insufficient observations exist
selected_df=selected_df[selected_df["DGPCM_code"].isin(relevant_crops)]
# %%
data=np.array(selected_df.iloc[:,1:])
scaler=StandardScaler()
scales=scaler.fit(data)
scaler_dict={"variable":np.array(selected_df.columns)[1:],
             "mean":scales.mean_,
             "std":np.sqrt(scales.var_)}

scaler_df=pd.DataFrame(scaler_dict)
scaled_X=scales.transform(data)
#%%
scaler_df.to_csv(resulting_parameters_dir+"scaler_"+country+".csv")
# %%
regression_scaled_df=pd.DataFrame(scaled_X,columns=np.array(selected_df.columns)[1:])
regression_scaled_df["y"]=np.array(selected_df["DGPCM_code"])
regression_scaled_df.sort_values("y",inplace=True)

# %%
#add a constant 
regression_scaled_df=sm.add_constant(regression_scaled_df)
sel_features=np.array(regression_scaled_df.columns)[:-1]
# %%
print("logistic regression starts...")
"""RUN LOGISTIC REGRESSION"""
logit_model=sm.MNLogit(regression_scaled_df['y'],regression_scaled_df[sel_features])
result=logit_model.fit(method='lbfgs',maxiter=5000)
# %%
multinom_params=result.params
multinom_covm=result.cov_params().reset_index()
# %%
multinom_params.columns=sorted(regression_scaled_df['y'].value_counts().keys())[1:]
# %%
"""EXPORT DATA  """
print("export results...")
multinom_params.to_excel(resulting_parameters_dir+"multinomial_logit_"+country+"_statsmodel_params_obsthreshold"+str(crop_min_freq)+".xlsx")
multinom_covm.to_excel(resulting_parameters_dir+"multinomial_logit_"+country+"_statsmodel_covariance_obsthreshold"+str(crop_min_freq)+".xlsx")

# %%
test=pd.read_excel(resulting_parameters_dir+"multinomial_logit_"+country+"_statsmodel_params_obsthreshold"+str(crop_min_freq)+".xlsx")
# %%
test
# %%
