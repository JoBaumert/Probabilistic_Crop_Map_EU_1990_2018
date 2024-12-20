#%%
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import os
import rasterio as rio
from rasterio.plot import show
# %%
# user settings
# when calculating the number of fields per cell, in many cases we get only 1 or 2 fields (as the agshare is in some cells very small)
# this would mean that the stochastic uncertainty is very big. We therefore can impose that a cell should have min_n_of_fields fields, at least, and for all cells
# with a lower calculated number of fields min_n_of_fields is instead used.
# to allow all number of fields, just set min_n_of_fields to 0
min_n_of_fields = 5
# note that this will be overwritten, if min_n_of_fields is defined in the user specifications
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
posterior_proba_output_dir=results_dir+"/numpy_arrays/posterior_crop_probas/"
os.makedirs(posterior_proba_output_dir,exist_ok=True)
resulting_parameters_dir=results_dir+"/csv/estimation_parameters_and_scalers/"
os.makedirs(resulting_parameters_dir, exist_ok=True)
parameter_path=data_main_path+"/delineation_and_parameters/"
user_parameter_path=parameter_path+"user_parameters.xlsx"
output_dir=results_dir+"multi_band_raster/"
os.makedirs(output_dir,exist_ok=True)
#%%
# input paths
LUCAS_years=np.array([2006,2009,2012,2015,2018])
all_years=np.arange(1990,2019)

#%%
#load LUCAS data
LUCAS_preprocessed=pd.read_csv(preprocessed_csv_dir+"LUCAS_preprocessed.csv")
#%%
nuts_region_data=pd.read_csv(preprocessed_csv_dir+"preprocessed_CAPREG_step2.csv")
uaa_calculated=pd.read_csv(preprocessed_csv_dir+"uaa_calculated_allyears.csv")
# %%
all_nuts_regions=nuts_region_data["NUTS_ID"].unique()
#%%
if __name__ == "__main__":
    # import parameters

    LUCAS_fieldsize_conversion = pd.read_excel(
        user_parameter_path, sheet_name="LUCAS_fieldsize"
    )


    # import data
    #%%
    print("calculate field size...")
    print("assumed minimum number of fields per cell: "+ str(min_n_of_fields))
    #%%
    # load data on NUTS regions in a year
   # NUTS_data = pd.read_csv(nuts_input_path)
   # NUTS_data=NUTS_data[(NUTS_data["CNTR_CODE"].isin(country_codes_relevant))& (NUTS_data["year"].isin(selected_years))]
   # LUCAS_preprocessed = pd.read_csv(LUCAS_preprocessed_path)
    #%%
    LUCAS_parcel = LUCAS_preprocessed[
        ["nuts0", "nuts1", "nuts2", "nuts3", "parcel_area_ha"]
    ]
    LUCAS_parcel.dropna(inplace=True)
    
    #%%
    converted_parcel_size_array = np.ndarray(len(LUCAS_parcel))
    for i, size in enumerate(np.array(LUCAS_fieldsize_conversion["LUCAS"])):
        converted_parcel_size_array[
            np.where(np.array(LUCAS_parcel["parcel_area_ha"]) == size)[0]
        ] = LUCAS_fieldsize_conversion["field_size_in_ha"].iloc[i]
    LUCAS_parcel["converted_parcel_size"] = converted_parcel_size_array

    #%%
    # use the median field size for the nuts3 region. if this is nan (because there is no data), use the median of the higher NUTS level for which data is available

    median_field_size_dict = {
        "nuts_reg": [],
        "median_field_size": [],
    }
    for reg in all_nuts_regions:
        level=np.char.str_len(reg)-2

        median_field_size_dict["nuts_reg"].append(reg)
        median_selected_reg = LUCAS_parcel[LUCAS_parcel[f"nuts{level}"] == reg][
            "converted_parcel_size"
        ].median()
        if (np.isnan(median_selected_reg))&(level!=0):
            invalid=True
            while invalid:
                level=level-1
                median_selected_reg = LUCAS_parcel[
                LUCAS_parcel[f"nuts{level}"] == str(np.array(reg).astype(f"U{level+2}"))
            ]["converted_parcel_size"].median()
                if not np.isnan(median_selected_reg): 
                    invalid=False
                if level==0:
                    invalid=False
        median_field_size_dict["median_field_size"].append(median_selected_reg)

        median_field_size_df = pd.DataFrame(median_field_size_dict)
        median_field_size_df["median_n_of_fields_per_km2"] = 100 / np.array(
            median_field_size_df["median_field_size"]
        )

    #%%
    """load UAA data"""
    uaa_raster_allyears_raw=rio.open(preprocessed_raster_dir+"uaa_raster_allyears.tif")
    transform=uaa_raster_allyears_raw.transform
    uaa_raster_allyears=uaa_raster_allyears_raw.read()
    #%%
    nuts_indices_relevant=rio.open(preprocessed_raster_dir+"nuts_indices_relevant_allyears.tif").read()
    #%%
    fieldsize_raster=np.ndarray(uaa_raster_allyears.shape)
    #%%
    for y,year in enumerate(all_years):
        print(year)
        selection=uaa_calculated[uaa_calculated["year"]==year]
        for capri_reg in np.unique(selection["CAPRI_code"]):
            nuts_code=str(np.unique(
                nuts_region_data[(nuts_region_data["year"]==year)&(nuts_region_data["CAPRI_code"]==capri_reg)]["NUTS_ID"])[0])
            level=np.char.str_len(nuts_code)-2
            n_fields_reg=float(median_field_size_df[median_field_size_df["nuts_reg"]==nuts_code]["median_n_of_fields_per_km2"])
            """
            if (np.isnan(n_fields_reg))&(level!=0):
                invalid=True
                while invalid:
                    level=level-1
                    new_reg=str(np.array(nuts_code).astype(f"U{level+2}"))
                    float(median_field_size_df[median_field_size_df["nuts_reg"]==new_reg]["median_n_of_fields_per_km2"])
                    if not np.isnan(median_selected_reg): 
                        invalid=False
                    if level==0:
                        invalid=False
            """

            index=selection[selection["CAPRI_code"]==capri_reg]["index"].iloc[0]
            region_position=np.where(nuts_indices_relevant[y]==index)
            fieldsize_raster[y][region_position]=uaa_raster_allyears[y][region_position]*n_fields_reg

    #%%
    # export data for country
    print(f"export data")
    with rio.open(preprocessed_raster_dir+"n_of_fields_raster_allyears.tif", 'w',
        width=int(fieldsize_raster.shape[2]),height=int(fieldsize_raster.shape[1]),
        transform=transform,count=fieldsize_raster.shape[0],dtype=rio.float32,crs="EPSG:3035") as dst:
        dst.write(fieldsize_raster)

# %%
