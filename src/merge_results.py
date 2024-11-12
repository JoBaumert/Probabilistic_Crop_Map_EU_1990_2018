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
    return random_results_selected,proba_corrected,deviation.T[order][:n_samples_aleatoric]

#%%
year=2000
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
for country in CAPREG_data["country"].unique()[:1]:
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
#%%

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
#%%
result_country_raster=np.zeros((int(
    (n_samples_epistemic*n_samples_aleatoric+1)*n_considered_crops+2),int(height),int(width)))

#%%
index_europe_map=np.where(country_raster==1)

width_values=(index_europe_map[1]-west_rel/1000).astype(int)
height_values=(index_europe_map[0]-north_rel/1000).astype(int)
index_country_map=(height_values,width_values)
#%%
band_list=[]

#write weight as first band
weights_country=weights[np.where(all_years==year)[0][0]][index_europe_map]
result_country_raster[0][index_country_map]=weights_country
band_list.append("weight")
#%%
#write number of fields per cell as second band
n_of_fields_country=n_of_fields[np.where(all_years==year)[0][0]][index_europe_map]
result_country_raster[1][index_country_map]=n_of_fields_country
band_list.append("n_of_fields")
#%%
selected_posterior_probas=result_raster_year[index_europe_map]
#%%
for beta in range(n_samples_epistemic):
    print("beta: "+str(beta))
    proba=selected_posterior_probas.transpose(1,0,2)[beta].T
    random_results_selected,proba_corrected,deviation=generate_random_results(proba,n_of_fields_country,n_samples_aleatoric)
    
    random_results_selected=random_results_selected.transpose(1,0,2).reshape(n_samples_aleatoric*n_considered_crops,-1)
    
    if beta==0:
        for c in range(n_considered_crops):
            result_country_raster[c+2][index_country_map]=proba_corrected.T[c]
            band_list.append("expected_share_"+considered_crops[c])

    for i in range(n_considered_crops*n_samples_aleatoric):
        result_country_raster[n_considered_crops+2+beta*(n_considered_crops*n_samples_aleatoric)+i][index_country_map]=random_results_selected[i]
        band_list.append(str(considered_crops[i//n_samples_aleatoric]+"_"+str(beta)+str(i%n_samples_aleatoric)))

#%%
crop="VINY"
np.where(np.char.find(np.array(band_list),crop)>=0)

#%%
i=11
i%n_samples_aleatoric
#%%
show(result_country_raster[26])
#%%
result_country_raster.transpose(1,2,0)[index_country_map].T[np.arange(2)].shape

#%%
random_results_selected.shape
#%%
show(result_raster_year.T[5][0].T)
# %%

np.where(considered_crops=="GRAS")

#%%




#%%
if __name__ == "__main__":
    for country in country_codes_relevant:
        
  

            crops=np.unique(posterior_probas["crop"])
            betas=np.unique(posterior_probas["beta"])

            band_list=[]
            #add cellweight as first band
            posterior_probas.sort_values(by=["NOFORIGIN","EOFORIGIN"],ascending=[False,True],inplace=True)
            resulting_matrix=np.ndarray((crops.shape[0]*postsampling_reps*betas.shape[0]+crops.shape[0]+2,int(height),int(width)))
            print("rasterizing cell weight...")
            selection=posterior_probas.drop_duplicates(["CELLCODE"])
            geom_value = ((geom,value) for geom, value in zip(selection.geometry, selection.weight))
            rasterized=features.rasterize(
                geom_value,
                out_shape=(int(height),int(width)),
                transform=transform,
                default_value=1 
            )
            #first band is the cellweight
            resulting_matrix[0]=rasterized
            band_list.append("weight")
            #add number of fields per cell as second band
            print("rasterizing number of fields...")
            geom_value = ((geom,value) for geom, value in zip(selection.geometry, selection.n_of_fields_assumed))
            rasterized=features.rasterize(
                geom_value,
                out_shape=(int(height),int(width)),
                transform=transform,
                default_value=1 
            )
            #second band is assumed number of fields per cell (minimum= 5)
            resulting_matrix[1]=rasterized
            #replace a few nan values with mean
            resulting_matrix[1][np.where(np.isnan(resulting_matrix[1]))]=np.nanmean(resulting_matrix[1])

            band_list.append("n_of_fields")
            n_of_fields_array=resulting_matrix[1,np.where(resulting_matrix[0]>0)[0],np.where(resulting_matrix[0]>0)[1]]

            n_fields=resulting_matrix[1].flatten()
            weights_matrix=resulting_matrix[0]

            deviation_matrix=np.ndarray((postsampling_reps*len(betas),len(crops))).astype(float)

            for beta in betas:
                print("random sampling of crop shares for probability sample "+str(beta))
                helper_matrix=np.ndarray((crops.shape[0],int(height),int(width)))
                c=0
                for crop in crops:
                    print(crop)
                    selection=posterior_probas[(posterior_probas["crop"]==crop)&(posterior_probas["beta"]==beta)]
                    result_array=np.zeros(int(height*width),dtype=np.float16)
                    result_array[np.where(resulting_matrix[0].flatten()>0)[0]]=np.array(selection.posterior_probability,dtype=np.float16)
                    #add expected shares as bands 2:n_crops-3
                    helper_matrix[c]=result_array.reshape(resulting_matrix[0].shape)
                    c+=1
                    band_list.append(f"expected_share_{crop}")

                proba=helper_matrix
                proba=proba.reshape(len(crops),-1)

                proba=proba.T[np.where(weights_matrix.flatten()>0)].T
                n_fields_relevant=n_fields[np.where(weights_matrix.flatten()>0)]

                crop_expectation_matrix,sampled_matrix,deviation=generate_random_results(proba,n_fields_relevant,postsampling_reps,target_shape=weights_matrix.shape,weights_matrix=weights_matrix,beta=beta)

                if beta==0:
                    resulting_matrix[2:len(crops)+2]=crop_expectation_matrix

                resulting_matrix[2+len(crops)+beta*len(crops)*postsampling_reps:(beta+1)*len(crops)*postsampling_reps+2+len(crops)]=sampled_matrix

                deviation_matrix[beta*postsampling_reps:beta*postsampling_reps+postsampling_reps]=deviation

            
                del proba
                del crop_expectation_matrix
                del helper_matrix
                del sampled_matrix
                del deviation

                gc.collect()
            

            factor=np.repeat(1000,resulting_matrix.shape[0])
            factor[0]=10
            factor[1]=1

            refactored_data=factor.reshape(-1,1,1)*resulting_matrix
            refactored_data=refactored_data.round()

            print("save raster files...")
            Path(Simulated_cropshares_path+country+"/").mkdir(parents=True, exist_ok=True)
            with rio.open(Simulated_cropshares_path+country+"/"+country+str(year)+"simulated_cropshare_"+str(postsampling_reps)+"reps_int.tif", 'w',
                        width=int(width),height=int(height),transform=transform,count=refactored_data.shape[0],dtype=rio.int16,crs="EPSG:3035") as dst:
                dst.write(refactored_data.astype(rio.int16))

            #export csv file with meta data (information about the bands)
            band_list=["weight","n_of_fields"]
            for crop in crops:
                band_list.append("expected_share_"+crop)
            for beta in betas:
                bands=list(np.char.add(
                        np.char.add(
                            np.char.add(
                            np.repeat(crops,postsampling_reps).astype(str),
                            np.repeat("_",len(crops)*postsampling_reps)),
                        np.repeat(str(beta),len(crops)*postsampling_reps)),
                        np.tile(np.arange(postsampling_reps),len(crops)).astype(str)))
                for band in bands:
                    band_list.append(band)
                    

            band_df=pd.DataFrame({"band":np.arange(len(band_list)),"name":np.array(band_list)})

            band_df.to_csv(
                Simulated_cropshares_path+country+"/"+country+str(year)+"simulated_cropshare_"+str(postsampling_reps)+"reps_bands.csv"
            )

            Path(Simulated_cropshares_path+"Deviations/").mkdir(parents=True, exist_ok=True)
            pd.DataFrame(deviation_matrix,columns=crops).to_csv(
                Simulated_cropshares_path+"Deviations/"+country+str(year)+"simulated_cropshare_"+str(postsampling_reps)+"reps_deviations.csv"
            )

#%%
# %%