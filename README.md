# Probabilistic Crop Map EU 1990-2018 
### (by Josef Baumert, Thomas Heckelei, and Hugo Storm)
![alt text](https://github.com/JoBaumert/Probabilistic_Crop_Map_EU_1990_2018/blob/master/map.png) <br>
All generated maps are available for download from here: https://zenodo.org/records/14409498
The code for reproduction is contained in the directory "src". The folder "delineation_and_parameters" contains some rather small excel files that describe how crop types are matched between different data sources (e.g., LUCAS and Eurostat) and predefine some hyperparameters. The following is a guideline for the reproduction of the maps.

## Step 1: Installation of Dependencies
### option 1) using docker container
The repository is set up to run in a Docker container. Pull the repository and open it in VS Code with the Remote-Containers extension. This requires that you have a) a Docker Engine installed (https://docs.docker.com/engine/install/) and b) the VS Code Dev-Containers extension installed (Extension identifier: ```ms-vscode-remote.remote-containers ```). <br>
With this in place follow the instructions to create the development container in VS Code:
1) Clone the repository: ```git clone https://github.com/JoBaumert/Probabilistic_Crop_Map_EU_1990_2018.git```
2) Open the clone folder in VS Code and hit ```Ctrl+Shift+P``` and select ```remote-Containers: Reopen in Container```
All the necessary dependencies are then automatically installed in the Docker container.
### option 2) without docker container
1) Install Pipenv from https://pipenv.pypa.io/en/latest/
2) Clone the repository: ```git clone https://github.com/JoBaumert/Probabilistic_Crop_Map_EU_1990_2018.git```
3) Create pipenv environment by running ```pipenv install --dev```
4) You will have to install jax (+jaxlib) and jaxopt individually according to a procedure that depends on whether you have a GPU or not. See here for a general description of the installation: https://jax.readthedocs.io/en/latest/installation.html. For us (using an NVIDIA GPU and CUDA version 11.4) the following worked to install jax (using this procedure jaxlib is installed automatically) and connect it to the GPU:
```
pip install -U jax[cuda11_cudnn86] -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```
Then, you can install jaxopt:
```
pip install jaxopt==0.8.3
```
If you are not using a GPU then the following should still work:
```
pip install jax==0.4.13 jaxlib==0.4.13 jaxopt==0.8.3
```
```
pipenv shell
pip install numpyro[cuda] -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
   ```
## Step 2: Establish correct folder structure
1) create a folder named "data" where the input and output data will be stored. The user can choose where to locate this directory on the local machine. However, when choosing the location of this directory consider that some input and output files are quite large. To ensure that the python scripts find this folder, you must specify the path to the main data directory with a text file that is stored in the same directory as the code (i.e., in Probabilistic_Crop_Mapping_EU). In the directory "src" you find a file named "data_main_path.txt". Overwrite the text that is currently in it and provide the path to your data directory.
2) move the directory "delineation_and_parameters" with its content into this newly generated "data" directory
3) create 3 empty directories, named "raw", "preprocessed", and "results" within the "data" directory

## Step 3: Manual download of some input data
- Download NUTS region boundary shapefiles from here: https://ec.europa.eu/eurostat/web/gisco/geodata/statistical-units/territorial-units-statistics
  Select the years 2003, 2006, 2010, 2013, 2016, and 2021, "01M" for scale and "EPSG:3035" for coordinate reference system
- Save the downloaded Zip files in a directory named "nuts_shapedata" in the "raw" directory
- Get regional information on produced crop acreages according to the procedure described here: https://www.capri-model.org/dokuwiki_help/doku.php    (see section "CAPREG").
- Save the output of this regional data preprocessing procedure in a csv file named "res_time_series_17.csv" in the directory "raw"
-    ALTERNATIVE 1: In case you would like to use regional data directly from Eurostat, follow the procedure described here: https://github.com/JoBaumert/Probabilistic_Crop_Mapping_EU
-    ALTERNATIVE 2: contact josef.baumert@ilr.uni-bonn for a version of the csv file named "res_time_series_17.csv"
- then download this data: https://data.jrc.ec.europa.eu/dataset/f85907ae-d123-471f-a44a-8cca993485a2#dataaccess and save the csv file named "lucas_harmu_uf.csv in a directory named "LUCAS" in the "raw" directory
  
## Step 4: import data from Google Earth Engine and further preprocessing of input data
run the following files
1. generate_multi_band_raster.py
2.	csv_cleaner.py (loads the raw data of regional crop acreages from CAPRI and saves only data for relevant crops in smaller chunks)
3. find_nuts_irregularities.py (checks in which years NUTS region names appear and in which not)
4.	check_NUTS_region_availability.py (maps region codes used by CAPRI with region codes used by Eurostat etc.)
now you need to make sure that you have a Google Earth Engine account. If you run the next script, GEE will ask you to authenticate. The data will then be saved to your google drive account:
5. get_local_characteristics.py (loads relevant data from google earth engine)
save all geotiff files that you receive from GEE in a directory named "GEE" in the "raw" directory
then run: <br>
6.	LUCAS_preprocessing.py (selects only agricultural land use classes from the raw LUCAS dataset)
7.	covariates_preprocessing.py (get variables at LUCAS positions and calculate 3-year averages for varying variables)
8.	calculate_cell_weights.py (calculate UAA from CORINE, aggregate regional crop types and calculate cell weights)

## Step 5: estimation and generation of crop maps
finally, run the following files in this order: <br>
9. parameter_estimation.py (scale explanatory variables from LUCAS for every country separately and run logistic regression for each country) <br>
10. calculate_prior_probabilities.py (load estimated parameters and covariates and calculate the (prior) probability of each crop for every region in every year) <br>
11.	incorporate_regional_data.py (find posterior probabilities that are as close as possible to the prior probabilities while abiding by the regional constraints. We do this for each region and every year separately). <br>
12.	LUCAS_field_size_calculation.py (calculate estimated field sizes and from this the number of fields per cell using LUCAS data. This information is used in the random sampling step) <br>
13.	generate_EU_posterior_crop_map.py (load posterior crop probabilities for all regions and spatially merge them in one Europe-wide raster file per year, i.e., the file “EU_expected_crop_shares_year.tif”) <br>
14.	merge_results.py (load regional posterior probabilities for a country and year and sample randomly using the estimated number of fields. Outputs the ensemble of crop maps for each country)
