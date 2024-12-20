Recommended order of running files:
1.	generate_multi_band_raster.py
2.	csv_cleaner.py (loads the raw data from CAPRI and saves only data for relevant crops in smaller chunks)
3.	check_NUTS_region_availability.py (maps region codes used by CAPRI with region codes used by Eurostat etc.)
4.	get_local_characteristics.py (loads relevant data from google earth engine)
5.	LUCAS_preprocessing.py (selects only agricultural land use classes from the raw LUCAS dataset)
6.	covariates_preprocessing.py (get variables at LUCAS positions and calculate 3-year averages for varying variables)
7.	calculate_cell_weights.py (calculate UAA from CORINE, aggregate regional crop types and calculate cell weights)
8.	parameter_estimation.py (scale explanatory variables from LUCAS for every country separately and run logistic regression for each country)
9.	calculate_prior_probabilities.py (load estimated parameters and covariates and calculate the (prior) probability of each crop for every region in every year)
10.	incorporate_regional_data.py (find posterior probabilities that are as close as possible to the prior probabilities while abiding by the regional constraints. We do this for each region and every year separately). 
11.	LUCAS_field_size_calculation.py (calculate estimated field sizes and from this the number of fields per cell using LUCAS data. This information is used in the random sampling step)
12.	generate_EU_posterior_crop_map.py (load posterior crop probabilities for all regions and spatially merge them in one Europe-wide raster file per year, i.e., the file “EU_expected_crop_shares_year.tif”)
13.	merge_results.py (load regional posterior probabilities for a country and year and sample randomly using the estimated number of fields. Outputs the ensemble of crop maps for each country)
