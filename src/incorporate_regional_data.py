# %%
import argparse
from cProfile import label
import jax.numpy as jnp  # it works almost exactly like numpy
from jax import grad, jit, vmap, hessian
from jax import random
from jax import jacfwd, jacrev
from jax.numpy import linalg

import jax
import jax.numpy as jnp
from jaxopt import projection
from jaxopt import ProjectedGradient
from jaxopt.projection import projection_non_negative
from jaxopt.projection import projection_box
from jaxopt import BoxOSQP
import json
import matplotlib.pyplot as plt
import numpy as np
from numpy import nanargmin, nanargmax
import os
import pandas as pd
from pathlib import Path
import pyarrow as pa
import pyarrow.parquet as pq
from rasterio.plot import show
from scipy.optimize import minimize
from scipy.optimize import Bounds
from scipy.optimize import LinearConstraint
from scipy.optimize import NonlinearConstraint
from sklearn.metrics import r2_score
import sys
import warnings
import gc
import rasterio as rio
# %%
year,country=1990,"DE"
all_years=np.arange(1990,2019)
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
prior_proba_output_dir=results_dir+"/numpy_arrays/prior_crop_probas/"
os.makedirs(prior_proba_output_dir, exist_ok=True)
resulting_parameters_dir=results_dir+"/csv/estimation_parameters_and_scalers/"
os.makedirs(resulting_parameters_dir, exist_ok=True)
parameter_path=data_main_path+"/delineation_and_parameters/"
user_parameter_path=parameter_path+"user_parameters.xlsx"
GEE_data_path=raw_dir+"/GEE/"
# %%
CAPREG_data=pd.read_csv(preprocessed_csv_dir+"preprocessed_CAPREG_step3.csv")
cellweight=rio.open(preprocessed_raster_dir+"cellweight_raster_allyears.tif").read()
nuts_indices=rio.open(preprocessed_raster_dir+"nuts_indices_relevant_allyears.tif").read()
uaa_allyears=pd.read_csv(preprocessed_csv_dir+"uaa_calculated_allyears.csv")
# %%
regs=np.unique(CAPREG_data[(CAPREG_data["country"]==country)&(CAPREG_data["year"]==year)]["CAPRI_code"])
considered_crops_country_year=np.load(prior_proba_output_dir+country+"_"+str(year)+"_considered_crops.npy",allow_pickle=True)
#%%
reg="DEA20000"
index=uaa_allyears[(uaa_allyears["year"]==year)&(uaa_allyears["CAPRI_code"]==reg)]["index"].iloc[0]
region_indices=np.where(nuts_indices[np.where(all_years==year)[0][0]]==index)
weight_array=cellweight[np.where(all_years==year)[0][0]][region_indices]

#%%


#%%
CAPREG_regional_data=CAPREG_data[(CAPREG_data["CAPRI_code"]==reg)&(CAPREG_data["year"]==year)]
true_croparea_region=np.array(CAPREG_regional_data["value"])*10
# %%
prior_probas=np.load(prior_proba_output_dir+reg+"_"+str(year)+".npy")
#%%
C=len(CAPREG_regional_data)
I=prior_probas.shape[1]

#%%
weight_array_long=np.repeat(weight_array,C)
beta=0
priorprob_relevant=prior_probas[beta].T

priorprob_relevant_corrected=np.zeros((C,I))

for c,crop in enumerate(considered_crops_country_year):
    corrected_position=np.where(CAPREG_regional_data["DGPCM_crop_code"]==crop)[0][0]
    priorprob_relevant_corrected[corrected_position]=priorprob_relevant[c]

priorprob_relevant_corrected = np.where(priorprob_relevant_corrected <= 0.000001, 0.000001, priorprob_relevant_corrected)
priorprob_relevant_corrected = np.array([p_cell / p_cell.sum() for p_cell in priorprob_relevant_corrected.T])
q=priorprob_relevant_corrected.flatten()
q_log=np.log(q)

# %%
q_log
# %%
priorprob_relevant_corrected.shape
# %%
def complete_obj_function(x):
    return jnp.log(
        obj_function(x) * obj_function_factor
        + combined_constraints(x)
    )

def obj_function(x):
    p = x

    p = jax.nn.relu(p) + 0.0000001
    # return (jnp.dot(-(p * q_log) + p * jnp.log(p), jnp.ones_like(p)))
    return -(jnp.dot(p * q_log - p * jnp.log(p), weight_array_long))

def combined_constraints(x):
    return regional_constraint(x) + cell_constraint(x) * penalty_cell

def regional_constraint(x):
    x_reshaped = x.reshape(I, C).transpose()
    weighted_crops = jnp.multiply(weight_array, x_reshaped)
    
    weighted_aggregated_crops = weighted_crops.sum(axis=1)
    
    return jnp.square(
        (weighted_aggregated_crops-true_croparea_region)*penalty_region
        ).sum()

def cell_constraint(x):
    x = x.reshape(I, C)
    return jnp.square(x.sum(axis=1)-jnp.ones(I)).sum()

def max_cell_deviation(x):
    x = x.reshape(I, C)
    return jnp.abs(x.sum(axis=1) - jnp.ones(I)).max()

def run_optimization_problem(maxiter_optimization,p_init):

    pg_boxsection = ProjectedGradient(
        fun=complete_obj_function,
        projection=projection_box,
        maxiter=maxiter_optimization,
        maxls=200,
        tol=0,  # 1e-2,
        stepsize=-0,
        decrease_factor=0.8,
        implicit_diff=True,
        verbose=False,
    )

    pg_sol_boxsection = pg_boxsection.run(p_init, hyperparams_proj=proj_params)
    return pg_sol_boxsection

def regional_crop_deviation(x):
    x_reshaped = x.reshape(I,C).transpose()
    weighted_crops = jnp.multiply(weight_array, x_reshaped)
    weighted_aggregated_crops = weighted_crops.sum(axis=1)
    return weighted_aggregated_crops-true_croparea_region

    


def evaluate_quality(x):
    max_dev_list = []
    max_dev = (abs(regional_crop_deviation(p_result))/true_croparea_region.sum()).max()
    print(f"maximal relative_deviation from regional data: {max_dev}")
    max_dev_list.append(max_dev)

    max_cell_dev = max_cell_deviation(x)
    print(f"maximal cell deviation: {max_cell_dev}")

    r2 = r2_score(q, x)
    print(f"R2 score: {r2}")
    return max_dev_list, max_cell_dev, r2
# %%
q.shape
# %%
x=q
complete_obj_function(x)

# %%
combined_constraints(x)
# %%
priorprob_relevant_corrected.shape
# %%
x_test=np.tile((true_croparea_region/true_croparea_region.sum()),I).reshape(I,C).flatten()
# %%
complete_obj_function(x_test)
# %%
p_init=q
penalty_adjusted = False

obj_function_factor,penalty_cell,penalty_region=1,50,0.5


#while not penalty_adjusted:
box_lower = jnp.zeros_like(p_init)
box_upper = jnp.ones_like(p_init)
proj_params = (box_lower, box_upper)

max_iter = 100000
result = run_optimization_problem(
    maxiter_optimization=max_iter,p_init=p_init
)

p_result = result.params

# %%
evaluate_quality(p_result)
# %%

# %%
x=p_result
x_reshaped = x.reshape(I,C).transpose()
weighted_crops = jnp.multiply(weight_array, x_reshaped)
# %%
plt.scatter(weighted_crops.sum(axis=1),true_croparea_region)
# %%
x=p_result
x = x.reshape(I, C)
t=x.sum(axis=1) - jnp.ones(I)
# %%
plt.hist(t)
# %%
x.sum(axis=1)
# %%
plt.scatter(p_result,q)
# %%
