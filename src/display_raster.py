#%%
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import rasterio as rio
from rasterio.plot import show
from utils.preprocess_raster import read_raster

output_raster_path= Path('data/results/output_raster_1990.tif')

src_output, img, _ , _ = read_raster(output_raster_path)

show(img)
# %%
np.unique(img)

# %%
src_output.meta


# %%
import matplotlib.pyplot as plt
plt.hist(img.flatten())
# %%
