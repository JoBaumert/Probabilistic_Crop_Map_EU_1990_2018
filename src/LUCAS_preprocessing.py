#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import os

# %%
main_path = str(Path(Path(os.path.abspath(__file__)).parents[0]))
data_main_path=open(main_path+"/src/data_main_path.txt").read()[:-1]

#%%

LUCAS_path=data_main_path+"/raw/LUCAS/"
output_path=data_main_path+"/preprocessed/csv/"


# %%
if __name__ == "__main__":

    if not os.path.isfile(output_path+"LUCAS_preprocessed.csv"):

        print("importing and preprocessing raw LUCAS data...")
        LUCAS_raw=pd.read_csv(LUCAS_path+"/lucas_harmo_uf.csv")

        """
        not all of the LUCAS observations are agricultural land. Select only those that are cropland 
        (letter group B) or grassland (letter group E) and land use agriculture (U111) or land use fallow land (U112)
        for a definition of the groups see "LUCAS technical reference document C3" (https://ec.europa.eu/eurostat/documents/205002/8072634/LUCAS2018-C3-Classification.pdf) 
        """

        LUCAS_agri=LUCAS_raw[((LUCAS_raw['letter_group']=='B')|(LUCAS_raw['letter_group']=='E'))& \
            ((LUCAS_raw['lu1']=='U111')|(LUCAS_raw['lu1']=='U112'))]


    Path(output_path).mkdir(parents=True, exist_ok=True)
    LUCAS_agri.to_csv(output_path+"LUCAS_preprocessed.csv", index=False, header=True)
    print("successfully completed preprocessing of LUCAS data")
# %%
