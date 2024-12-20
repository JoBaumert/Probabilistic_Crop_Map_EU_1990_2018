#%%
import numpy as np
import pandas as pd
# %%
def cluster_crop_names(crop_data_raw,crop_delineation,all_crops):
    crop_data=crop_data_raw.copy()
    crop_data.drop("type",inplace=True,axis=1)
    crop_data.drop("Unnamed: 0",inplace=True,axis=1)
    
    calc_ocer_df1=crop_data[crop_data["crop"].isin(["SWHE","DWHE","RYEM","BARL","OATS","MAIZ"])].groupby([
        "CAPRI_code","NUTS_ID","year","country","NUTS_level"
    ]).sum().reset_index()

    calc_ocer_df2=crop_data[crop_data["crop"]=="CERE"].groupby([
        "CAPRI_code","NUTS_ID","year","country","NUTS_level"
    ]).sum().reset_index()

    calc_ocer_df=pd.merge(calc_ocer_df2,calc_ocer_df1,how="left",on=["CAPRI_code","NUTS_ID","year","country","NUTS_level"])
    calc_ocer_df.fillna(0,inplace=True)
    calc_ocer_df["value"]=calc_ocer_df["value_x"]-calc_ocer_df["value_y"]
    calc_ocer_df["DGPCM_crop_code"]=np.repeat("OCER",len(calc_ocer_df))
    calc_ocer_df=calc_ocer_df[["CAPRI_code","NUTS_ID","year","country","NUTS_level","DGPCM_crop_code","value"]]

    calc_text_df1=crop_data[crop_data["crop"].isin(["RAPE","SUNF","SOYA"])].groupby([
        "CAPRI_code","NUTS_ID","year","country","NUTS_level"
    ]).sum().reset_index()

    calc_text_df2=crop_data[crop_data["crop"].isin(["TEXT","OILS"])].groupby([
        "CAPRI_code","NUTS_ID","year","country","NUTS_level"
    ]).sum().reset_index()

    calc_text_df=pd.merge(calc_text_df2,calc_text_df1,how="left",on=["CAPRI_code","NUTS_ID","year","country","NUTS_level"])

    calc_text_df.fillna(0,inplace=True)
    calc_text_df["value"]=calc_text_df["value_x"]-calc_text_df["value_y"]
    calc_text_df["DGPCM_crop_code"]=np.repeat("TEXT",len(calc_text_df))
    calc_text_df=calc_text_df[["CAPRI_code","NUTS_ID","year","country","NUTS_level","DGPCM_crop_code","value"]]

    calc_maiz_df=crop_data[crop_data["crop"].isin(["MAIZ","MAIF"])].groupby([
        "CAPRI_code","NUTS_ID","year","country","NUTS_level"
    ]).sum().reset_index()
    calc_maiz_df["DGPCM_crop_code"]=np.repeat("MAIZ",len(calc_maiz_df))

    calc_oliv_df=crop_data[crop_data["crop"].isin(["OLIV","TABO"])].groupby([
        "CAPRI_code","NUTS_ID","year","country","NUTS_level"
    ]).sum().reset_index()
    calc_oliv_df["DGPCM_crop_code"]=np.repeat("OLIV",len(calc_oliv_df))

    calc_viny_df=crop_data[crop_data["crop"].isin(["VINY","TWIN"])].groupby([
        "CAPRI_code","NUTS_ID","year","country","NUTS_level"
    ]).sum().reset_index()
    calc_viny_df["DGPCM_crop_code"]=np.repeat("VINY",len(calc_viny_df))

    remaining_crops=crop_delineation["CAPRI_code"].unique()[np.where(crop_delineation["CAPRI_code"].unique().astype(str)!="nan")]
    
    remaining_crop_df=crop_data[crop_data["crop"].isin(remaining_crops)]
    remaining_crop_df.rename(columns={"crop":"DGPCM_crop_code"},inplace=True)
    
    new_crop_df=pd.concat((remaining_crop_df,pd.concat((
        calc_ocer_df,pd.concat((
            calc_text_df,pd.concat((
                calc_maiz_df,pd.concat((
                    calc_oliv_df,calc_viny_df
                    ))
                ))
            ))
        ))
    ))
    
    new_crop_df.sort_values(by=["year","CAPRI_code","NUTS_ID","country","DGPCM_crop_code"],inplace=True)
    
    new_crop_df=new_crop_df[new_crop_df["value"]>=0]
    
    a=new_crop_df.groupby(["CAPRI_code","NUTS_ID","year","country","NUTS_level"]).sum().reset_index()
    a=a[["CAPRI_code","NUTS_ID","year","country","NUTS_level"]]

    b=pd.DataFrame(np.repeat(np.array(a),len(all_crops),axis=0),columns=a.columns)
    b["DGPCM_crop_code"]=np.tile(all_crops,len(a))

    c=pd.merge(b,new_crop_df,how="left",
            on=['CAPRI_code','NUTS_ID', 'year', 'country', 'NUTS_level','DGPCM_crop_code'])

    c.fillna(0,inplace=True)
    c.sort_values(by=["year","CAPRI_code","NUTS_ID","country","DGPCM_crop_code"],inplace=True)

    return c
#%%
