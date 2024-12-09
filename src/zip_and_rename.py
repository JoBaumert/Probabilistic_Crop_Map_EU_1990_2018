#%%
import os
import shutil
# %%
path="/home/baumert/fdiexchange/baumert/DGPCM_19902020/Data/data/results/multi_band_raster/simulted_crop_shares/"
# %%
#for dir in os.listdir(path):
#    print(dir)
#    for file in os.listdir(path+dir):
#        if file[-14:-8]=="10reps":
#            old_file=path+dir+"/"+file
#            new_file=path+dir+"/"+file[:-14]+"int.tif"
#            os.rename(old_file,new_file)
#        #    os.rename(file,)
## %%
#old_file
## %%
#"""copy any bands file and save it to all directories (all countries/years have the same bands)"""
#
## %%
#file=path+"bands.csv"
#for dir in os.listdir(path):
#    print(dir)
#    if dir!= "bands.csv":
#        shutil.copyfile(file,path+dir+"/bands.csv")
## %%
#for dir in os.listdir(path)[:1]:
#    print(path+dir)
# %%
def make_archive(source, destination):
        base = os.path.basename(destination)
        name = base.split('.')[0]
        format = base.split('.')[1]
        archive_from = os.path.dirname(source)
        archive_to = os.path.basename(source.strip(os.sep))
        shutil.make_archive(name, format, archive_from, archive_to)
        shutil.move('%s.%s'%(name,format), destination)

# %%
file=path+"bands.csv"
for dir in os.listdir(path):
    
    if (dir!= "bands.csv")&(dir[:2]!="PT"):
        print("zipping "+dir+"...")
        make_archive(path+dir, path+dir+".zip")
# %%
