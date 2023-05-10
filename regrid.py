import sys
import os
import xarray as xr
from glob import glob
import numpy as np

#get the model name that we are combining on
model_arg=sys.argv[1]

#good to make this is a little larger than it has to be just in case
#is the variable called lev? Lets hope so!
#note - if this isn't work, take out the {zsel} from the remap command 
zcut=400

#this is the outfile template
outfile_template=f'VAR.MODEL.YEAR.CadBasin.temp.nc'
finalname_template=f'VAR.MODEL.CadBasin.nc'

cdo='/opt/homebrew/bin/cdo'
haslevs=False

def make_cadbasin_average(model_name):
    #loop over the files
    file_list=glob(f'*_{model_name}_*.nc')
    print(file_list)
    fl0=file_list[0]
    
    #generate the regridding weights 
    wn=f'weights.{model_name}'
    print(wn)
    os.system(f'{cdo} genbil,1x1.grid.cadbasin {file_list[0]} {wn}')

    if haslevs:
        #make a list of levels to go over 
        d=xr.open_dataset(fl0)
        lev=d.lev.data
        d.close()
        zindex=np.where(lev<zcut)[0]
        zindex=zindex
        zsellist=str(zindex[0]+1)
        if len(zindex>1):
            for z in zindex[1:]:
                zsellist=zsellist+','+str(z+1)
                zsel=f'-sellevidx,{zsellist}'
                print(f'zsel is {zsel}')

    #loop over the files
    for f in file_list:
        fl=f.split('_')
        print(fl)
        #figure out what the outfile should be 
        outfile_name=outfile_template.replace('MODEL',fl[2]).replace('YEAR',fl[6]).replace('VAR',fl[0])
        #regrid the file and store temporarily
        if haslevs:
            os.system(f'{cdo} -remap,1x1.grid.cadbasin,{wn} {zsel} {f} {outfile_name}')
        else:
            os.system(f'{cdo} -remap,1x1.grid.cadbasin,{wn} {f} {outfile_name}')

    #use cdo to merge across times, why not
    finalname=finalname_template.replace('MODEL',fl[2]).replace('VAR',fl[0])
    os.system(f'{cdo} mergetime *{model_name}*.temp.nc {finalname}')
    #delete the temporary files 
    os.system(f'rm *{model_name}*.temp.nc')
    #delete the weight file
    os.system(f'rm {wn}')

if not model_arg=='all':

    make_cadbasin_average(model_arg)    

else:

    file_list=glob('*_gn_*.nc')
    for i,f in enumerate(file_list): file_list[i]=f.split('_')[2]
    model_list=list(set(file_list))
    print(model_list)

    for m in model_list:
        #note, if cdo returns a segfault (which it will if there is a downloading error)
        #then this function still "works", but there won't be a result 
        #can't really do anything with this for now, but good to note that this script
        #might run but still not return any errors to python. 
        make_cadbasin_average(m)
