#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#Figure 3: Map of the Arctic ocean
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

def plot_arctic_map(save_as_file=False, filename='arctic_map.png'):
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.NorthPolarStereo())
    ax.set_extent([-170, 160, 60, 90], ccrs.PlateCarree())

    # Read land geometries from the Natural Earth dataset
    land_shp = shapereader.natural_earth(resolution='50m', category='physical', name='land')
    land_geometries = shapereader.Reader(land_shp).geometries()

    ax.add_geometries(land_geometries, ccrs.PlateCarree(), facecolor='grey', edgecolor='black')

    ax.add_feature(cfeature.OCEAN, facecolor='white', zorder=0)
    ax.add_feature(cfeature.COASTLINE)
    ax.add_feature(cfeature.BORDERS, linestyle=':')

    if save_as_file:
        plt.savefig(filename, dpi=300)
    else:
        plt.show()

plot_arctic_map(save_as_file=True, filename='arctic_map.png')


# In[ ]:


#Import observational data and create arrays and get the annual salinity gradient
import scipy.io as spio
import numpy as np
#creating data arrays for the observational data
dm= spio.loadmat('/Users/nuhahameed/Documents/Data for MSci/ITPAJX_Robert .mat')
mo=dm['AJX'][0,0]['mo'][0]
nl=np.zeros(12)
TA=np.zeros([12,500])
SA=np.zeros([12,500])
for i,m in enumerate(mo):
    TA[m-1][np.where(~np.isnan(dm['AJX'][0,0]['T'][i,:]))]+=dm['AJX'][0,0]['T'][i,:][np.where(~np.isnan(dm['AJX'][0,0]['T'][i,:]))]
    SA[m-1][np.where(~np.isnan(dm['AJX'][0,0]['S'][i,:]))]+=dm['AJX'][0,0]['S'][i,:][np.where(~np.isnan(dm['AJX'][0,0]['S'][i,:]))]
    nl[m-1]+=1
TA/=nl[:,np.newaxis]
SA/=nl[:,np.newaxis]

mo=dm['ITP'][0,0]['mo'][0]
nl=np.zeros(12)
TI=np.zeros([12,500])
SI=np.zeros([12,500])
for i,m in enumerate(mo):
    TI[m-1][np.where(~np.isnan(dm['ITP'][0,0]['T'][i,:]))]+=dm['ITP'][0,0]['T'][i,:][np.where(~np.isnan(dm['ITP'][0,0]['T'][i,:]))]
    SI[m-1][np.where(~np.isnan(dm['ITP'][0,0]['S'][i,:]))]+=dm['ITP'][0,0]['S'][i,:][np.where(~np.isnan(dm['ITP'][0,0]['S'][i,:]))]
#    TI[m-1]+=dm['ITP'][0,0]['T'][i,:]
#    SI[m-1]+=dm['ITP'][0,0]['S'][i,:]
    nl[m-1]+=1
TI/=nl[:,np.newaxis]
SI/=nl[:,np.newaxis]

depth_res = 1  # depth resolution in meters
depth_values = np.arange(500) * depth_res  # create array of depth values

AJX_avg_salinity = np.mean(SA, axis=0)  # compute average salinity across all months
ITP_avg_salinity = np.nanmean(SI, axis=0) #compute average salinity across all months


# In[ ]:


#Compute surface salinity difference between observationsal and multi-model data
surface_salinity_1970 = [data[0] for data in interpolated_salinity_data['1970']]  # Salinity data at depth=0m for 1975
surface_salinity_2010 = [data[0] for data in interpolated_salinity_data['2010']]  # Salinity data at depth=0m for 2010
surface_salinity_diff = np.array(surface_salinity_1970) - np.array(surface_salinity_2010)

surface_salinity_diff[0] #Simulated multi model difference

obs_so_diff = ITP_avg_salinity[0:1] - AJX_avg_salinity[0:1] 
obs_so_diff #Observational surface salinity difference 


# In[ ]:


# Figure 6: Multi-model vertical salinity profile
import os
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from scipy import interpolate

input_folder ='/Volumes/Seagate Hub/cmip6_downloader/data/Salinity models/'
myfiles = os.listdir(input_folder)

for f in myfiles:
    if '._so' in f:
        myfiles.remove(f)
        
input_folder = '/Volumes/Seagate Hub/cmip6_downloader/data/Salinity models/'

# Create list of all files in folder
all_files = os.listdir(input_folder)

# Remove any hidden files from the list
all_files = [f for f in all_files if not f.startswith('.')]

def calc_mean_so(ds, year):
    #Year = ['1970', '1980', '1990', '2000', '2010']
    salinity = []
    #for j, year in enumerate(Year):
    ds_mean = ds.so.isel(time=np.arange((int(year)-1850)*12, (int(year)+1-1850)*12,1)).mean(dim='time')
    y = ds_mean.mean(dim=['longitude', 'latitude']) #works out the average within the longitudanal range
    salinity.append(y)
    return salinity

new_list = []


def interp(array):
    x_out = np.arange(0, 150, 1)
    y_in = array.values
    x_in = array.lev.values
    f = interpolate.interp1d(x_in, y_in, fill_value='extrapolate') #option: change to fill_value='extrapolate'
    y_out = f(x_out)
    return y_out

def plot_salinity_depth(years, salinity_depth_data):
    for i, data in enumerate(salinity_depth_data):
        Year = ['1970', '1980', '1990', '2000', '2010']
        plt.plot(data, np.arange(0, 150, 1), label=years[i], color=plt.cm.viridis(i/len(Year)), linewidth = 4)
    plt.ylim(150, 0)
    plt.xlim(27,35)
    plt.xlabel('Salinity (g/kg)', fontsize = 15)
    plt.ylabel('Depth (m)', fontsize = 15)
    plt.title(f"Multi Model Mean", fontsize = 20)
    
    plt.plot(AJX_avg_salinity, depth_values, label='1975 AIDJEX', color='red', linewidth = 4) 
    plt.plot(ITP_avg_salinity, depth_values, label='2006-2012 ITP', color='orange', linewidth = 4)
    plt.legend()
    plt.savefig('Multi Model Mean.png')
    plt.show()

# Loop over all files and calculate decadal average salinity data
all_salinity_depth_data = {}
for year in ['1970', '1980', '1990', '2000', '2010']:
    salinity_depth_data = []
    for file in all_files:
        salinity_depth_data.append(calc_mean_so(xr.open_dataset(input_folder+file), year))
    all_salinity_depth_data[year] = salinity_depth_data

# Interpolate the data
interpolated_salinity_data = {}
for year, salinity_depth_data in all_salinity_depth_data.items():
    interpolated_salinity_data[year] = [interp(array[0]) for array in salinity_depth_data]

# Calculate the mean salinity data for each decade
mean_salinity_data = {}
for year, interpolated_salinity in interpolated_salinity_data.items():
    mean_salinity_data[year] = np.mean(interpolated_salinity, axis=0)

# Plot the mean salinity data for all decades
plot_salinity_depth(['1970', '1980', '1990', '2000', '2010'], list(mean_salinity_data.values()))


# In[ ]:


#Figure 7: Simulated decadal mean for each induvidual model
import os
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import string

input_folder ='/Volumes/Seagate Hub/cmip6_downloader/data/Salinity models/'
myfiles = os.listdir(input_folder)

for f in myfiles:
    if '._so' in f:
        myfiles.remove(f)

fig, axs = plt.subplots(8, 4, layout='tight', figsize=(5 * 5, 5 * 8), sharex=True) #creates the subplots
axs.reshape(-4)
axs[-1, -1].remove()
axs[-1, -2].remove()  

def int_to_roman(integer):
    val = [
        1000, 900, 500, 400,
        100, 90, 50, 40,
        10, 9, 5, 4,
        1
        ]
    syb = [
        "M", "CM", "D", "CD",
        "C", "XC", "L", "XL",
        "X", "IX", "V", "IV",
        "I"
        ]
    roman_num = ''
    i = 0
    while integer > 0:
        for _ in range(integer // val[i]):
            roman_num += syb[i]
            integer -= val[i]
        i += 1
    return roman_num

def figureseven(file, i):
    ds = xr.open_dataset(input_folder+file) #opens the dataset
    if 'CESM' in file: 
        ds['lev'] = ds['lev']/100
        
    model_name = file.split('so.')[-1].split('.')[0]

    Year = ['1970', '1980', '1990', '2000', '2010']
    
    for j, year in enumerate(Year):
        ds_mean = ds.so.isel(time=np.arange((int(year)-1850)*12, (int(year)+1-1850)*12,1)).mean(dim='time')
        y = ds_mean.mean(dim=['longitude', 'latitude']) #works out the average within the longitudanal range
        
        ds_std = ds.so.isel(time=np.arange((int(year)-1850)*12, (int(year)+1-1850)*12,1)).std(dim='time')
        ds_std_latlon = ds_std.std(dim = ['longitude', 'latitude'])

        y_upper = y + ds_std_latlon
        y_lower = y - ds_std_latlon
        
        axs.reshape(-1)[i].plot(y, ds['lev'], label=year, color=plt.cm.viridis(j/len(Year)), linewidth = 4)
        
        for year in range(1970, 2011):
            ds_year = ds.so.isel(time=np.arange((year-1850)*12, (year+1-1850)*12, 1)).mean(dim='time')
            y_year = ds_year.mean(dim=['longitude', 'latitude'])
            axs.reshape(-1)[i].plot(y_year, ds['lev'], color='gray', alpha=0.3, linewidth=0.3)

        axs.reshape(-1)[i].fill_betweenx(ds['lev'], y_upper, y_lower, color=plt.cm.viridis(j/len(Year)), alpha=0.2)
        
        axs.reshape(-1)[i].set_xlabel('Salinity (g/kg)', fontsize = 20)
        axs.reshape(-1)[i].set_ylabel('Depth (m)', fontsize = 20)
        axs.reshape(-1)[i].set_ylim(0,150)
        axs.reshape(-1)[i].set_xlim(27,35)
        axs.reshape(-1)[i].tick_params(axis='x', labelsize=18)
        axs.reshape(-1)[i].tick_params(axis='y', labelsize=18)
        axs.reshape(-1)[i].xaxis.set_tick_params(which='both', labelbottom=True)

    axs.reshape(-1)[i].invert_yaxis()
    axs.reshape(-1)[i].set_title(model_name, fontsize = 25)
    axs.reshape(-1)[i].plot(AJX_avg_salinity, depth_values, label='1975 AIDJEX', color='red', linewidth = 4)
    axs.reshape(-1)[i].plot(ITP_avg_salinity, depth_values, label='2006-2012 ITP', color='orange', linewidth =4)
    
    
    if i < 26:
        subplot_label = string.ascii_lowercase[i]
    else:
        subplot_label = int_to_roman((i - 25))

    axs.reshape(-1)[i].text(0.02, 1.1, subplot_label, transform=axs.reshape(-1)[i].transAxes, fontsize=16, fontweight='bold', va='top', ha='left')


i=0
for file in myfiles:    
    figureseven(file, i)
    i=i+1 
    
Year = ['1970', '1980', '1990', '2000', '2010']
year_colors = [plt.cm.viridis(j/len(Year)) for j in range(len(Year))]
year_handles = [plt.Line2D([0], [0], color=color, lw=4, label=year) for year, color in zip(Year, year_colors)]
aidjex_handle = plt.Line2D([0], [0], color='red', lw=4, label='1975 AIDJEX')
itp_handle = plt.Line2D([0], [0], color='orange', lw=4, label='2006-2012 ITP')
legend_handles = year_handles + [aidjex_handle, itp_handle]

x, y = 0.713, 0.12
fig.legend(legend_handles, [handle.get_label() for handle in legend_handles], fontsize=30, bbox_to_anchor=(x, y), ncol=1, borderaxespad=0)
plt.savefig('Salinity_profiles_09_04.png')


# In[ ]:


#Figure 8: Salinity profiles of models in two subplots
import os
import xarray as xr
import matplotlib.pyplot as plt
import seaborn as sns

input_folder = '/Volumes/Seagate Hub/cmip6_downloader/data/Salinity models/'
myfiles = os.listdir(input_folder)

for f in myfiles:
    if '._so' in f:
        myfiles.remove(f)

colors = sns.color_palette("husl", len(myfiles))
linestyles = ['-', '--', '-.', ':'] * (len(myfiles) // 4 + 1)

fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(29, 12), sharex=True)
plt.subplots_adjust(right=0.7, hspace=0.3)

def plot_salinity_profile(ax, file, color, linestyle, start_year, end_year):
    ds = xr.open_dataset(input_folder + file)
    if 'CESM' in file:
        ds['lev'] = ds['lev'] / 100

    model_name = file.split('so.')[-1].split('.')[0]
    ds_mean = ds.so.isel(time=np.arange((start_year-1850)*12, (end_year+1-1850)*12, 1)).mean(dim='time')
    y = ds_mean.mean(dim=['longitude', 'latitude'])
    
    ax.plot(y, ds['lev'], label=model_name, color=color, linewidth=2, linestyle=linestyle)

for file, color, linestyle in zip(myfiles, colors, linestyles):
    plot_salinity_profile(axs[0], file, color, linestyle, 1970, 1980)
    plot_salinity_profile(axs[1], file, color, linestyle, 2000, 2010)

axs[0].plot(AJX_avg_salinity, depth_values, label='1975 AIDJEX', color='red', linewidth = 4)
axs[1].plot(ITP_avg_salinity, depth_values, label='2006-2012 ITP', color='orange', linewidth=4)

for i, ax in enumerate(axs):
    ax.set_ylabel('Depth (m)', fontsize=14)
    ax.set_xlabel('Salinity (g/kg)', fontsize=14)
    ax.set_ylim(0, 150)
    ax.set_xlim(27, 35)
    ax.tick_params(axis='y', labelsize=12)
    ax.tick_params(axis='x', labelsize=12)
    ax.invert_yaxis()
    
   
    label = chr(97 + i)  # Convert index to ASCII character ('a', 'b', etc.)
    ax.text(-0.1, 1, label, fontsize=19, fontweight='bold', transform=ax.transAxes)
    
axs[0].set_title('1970-1980 Salinity Profiles', fontsize=25)
axs[1].set_title('2000-2010 Salinity Profiles', fontsize=25)

# Get the handles and labels from the first subplot
handles, labels = axs[0].get_legend_handles_labels()

# Add the handle and label for the ITP line from the second subplot
handles_itp, labels_itp = axs[1].get_legend_handles_labels()
handles += handles_itp[-1:]
labels += labels_itp[-1:]

# Create a shared legend for all subplots
fig.legend(handles, labels, fontsize=15, bbox_to_anchor=(0.68, -0.1, 0.15, 1), ncol=1, borderaxespad=0)
    

plt.savefig('multi_model_Salinity_Profiles_13_04_.png', bbox_inches='tight')


# In[ ]:


# Work out surface salinity difference between two time periods for each model (Table S1)
import os
import xarray as xr
import pandas as pd

input_folder = '/Volumes/Seagate Hub/cmip6_downloader/data/Salinity models/'
myfiles = os.listdir(input_folder)

# Remove any filenames that contain '._so'
myfiles = [f for f in myfiles if '._so' not in f]

# Create a list to store the differences in average salinity at depth 0
salinity_diffs = []

for file in myfiles:
    ds = xr.open_dataset(input_folder + file)

    model_name = file.split('so.')[-1].split('.')[0] #extract model name

    # Calculate the mean salinity at depth 0 for each decade
    salinity_1970_1980 = ds['so'].sel(time=slice('1970', '1979'), lev=ds['lev'].sel(method='nearest', lev=0)).mean().values
    salinity_2000_2010 = ds['so'].sel(time=slice('2000', '2009'), lev=ds['lev'].sel(method='nearest', lev=0)).mean().values

    # Calculate the difference in decadal average salinity at depth 0 between 1970-1980 and 2000-2010
    salinity_diff = salinity_2000_2010 - salinity_1970_1980

    # Add the model name and salinity difference to the list
    salinity_diffs.append([model_name, salinity_diff])

# Create a pandas DataFrame from the list
df_salinitydifference = pd.DataFrame(salinity_diffs, columns=['Model name', 'Salinity difference'])
df_salinitydifference = df.sort_values('Salinity difference', ascending=False)


# In[ ]:


#Figure 9: Histogram of surface salinity difference
import matplotlib.pyplot as plt
# Plot a histogram of the salinity difference values
plt.hist(df['Salinity difference'], bins=10)
plt.title('Surface Salinity Difference', fontsize=15)
plt.xlabel('Difference in surface salinity between 2010 and 1970 (g/kg)', fontsize=12)
plt.ylabel('Frequency', fontsize=12)


# In[ ]:


#Maximum gradient and MLD for observed data (Table 2)
import numpy as np
import pandas as pd
# Calculate derivatives for both datasets
dSI_dz_AJX = np.gradient(AJX_avg_salinity, depth_values)
dSI_dz_ITP = np.gradient(ITP_avg_salinity, depth_values)
# Find the maximum gradient and its corresponding depth for both datasets
max_gradient_AJX = np.max(dSI_dz_AJX)
max_gradient_depth_AJX = depth_values[np.argmax(dSI_dz_AJX)]

max_gradient_ITP = np.max(dSI_dz_ITP)
max_gradient_depth_ITP = depth_values[np.argmax(dSI_dz_ITP)]

# Create a dictionary to store the dataset names, maximum gradients, and corresponding depths
data = {'Dataset': ['AIDJEX', 'ITP'],
        'Max Gradient': [max_gradient_AJX, max_gradient_ITP],
        'Depth of Max Gradient': [max_gradient_depth_AJX, max_gradient_depth_ITP]}

# Convert the dictionary to a pandas DataFrame
df_observations = pd.DataFrame(data)
df_observations.to_excel('Observations.xlsx', index=False)


# Display the DataFrame
print(df_observations)


# In[ ]:


#Figure 10: Salinity derivatives profiles of all of the models in two subplots
import os
import numpy as np
import xarray as xr
import seaborn as sns
import matplotlib.pyplot as plt

input_folder = '/Volumes/Seagate Hub/cmip6_downloader/data/Salinity models/'
myfiles = os.listdir(input_folder)

for f in myfiles:
    if '._so' in f:
        myfiles.remove(f)

colors = sns.color_palette("husl", len(myfiles))
linestyles = ['-', '--', '-.', ':'] * (len(myfiles) // 4 + 1)

fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(29, 12), sharex=True)
plt.subplots_adjust(right=0.7, hspace=0.3)

def plot_salinity_profile(ax, file, color, linestyle, start_year, end_year):
    ds = xr.open_dataset(input_folder + file)
    if 'CESM' in file:
        ds['lev'] = ds['lev'] / 100

    model_name = file.split('so.')[-1].split('.')[0]
    #start_year = 2000
    #end_year = 2010
    ds_mean = ds.so.isel(time=np.arange((start_year-1850)*12, (end_year+1-1850)*12, 1)).mean(dim='time')
    y = ds_mean.mean(dim=['longitude', 'latitude'])
    d_values = ds.lev.values
    zh = .5*(ds.lev[1:]+ds.lev[:-1]) #get the depth of the midpoints so that this is a centered derivative
    
    # calculate derivative
    ds_derivative = np.gradient(y, d_values, axis=0)

    ax.plot(ds_derivative[:-2], zh, label=model_name, color=color, linewidth=2, linestyle=linestyle)

for file, color, linestyle in zip(myfiles, colors, linestyles):
    plot_salinity_profile(axs[0], file, color, linestyle, 1970, 1980)
    plot_salinity_profile(axs[1], file, color, linestyle, 2000, 2010)

axs[0].plot(dSI_dz_AJX, depth_values, label='1975 AIDJEX', color='red', linewidth = 4)
axs[1].plot(dSI_dz_ITP, depth_values, label='2006-2012 ITP', color='orange', linewidth = 4)

for i, ax in enumerate(axs):
    ax.set_ylabel('Depth (m)', fontsize=14)
    ax.set_xlabel('Salinity Gradient (g/kg/m)', fontsize=14)
    ax.set_ylim(0, 150)
    ax.set_xlim(0,0.3) 
    ax.tick_params(axis='y', labelsize=12)
    ax.tick_params(axis='x', labelsize=12)
    ax.invert_yaxis()
    
    # Add subplot labels
    label = chr(97 + i)  # Convert index to ASCII character ('a', 'b', etc.)
    ax.text(-0.1, 1, label, fontsize=19, fontweight='bold', transform=ax.transAxes)
    
axs[0].set_title('1970-1980 Salinity Derivatives', fontsize=20)
axs[1].set_title('2000-2010 Salinity Derivatives', fontsize=20)

# Get the handles and labels from the first subplot
handles, labels = axs[0].get_legend_handles_labels()

# Add the handle and label for the ITP line from the second subplot
handles_itp, labels_itp = axs[1].get_legend_handles_labels()
handles += handles_itp[-1:]
labels += labels_itp[-1:]

# Create a shared legend for all subplots
fig.legend(handles, labels, fontsize=15, bbox_to_anchor=(0.68, -0.1, 0.15, 1), ncol=1, borderaxespad=0)


plt.savefig('multi_model_Salinity_Derivatives_14_04_.png', bbox_inches='tight')


# In[ ]:


#Figure 11: vertical salinity gradient for each model
import os
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import string

input_folder ='/Volumes/Seagate Hub/cmip6_downloader/data/Salinity models/'
myfiles = [f for f in os.listdir(input_folder) if not f.startswith('._')]

fig, axs = plt.subplots(8, 4, layout='tight', figsize=(5 * 5, 5 * 8), sharex=True)
axs.reshape(-4)
axs[-1, -1].remove()
axs[-1, -2].remove()

def int_to_roman(integer):
    val = [
        1000, 900, 500, 400,
        100, 90, 50, 40,
        10, 9, 5, 4,
        1
        ]
    syb = [
        "M", "CM", "D", "CD",
        "C", "XC", "L", "XL",
        "X", "IX", "V", "IV",
        "I"
        ]
    roman_num = ''
    i = 0
    while integer > 0:
        for _ in range(integer // val[i]):
            roman_num += syb[i]
            integer -= val[i]
        i += 1
    return roman_num


def derivativeplot(file, i):
    ds = xr.open_dataset(os.path.join(input_folder, file))
    if 'CESM' in file: 
        ds['lev'] = ds['lev']/100
    model_name = file.split('so.')[-1].split('.')[0]

    Year = ['1970', '1980', '1990', '2000', '2010']

    for j, year in enumerate(Year):
        ds_mean = ds.so.isel(time=np.arange((int(year)-1850)*12, (int(year)+1-1850)*12,1)).mean(dim='time')
        y = ds_mean.mean(dim=['longitude', 'latitude']) #works out the average within the longitudanal range
        d_values = ds.lev.values
        zh = .5*(ds.lev[1:]+ds.lev[:-1]) #get the depth of the midpoints so that this is a centered derivative

        # calculate derivative
        ds_derivative = np.gradient(y, d_values, axis=0)

        axs.reshape(-1)[i].plot(ds_derivative[:-2], zh, label=year, linewidth = 4, color=plt.cm.viridis(j/len(Year))) 
        #ax.set_yscale('log')
        
        #ax.set_xlabel('Salinity Gradient (g/kg/m)')
        axs.reshape(-1)[i].set_xlabel('Salinity Gradient (g/kg/m)', fontsize = 20)
        axs.reshape(-1)[i].set_ylabel('Depth (m)', fontsize = 20)
        axs.reshape(-1)[i].set_ylim(0,150)
        axs.reshape(-1)[i].set_xlim(0,0.3)             
        axs.reshape(-1)[i].tick_params(axis='x', labelsize=18)
        axs.reshape(-1)[i].tick_params(axis='y', labelsize=18)
        axs.reshape(-1)[i].xaxis.set_tick_params(which='both', labelbottom=True)
        
        
    axs.reshape(-1)[i].invert_yaxis()
    axs.reshape(-1)[i].set_title(model_name, fontsize = 25)
    axs.reshape(-1)[i].plot(dSI_dz_AJX, depth_values, label='1975 AIDJEX', color='red', linewidth = 4)
    axs.reshape(-1)[i].plot(dSI_dz_ITP, depth_values, label='2006-2012 ITP', color='orange', linewidth = 4)
   
    
    if i < 26:
        subplot_label = string.ascii_lowercase[i]
    else:
        subplot_label = int_to_roman((i - 25))

    axs.reshape(-1)[i].text(0.02, 1.1, subplot_label, transform=axs.reshape(-1)[i].transAxes, fontsize=16, fontweight='bold', va='top', ha='left')

i=0
for file in myfiles:    
    derivativeplot(file, i)
    i=i+1
    
Year = ['1970', '1980', '1990', '2000', '2010']
year_colors = [plt.cm.viridis(j/len(Year)) for j in range(len(Year))]
year_handles = [plt.Line2D([0], [0], color=color, lw=5, label=year) for year, color in zip(Year, year_colors)]
aidjex_handle = plt.Line2D([0], [0], color='red', lw=5, label='1975 AIDJEX')
itp_handle = plt.Line2D([0], [0], color='orange', lw=5, label='2006-2012 ITP')
legend_handles = year_handles + [aidjex_handle, itp_handle]

# Customize the position of the legend by changing x and y
x, y = 0.713, 0.12
fig.legend(legend_handles, [handle.get_label() for handle in legend_handles], fontsize=30, bbox_to_anchor=(x, y), ncol=1, borderaxespad=0)
    
plt.savefig('derivative_plot_09_04.png')


# In[ ]:


# The maximum gradient difference between two time periods for all of the models (Table S2) and the observed maximum gradient difference
import os
import xarray as xr
import numpy as np
import pandas as pd

# Calculate the maximum gradient and corresponding depth for both datasets
max_gradient_AJX = np.max(dSI_dz_AJX)
max_gradient_depth_AJX = depth_values[np.argmax(dSI_dz_AJX)]

max_gradient_ITP = np.max(dSI_dz_ITP)
max_gradient_depth_ITP = depth_values[np.argmax(dSI_dz_ITP)]

# Calculate the difference in maximum gradient between the ITP and AIDJEX datasets
max_gradient_difference = max_gradient_ITP - max_gradient_AJX

# Create a dictionary to store the dataset names and maximum gradient difference
data = {'Dataset': ['ITP - AIDJEX'],
        'Max Gradient Difference': [max_gradient_difference]}

# Convert the dictionary to a pandas DataFrame
df_gradient_difference = pd.DataFrame(data)

# Display the DataFrame
print(df_gradient_difference)


input_folder = '/Volumes/Seagate Hub/cmip6_downloader/data/Salinity models/'
myfiles = os.listdir(input_folder)

for f in myfiles:
    if '._so' in f:
        myfiles.remove(f)

# Modify the max_gradient_depth function to accept a range of years
def max_gradient_depth(file, year, max_depth=150):
    ds = xr.open_dataset(os.path.join(input_folder, file), engine='netcdf4')
    if 'CESM' in file: 
        ds['lev'] = ds['lev']/100
    model_name = file.split('so.')[-1].split('.')[0]

    # Extract salinity data for the specified range of years
    ds_mean = ds.so.sel(lev=slice(0, max_depth)).isel(time=np.arange((int(year)-1850)*12, (int(year)+1-1850)*12,1)).mean(dim='time')
    y = ds_mean.mean(dim=['longitude', 'latitude'])

    # Calculate salinity gradient (derivative)
    d_values = ds.lev.sel(lev=slice(0, max_depth)).values
    ds_derivative = np.gradient(y, d_values, axis=0)

    # Find maximum salinity gradient
    max_gradient = ds_derivative[:-2].max()

    return {'Model': model_name, 'Max Gradient': max_gradient}

# Create an empty list to store results
results = []

# Loop through each file and calculate maximum salinity gradient difference
for file in myfiles:
    result_2000_2010 = max_gradient_depth(file, year='2010')
    result_1970_1980 = max_gradient_depth(file, year='1970')

    max_gradient_diff = result_2000_2010['Max Gradient'] - result_1970_1980['Max Gradient']

    results.append({'Model': result_2000_2010['Model'], 'Max Gradient Difference': max_gradient_diff})

# Convert results to a pandas DataFrame
max_g_df = pd.DataFrame(results)
#max_g_df = max_g_df.sort_values('Max Gradient Difference', ascending=False)



# Print table
print(max_g_df[['Model', 'Max Gradient Difference']])
max_g_df.to_excel('Max gradient difference.xlsx', index=False)


# In[ ]:


# Table S3 and S4: MLD in two time periods and the difference between then and the corresponding surface salinity for each model
import os
import xarray as xr
import numpy as np
import pandas as pd

input_folder ='/Volumes/Seagate Hub/cmip6_downloader/data/Salinity models/'
myfiles = [f for f in os.listdir(input_folder) if not f.startswith('._')]

fig, axs = plt.subplots(8, 4, layout='tight', figsize=(5 * 5, 5 * 8), sharex=True)
axs.reshape(-4)
axs[-1, -1].remove()
axs[-1, -2].remove()


def max_gradient_depth(file, start_year=1970, end_year=1980, max_depth=150):
    ds = xr.open_dataset(os.path.join(input_folder, file))
    if 'CESM' in file: 
        ds['lev'] = ds['lev']/100
    model_name = file.split('so.')[-1].split('.')[0]

    # Extract salinity data for the specified time period
    ds_mean = ds.so.sel(lev=slice(0, max_depth)).isel(time=np.arange((int(start_year)-1850)*12, (int(end_year+1)-1850)*12,1)).mean(dim='time')
    y = ds_mean.mean(dim=['longitude', 'latitude'])

    # Calculate salinity gradient (derivative)
    d_values = ds.lev.sel(lev=slice(0, max_depth)).values
    ds_derivative = np.gradient(y, d_values, axis=0)

    # Find maximum salinity gradient and depth
    max_gradient = ds_derivative[:-2].max()
    depth_max_gradient = d_values[ds_derivative[:-2].argmax()]

    # Extract salinity value at depth 0 for the specified time period
    salinity_at_depth_0 = ds['so'].sel(time=slice(start_year, end_year), lev=ds['lev'].sel(method='nearest', lev=0)).mean().values

    return {'Model': model_name, 'Max Gradient': max_gradient, 'Mixed layer depth': depth_max_gradient, 'Surface Salinity': salinity_at_depth_0}

# Create empty lists to store results for 1970 and 2010
results_1970 = []
results_2010 = []

# Loop through each file and calculate maximum salinity gradient and depth for 1970-1980 and 2000-2010
for file in myfiles:
    result_1970 = max_gradient_depth(file, start_year=1970, end_year=1980)
    results_1970.append(result_1970)
    
    result_2010 = max_gradient_depth(file, start_year=2000, end_year=2010)
    results_2010.append(result_2010)
# Convert results to pandas DataFrames
df_1970 = pd.DataFrame(results_1970)


df_2010 = pd.DataFrame(results_2010)

# Print tables
print("1970 Results:")
print(df_1970[['Model', 'Surface Salinity', 'Max Gradient', 'Mixed layer depth']])
print("\n2010 Results:")
print(df_2010[['Model', 'Surface Salinity', 'Max Gradient', 'Mixed layer depth']])

# Save results to Excel files
df_1970.to_excel('Mixed_layer_depth_1970.xlsx', index=False)
df_2010.to_excel('Mixed_layer_depth_2010.xlsx', index=False)



# In[ ]:


#Figure 13: Histograms of MLD in two time periods and histogram of difference between them
import matplotlib.pyplot as plt
import pandas as pd
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10, 8))
subplot_labels = ['a', 'b', 'c']

# Plot the 1970-1980 histogram
axes[0, 0].hist(df_1970['Mixed layer depth'], bins=10, color='blue')
axes[0, 0].set_title('MLD (1970-1980)')
axes[0, 0].set_xlabel('Mixed Layer Depth (m)')
axes[0, 0].set_ylabel('Frequency')
axes[0, 0].set_ylim(0,15)
axes[0, 0].axvline(df_observations['Depth of Max Gradient'][0], color='r', linestyle='--', label='MLD (AIDJEX)')

# Plot the 2000-2010 histogram
axes[0, 1].hist(df_2010['Mixed layer depth'], bins=10, color='green')
axes[0, 1].set_title('MLD (2000-2010)')
axes[0, 1].set_xlabel('Mixed Layer Depth (m)')
axes[0, 1].set_ylabel('Frequency')
axes[0, 1].set_ylim(0,14)
axes[0, 1].axvline(df_observations['Depth of Max Gradient'][1], color='darkorange', linestyle='--', label='MLD (ITP)')

# Plot the difference histogram
axes[1, 0].hist(mixed_layer_depth_diff_df['Mixed layer depth difference'], bins=10, color='purple')
axes[1, 0].set_title('Difference MLD (2000-2010 vs 1970-1980)')
axes[1, 0].set_xlabel('Mixed Layer Depth Difference between 1970-1980 and 2000-2012 (m)')
axes[1, 0].set_ylabel('Frequency')

# Remove the fourth subplot
fig.delaxes(axes[1,1])

# Add the vertical line for the difference between AIDJEX and ITP observations
mld_difference_observed = df_observations['Depth of Max Gradient'][1] - df_observations['Depth of Max Gradient'][0]
axes[1, 0].axvline(mld_difference_observed, color='blue', linestyle='--', label='Observed MLD Difference\n (ITP - AIDJEX)')

for i, ax in enumerate(axes.flatten()[:3]):
    ax.text(-0.07, 1.08, subplot_labels[i], transform=ax.transAxes, fontsize=12, fontweight='bold', va='top', ha='right')

axes[0, 0].legend(ncol=1, loc='upper right', fontsize=8)
axes[0, 1].legend(ncol=1, loc='upper right', fontsize=8)
axes[1, 0].legend(ncol=2, loc='upper right', fontsize=8)

plt.tight_layout()
plt.savefig('MLD_hist_comparison_subplots.png', bbox_inches='tight')
plt.show()


# In[ ]:


#Figure 14 and 15
import matplotlib.pyplot as plt
import pandas as pd

plt.scatter(df_saldif['Salinity difference'], mixed_layer_depth_diff_df['Mixed layer depth difference'])

# Set axis labels and title
plt.xlabel('Mixed layer depth')
plt.ylabel('Max Gradient')
plt.title('Max Gradient vs. Mixed layer depth in 2010')

plt.savefig('MLD V MG')
# Show plot
plt.show()

fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 10))
subplot_labels = ['a', 'b', 'c']

# Plot SS V MLD 2010
axes[0, 0].scatter(df_2010['Surface Salinity'], df_2010['Mixed layer depth'])
axes[0, 0].set_title('MLD Vs Suface Salinity (2010)')
axes[0, 0].set_ylabel('Mixed Layer Depth (m)')
axes[0, 0].set_xlabel('Surface salinity (g/kg)')

# Plot MG V MLD 2010
axes[0, 1].scatter(df_2010['Surface Salinity'], df_2010['Max Gradient']) 
axes[0, 1].set_title('Max Gradient Vs Surface Salinity (2010)')
axes[0, 1].set_xlabel('Surface salinity (g/kg)')
axes[0, 1].set_ylabel('Maximum gradient (g/kg/m)')

# Plot MLD diff V SS Diff
axes[1, 0].scatter(df_saldif['Salinity difference'], mixed_layer_depth_diff_df['Mixed layer depth difference'])
axes[1, 0].set_title('MLD difference Vs Salinity difference (1970-2010)')
axes[1, 0].set_ylabel('MLD difference (m)')
axes[1, 0].set_xlabel('Surface salinity difference (g/kg)')

# Remove the fourth subplot
axes[1, 1].remove()

# Add subplot labels
for i, ax in enumerate(axes.flatten()[:3]):
    ax.text(-0.05, 1.07, subplot_labels[i], transform=ax.transAxes, fontsize=12, fontweight='bold', va='top', ha='right')

plt.tight_layout()
plt.savefig('surface_salinity_comparison_subplots.png', bbox_inches='tight')
plt.show()


# In[ ]:


# Figure 16: Ocean component plot
import os
import numpy as np
import pandas as pd
import xarray as xr
import seaborn as sns
import matplotlib.pyplot as plt

input_folder = '/Volumes/Seagate Hub/cmip6_downloader/data/Salinity models/'
myfiles = os.listdir(input_folder)

for f in myfiles:
    if '._so' in f:
        myfiles.remove(f)

colors = sns.color_palette("husl", len(myfiles))
linestyles = ['-', '--', '-.', ':'] * (len(myfiles) // 4 + 1)

fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(25, 20), sharex=True)
plt.subplots_adjust(right=0.7, hspace=0.3)

def plot_salinity_profile(ax, file, color, linestyle, start_year, end_year):
    ds = xr.open_dataset(input_folder + file)
    if 'CESM' in file:
        ds['lev'] = ds['lev'] / 100

    model_name = file.split('so.')[-1].split('.')[0]
    ds_mean = ds.so.isel(time=np.arange((start_year-1850)*12, (end_year+1-1850)*12, 1)).mean(dim='time')
    y = ds_mean.mean(dim=['longitude', 'latitude'])

    ax.plot(y, ds['lev'], label=model_name, color=color, linewidth=2, linestyle=linestyle)

for file, color, linestyle in zip(myfiles, colors, linestyles):
    plot_salinity_profile(axs[0, 0], file, color, linestyle, 1970, 1980)
    plot_salinity_profile(axs[0, 1], file, color, linestyle, 2000, 2010)

axs[0, 0].plot(AJX_avg_salinity, depth_values, label='1975 AIDJEX', color='red', linewidth=4)
axs[0, 1].plot(ITP_avg_salinity, depth_values, label='2006-2012 ITP', color='orange', linewidth=4)

for i in range(2):
    for j in range(2):
        ax = axs[i, j]
        ax.set_ylabel('Depth (m)', fontsize=14)
        ax.set_xlabel('Salinity (g/kg)', fontsize=14)
        ax.set_ylim(0, 150)
        ax.set_xlim(27, 35)
        ax.tick_params(axis='y', labelsize=12)
        ax.tick_params(axis='x', labelsize=12)
        ax.invert_yaxis()

        # Add subplot labels
        label = chr(97 + i * 2 + j)  # Convert index to ASCII character ('a', 'b', etc.)
        ax.text(-0.1, 1, label, fontsize=19, fontweight='bold', transform=ax.transAxes)

axs[0, 0].set_title('1970-1980 Salinity Profiles', fontsize=20)
axs[0, 1].set_title('2000-2010 Salinity Profiles', fontsize=20)

def plot_salinity_profile_ocean_component(ax, file, color, start_year, end_year):
    ds = xr.open_dataset(input_folder + file)
    if 'CESM' in file:
        ds['lev'] = ds['lev'] / 100

    model_name = file.split('so.')[-1].split('.')[0]
    ds_mean = ds.so.isel(time=np.arange((start_year-1850)*12, (end_year+1-1850)*12, 1)).mean(dim='time')
    y = ds_mean.mean(dim=['longitude', 'latitude'])

    ax.plot(y, ds['lev'], label=model_name, color=color, linewidth=2, linestyle='-')

for file in myfiles:
    model_name = file.split('so.')[-1].split('.')[0]
    color = model_info_df.loc[model_info_df['Model Name'] == model_name, 'Color'].values[0]

    plot_salinity_profile_ocean_component(axs[1, 0], file, color, start_year=1970, end_year=1980)
    plot_salinity_profile_ocean_component(axs[1, 1], file, color, start_year=2000, end_year=2010)

axs[1, 0].plot(AJX_avg_salinity, depth_values, label='1975 AIDJEX', color='black', linewidth=4)
axs[1, 1].plot(ITP_avg_salinity, depth_values, label='2006-2012 ITP', color='black', linewidth=4)



# Get the handles and labels from the first subplot
handles, labels = axs[0, 0].get_legend_handles_labels()

# Add the handle and label for the ITP line from the second subplot
handles_itp, labels_itp = axs


# In[ ]:


#Figure 17: relationship between MLD difference and SS difference grouped by ocean component
import matplotlib.pyplot as plt
import seaborn as sns

# Set the style for the plot without gridlines
sns.set(style='ticks')

# Custom color mapping for the 'Ocean Component' column
ocean_component_colors = {
    'MOM4-L40': 'magenta',
    'MOM4': 'navy',
    'MOM6': 'saddlebrown',
    'NEMO3.6': 'green',
    'Nemo3.4.1': 'lime',
    'NEMO-HadGEM3-GO6.0': 'darkturquoise',
    'NEMO-LIM3.3.6': 'darkkhaki',
    'GISS Ocean': 'teal',
    'POP2': 'lightcoral',
    'MPIOM1.6.3': 'purple',
    'MRI.COM4.4': 'lightskyblue',
    'MPAS-Ocean': 'slategrey',
    'EC-Earth NEMO3.6': 'blue',
}

# Create a scatter plot of the MLD difference vs. Salinity difference, colored by Ocean Component
scatter_plot = sns.scatterplot(data=final_combined_df, x='Salinity difference', y='Mixed layer depth difference', hue='Ocean Component', palette=ocean_component_colors)

# Set the title and axis labels
scatter_plot.set_title('MLD difference Vs Salinity difference (1970-2010)')
scatter_plot.set_ylabel('MLD difference (m)')
scatter_plot.set_xlabel('Surface salinity difference (g/kg)')

# Move the legend outside the plot
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)

# Save the plot to a file
plt.savefig('MLDVSAldifoceancomponent', bbox_inches='tight')

# Display the plot
plt.show()


# In[ ]:


#work our phi for AIDJEX and ITP (Table S5)
import numpy as np

depth_range = (depth_values >= 0) & (depth_values <= 150)

#AIDJEX
AJX_salinity_depth_range = AJX_avg_salinity[depth_range]  #salinity profile up to 150 meters
AJX_Phi = np.sum(AJX_salinity_depth_range - 33) 

#ITP
ITP_salinity_depth_range = ITP_avg_salinity[depth_range]  #salinity profile up to 150 meters
ITP_Phi = np.sum(ITP_salinity_depth_range - 33)  

print("Phi for AIDJEX:", AJX_Phi)
print("Phi for ITP:", ITP_Phi)

# Calculate the ratio of Phi values for AIDJEX and ITP
Phi_ratio = ITP_Phi / AJX_Phi

print("Phi ratio (AIDJEX to ITP):", Phi_ratio)


# In[ ]:


#Figure 18: rescaled aidjex salinity profiles
import matplotlib.pyplot as plt
import numpy as np

ratio = ITP_Phi / AJX_Phi

depth_range = (depth_values >= 0) & (depth_values <= 150)
depth_range_values = depth_values[depth_range]

# rescaled AJX profile
rescaled_AJX_salinity_depth_range = (AJX_salinity_depth_range - 33) * ratio + 33

# Plot 
plt.plot(AJX_salinity_depth_range, depth_range_values, label='Original AIDJEX', color='red', linewidth=4)
plt.plot(rescaled_AJX_salinity_depth_range, depth_range_values, label='Rescaled AIDJEX', color='blue', linewidth=4)
plt.plot(ITP_salinity_depth_range, depth_range_values, label='ITP', color='orange', linewidth=4)

plt.gca().invert_yaxis() 
plt.xlabel('Salinity (g/kg)', fontsize =14)
plt.ylabel('Depth (m)', fontsize =14)
plt.title('Rescaled AIDJEX Salinity Profile', fontsize=15)
plt.legend()
plt.savefig('rescaled AJX.png')
plt.show()


# In[ ]:


#Figure 19: vertical salinity gradient rescaled
import matplotlib.pyplot as plt
import numpy as np

# Calculate derivatives for both datasets
dSI_dz_AJX = np.gradient(AJX_avg_salinity, depth_values)
dSI_dz_ITP = np.gradient(ITP_avg_salinity, depth_values)
dSI_rescaled_AJX_salinity_depth_range = np.gradient(rescaled_AJX_salinity_depth_range, depth_range_values)

# Plot the derivatives
plt.plot(dSI_dz_AJX, depth_values, label='1975 AIDJEX', color='red', linewidth=4)
plt.plot(dSI_dz_ITP, depth_values, label='2006-2012 ITP', color='orange', linewidth=4)
plt.plot(dSI_rescaled_AJX_salinity_depth_range, depth_range_values, label='Rescaled AIDJEX', color='blue', linewidth=4)

# Set y-axis limits and invert the y-axis
plt.ylim(150, 0)

# Add labels, a title, and a legend to the plot
plt.xlabel('Salinity Gradient (g/kg/m)', fontsize=14)
plt.ylabel('Depth (m)', fontsize=14)
plt.title('Rescaled Vertical Salinity Gradient', fontsize=15)
plt.legend()
plt.savefig('vertical salinity gradiet observations')
plt.show()


# In[ ]:


# Table S6:  number of vertical levels in top most 150 m 
import os
import xarray as xr
import pandas as pd

input_folder = '/Volumes/Seagate Hub/cmip6_downloader/data/Salinity models/'

# Get a list of all files in the folder
myfiles = [f for f in os.listdir(input_folder) if os.path.isfile(os.path.join(input_folder, f))]
for f in myfiles:
    if '._so' in f:
        myfiles.remove(f)

# Initialize an empty list to store the data
data = []

# Iterate over each file and calculate the number of levels in the top 150m
for filename in myfiles:
    ds = xr.open_dataset(input_folder + filename, engine='netcdf4')
    if 'CESM' in filename: 
        ds['lev'] = ds['lev']/100
    model_name = filename.split('so.')[-1].split('.')[0]
    num_levels = (ds.lev <= 150).sum().item()
    data.append({'Model Name': model_name, 'Levels in top 150m': num_levels})


df = pd.DataFrame(data)

# Display the DataFrame
print(df)

