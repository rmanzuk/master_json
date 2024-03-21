# script with sections do some initial data processing and PCA on the gridded data
# written by R. A. Manzuk 02/29/2024
# last updated 03/19/2024
# NOT USING ANYMORE, SPLIT INTO SEPARATE SCRIPTS
##########################################################################################
# package imports
##########################################################################################
# %%
import json # for json handling
import numpy as np # for array handling
from sklearn.decomposition import PCA # for PCA
import matplotlib # for color handling
import matplotlib.pyplot as plt # for plotting
import os # for file handling

#%%
##########################################################################################
# local function imports
##########################################################################################
# %% 
from json_processing import assemble_samples, select_gridded_geochem, select_gridded_im_metrics, select_gridded_point_counts
# %%
##########################################################################################
# script lines
##########################################################################################
# %% set up some plotting stuff

# define a default color order for plotting, from Paul Tol's "Colour Schemes"
# https://personal.sron.nl/~pault/
# and we'll use the same colors for the same things throughout the paper
indigo = '#332288'
cyan = '#88CCEE'
teal = '#44AA99'
green = '#117733'
olive = '#999933'
sand = '#DDCC77'
rose = '#CC6677'
wine = '#882255'
purple = '#AA4499'

muted_colors = [rose, indigo, sand, green, cyan, wine, teal, olive, purple]

# set the muted colors as the default color cycle
muted_cmap = matplotlib.colors.ListedColormap(muted_colors)
plt.rcParams['axes.prop_cycle'] = plt.cycler(color=muted_cmap.colors)

# and turn the grid on by default, with thin dotted lines
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.linestyle'] = ':'
plt.rcParams['grid.linewidth'] = 0.5

# %% define paths, and read in the outcrop json, and assemble samples

outcrop_json_file = '/Users/ryan/Dropbox (Princeton)/code/master_json/stewarts_mill.json'
sample_json_dir = '/Users/ryan/Dropbox (Princeton)/code/master_json/stewarts_mill_grid_samples/'

with open(outcrop_json_file, 'r') as f:
    outcrop_data = json.load(f)

outcrop_data = assemble_samples(outcrop_data, sample_json_dir, data_type=['grid_data'], data_name=["Stewart's Mill Grid"])


# %% select the gridded geochem data, im metrics, and point counts
geochem_df = select_gridded_geochem(outcrop_data, desired_metrics=['delta13c', 'delta18o', 'Li_Ca', 'Na_Ca', 'Mg_Ca', 'K_Ca', 'Mn_Ca', 'Fe_Ca', 'Sr_Ca'])

im_metric_df =  select_gridded_im_metrics(outcrop_data, desired_metrics=['percentile', 'rayleigh_anisotropy', 'entropy'], desired_scales=[1,0.5, 0.25, 0.125, 0.0625, 0.03125, 0.015625, 0.0078125, 0.00390625, 0.001953125])

point_count_df = select_gridded_point_counts(outcrop_data)

# and as a useful point of comparison, it may be helpul to have the field_lithology for each sample
all_sample_jsons = [os.path.join(sample_json_dir, x) for x in os.listdir(sample_json_dir) if x.endswith('.json')]
field_names = []
field_liths = []
for sample_json in all_sample_jsons:
    with open(sample_json, 'r') as f:
        sample_data = json.load(f)
    field_names.append(sample_data['sample_name'])
    field_liths.append(sample_data['field_lithology'])

unique_liths = np.unique(field_liths)

# %% Look at PCAs for color percentile spectra

# first need to extract the color percentile spectra from the im_metric_df
# we'll make a 3D array that is n_samples x n_percentiles x n_bands, which we can do a PCA on each band

# get a list of the unique samples
unique_samples = im_metric_df.sample_name.unique()

# look in the metrics to find the unique percentiles, which are listed as 'percentile_XX'
all_metrics = im_metric_df.metric_name
percentile_metrics = all_metrics[all_metrics.str.contains('percentile')]
unique_percentiles = percentile_metrics.unique()
# and extract just the number and sort them
unique_percentiles = np.sort([int(x.split('_')[1]) for x in unique_percentiles])

# get the unique bands
unique_bands = im_metric_df.wavelength.unique()

# and flag if we want the normalized or unnormalized spectra
normalized = False

# make a 3D array to hold the data
n_samples = len(unique_samples)
n_percentiles = len(unique_percentiles)
n_bands = len(unique_bands)

percentile_spectra = np.zeros((n_samples, n_percentiles, n_bands))

# and a separate one to hold geographic information for each sample
sample_lat_lon_msl = np.zeros((n_samples, 3))

# iterate through all rows of the im_metric_df
for index, row in im_metric_df.iterrows():
    # first check if this metric is a percentile, otherwise skip
    if 'percentile' in row['metric_name'] and row['normalized'] == normalized:
        # get the sample index
        sample_index = np.where(unique_samples == row['sample_name'])[0][0]
        # get the percentile index
        percentile_index = np.where(unique_percentiles == int(row['metric_name'].split('_')[1]))[0][0]
        # get the band index
        band_index = np.where(unique_bands == row['wavelength'])[0][0]
        # put the value in the array
        percentile_spectra[sample_index, percentile_index, band_index] = row['value']

        # if this is the first time we've seen this sample, get the lat, lon, and msl
        if sample_lat_lon_msl[sample_index, 0] == 0:
            sample_lat_lon_msl[sample_index, 0] = row['latitude']
            sample_lat_lon_msl[sample_index, 1] = row['longitude']
            sample_lat_lon_msl[sample_index, 2] = row['msl']

# normalize the spectra matrix to have a mean of 0 and a standard deviation of 1
#percentile_spectra = (percentile_spectra - np.nanmean(percentile_spectra, axis=0))/np.nanstd(percentile_spectra, axis=0)

# there should be no variance in the 100th percentile, so we can remove it
percentile_spectra = percentile_spectra[:,:-1,:]

# now we can do a PCA on each band
n_components = percentile_spectra.shape[1]
pca_list = []
for band_index in range(n_bands):
    pca = PCA(n_components=n_components)
    pca.fit(percentile_spectra[:,:,band_index])
    pca_list.append(pca)

# and make a new 3d array to hold the PCA reprjections
percentile_scores = np.zeros((n_samples, n_components, n_bands))
for band_index in range(n_bands):
    percentile_scores[:,:,band_index] = pca_list[band_index].transform(percentile_spectra[:,:,band_index])

# make an array to hold the explained variance
explained_variance = np.zeros((n_bands, n_components))
for band_index in range(n_bands):
    explained_variance[band_index,:] = pca_list[band_index].explained_variance_ratio_

# make an array to hold the loadings
loadings = np.zeros((n_bands, n_components, n_components))
for band_index in range(n_bands):
    loadings[band_index,:,:] = pca_list[band_index].components_


# %% plot explained variance as an image
fig, ax = plt.subplots(1,1, figsize=(5,5))
ax.imshow(explained_variance[:,:3], aspect='equal')
ax.set_xlabel('Component')
ax.set_ylabel('Band')
ax.set_title('Explained Variance')
# make a colorbar
cbar = plt.colorbar(ax.imshow(explained_variance[:,:3], aspect='auto'), ax=ax)
# label the y ticks with the band names
ax.set_yticks(np.arange(n_bands))
ax.set_yticklabels(unique_bands)
ax.set_xticks(np.arange(3))
ax.set_xticklabels(['PC1', 'PC2', 'PC3'])
plt.show()

# %% bar plot of the loadings for the first three components

# make a bar chart for each band showing the loadings for the first three components
fig, ax = plt.subplots(3,3, figsize=(15,10))
for band_index in range(n_bands):
    for component in range(3):
        ax[int(band_index/3), band_index%3].bar(unique_percentiles[:-1]+(component), loadings[band_index,component,:], width = 1)
    ax[int(band_index/3), band_index%3].set_title('Band: '+str(int(unique_bands[band_index])))
    ax[int(band_index/3), band_index%3].set_xlabel('Percentile')   
    ax[int(band_index/3), band_index%3].set_ylabel('Loading')

    # ensure labels don't overlap
    plt.tight_layout()

    # make a legend in the first plot
    if band_index == 0:
        ax[int(band_index/3), band_index%3].legend(['PC1', 'PC2', 'PC3'])


plt.show()

# %% We'll make a plot of end member spectra for each band for the first 3 components

# take the mean spectra, just as a point of comparison
mean_spectra = np.mean(percentile_spectra, axis=0)

# make the figure
fig, ax = plt.subplots(3,3, figsize=(15,8))
for band_index in range(n_bands):

    # plot the highest and lowest scoring spectra for each component
    line_styles = ['-', '--', '-.']
    for i in range(3):
        # get the highest and lowest scoring spectra for this component
        pc_sort = np.argsort(percentile_scores[:,i,band_index])
        ax[int(band_index/3), band_index%3].plot(unique_percentiles[:-1], percentile_spectra[pc_sort[-1],:,band_index], label='PC'+str(i+1)+' High', linestyle=line_styles[i], color='black')
        ax[int(band_index/3), band_index%3].plot(unique_percentiles[:-1], percentile_spectra[pc_sort[0],:,band_index], label='PC'+str(i+1)+' Low', linestyle=line_styles[i], color='red')

        # and label the spectra with their names
        ax[int(band_index/3), band_index%3].text(unique_percentiles[-1], percentile_spectra[pc_sort[-1],-1,band_index], unique_samples[pc_sort[-1]], fontsize=8)
        ax[int(band_index/3), band_index%3].text(unique_percentiles[-1], percentile_spectra[pc_sort[0],-1,band_index], unique_samples[pc_sort[0]], fontsize=8)

    # plot the mean spectrum
    ax[int(band_index/3), band_index%3].plot(unique_percentiles[:-1], mean_spectra[:,band_index], label='Mean', linestyle='-', color='gray')

    # label the plot
    ax[int(band_index/3), band_index%3].set_title('Band: '+str(int(unique_bands[band_index])))
    ax[int(band_index/3), band_index%3].set_xlabel('Percentile')
    ax[int(band_index/3), band_index%3].set_ylabel('Value')
    
    # if this is the first plot, make a legend
    if band_index == 0:
        ax[int(band_index/3), band_index%3].legend()

    # ensure labels don't overlap
    plt.tight_layout()

plt.show()



# %% list out the names of the samples with the highest and lowest PC2 and PC3 scores

band_index = 4

# to associate these pc scores with things like field lithology and location, need to compare the sample names to the unique samples to get the indices
# we need to know where unique_samples is in the sample_names list
field_index = np.zeros(n_samples)
for i in range(n_samples):
    field_index[i] = np.where(np.array(field_names) == unique_samples[i])[0][0]

pc1_sort = np.argsort(percentile_scores[:,0,band_index])
pc2_sort = np.argsort(percentile_scores[:,1,band_index])
pc3_sort = np.argsort(percentile_scores[:,2,band_index])
print('Samples with the highest PC1 scores:')
for i in range(5):
    print(field_names[int(field_index[pc1_sort[-(i+1)]])])
print('Samples with the lowest PC1 scores:')
for i in range(5):
    print(field_names[int(field_index[pc1_sort[i]])])
print('Samples with the highest PC2 scores:')
for i in range(5):
    print(field_names[int(field_index[pc2_sort[-(i+1)]])])
print('Samples with the lowest PC2 scores:')
for i in range(5):
    print(field_names[int(field_index[pc2_sort[i]])])
print('Samples with the highest PC3 scores:')
for i in range(5):
    print(field_names[int(field_index[pc3_sort[-(i+1)]])])
print('Samples with the lowest PC3 scores:')
for i in range(5):
    print(field_names[int(field_index[pc3_sort[i]])])

# %% make a cross plot of any 2 components for each band (on same figure), color code by field lithology

pc_to_plot = [0,2]

# to associate these pc scores with things like field lithology and location, need to compare the sample names to the unique samples to get the indices
# we need to know where unique_samples is in the sample_names list
field_index = np.zeros(n_samples)
for i in range(n_samples):
    field_index[i] = np.where(np.array(field_names) == unique_samples[i])[0][0]


# make a cross plot of the first two components for each band (on same figure)
# color code by field lithology
# first give each row in the array a lithology index
lith_index = np.zeros(n_samples)
for i in range(n_samples):
    lith_index[i] = np.where(unique_liths == field_liths[int(field_index[i])])[0][0]

# then make the plot
fig, ax = plt.subplots(3,3, figsize=(15,15))
for band_index in range(n_bands):
    for lith in range(len(unique_liths)):
        # turn the grid on
        ax[int(band_index/3), band_index%3].grid(True)
        # and plot
        ax[int(band_index/3), band_index%3].scatter(percentile_scores[lith_index == lith, pc_to_plot[0], band_index], percentile_scores[lith_index == lith, pc_to_plot[1], band_index], label=unique_liths[lith])

        # put sample names on the plot
        #for i in range(n_samples):
            #ax[int(band_index/3), band_index%3].text(percentile_scores[i, pc_to_plot[0], band_index], percentile_scores[i, pc_to_plot[1], band_index], sample_names[int(field_index[i])], fontsize=5)

    ax[int(band_index/3), band_index%3].set_title('Band: '+str(unique_bands[band_index]))
    ax[int(band_index/3), band_index%3].set_xlabel('PC '+str(pc_to_plot[0]+1))  
    ax[int(band_index/3), band_index%3].set_ylabel('PC '+str(pc_to_plot[1]+1))
    
    # make a legend on the blank plot in the lower right
    if band_index == n_bands-1:
        ax[int(band_index/3), band_index%3].legend()

    # ensure labels don't overlap
    plt.tight_layout()
plt.show()

# %%
# plot the samples geographically, color coded by pc1, 2, and 3


# then make the plot
fig, ax = plt.subplots(1,3, figsize=(15,5))
for i in range(3):
    ax[i].scatter(sample_lat_lon_msl[:,1], sample_lat_lon_msl[:,0], c=percentile_scores[:,i,0])
    ax[i].set_xlabel('Longitude')
    ax[i].set_ylabel('Latitude')
    ax[i].set_title('PC'+str(i+1))
    # add a colorbar
    cbar = plt.colorbar(ax[i].scatter(sample_lat_lon_msl[:,1], sample_lat_lon_msl[:,0], c=percentile_scores[:,i,0]), ax=ax[i])
plt.show()
 
# %% Look at PCAs for entropy scale spectra

# first get all the unique scales that entropy was calculated at
# get the index of all rows where the metric name is entropy
all_metrics = im_metric_df.metric_name
entropy_inds = [x for x in range(len(all_metrics)) if 'entropy' in all_metrics[x]]

unique_scales = np.unique([im_metric_df.scale[x] for x in entropy_inds])

unique_samples = im_metric_df.sample_name.unique()

# and the unique bands it was calculated at
unique_bands = np.unique([im_metric_df.wavelength[x] for x in entropy_inds])

# make an array to hold the entropy spectra
n_samples = len(unique_samples)
n_scales = len(unique_scales)
n_bands = len(unique_bands)

entropy_spectra = np.zeros((n_samples, n_scales, n_bands))

# iterate through all rows of the im_metric_df
for index, row in im_metric_df.iterrows():
    # first check if this metric is entropy, otherwise skip
    if 'entropy' in row['metric_name']:
        # get the sample index
        sample_index = np.where(unique_samples == row['sample_name'])[0][0]
        # get the scale index
        scale_index = np.where(unique_scales == row['scale'])[0][0]
        # get the band index
        band_index = np.where(unique_bands == row['wavelength'])[0][0]
        # put the value in the array
        entropy_spectra[sample_index, scale_index, band_index] = row['value']

# normalize the spectra matrix to have a mean of 0 and a standard deviation of 1
original_entropy_spectra = entropy_spectra.copy()
entropy_spectra = (entropy_spectra - np.mean(entropy_spectra, axis=0))/np.std(entropy_spectra, axis=0)

# now we can do a PCA on each band
n_components = entropy_spectra.shape[1]
pca_list = []
for band_index in range(n_bands):
    pca = PCA(n_components=n_components)
    pca.fit(entropy_spectra[:,:,band_index])
    pca_list.append(pca)

# and make a new 3d array to hold the PCA reprjections
entropy_scores = np.zeros((n_samples, n_components, n_bands))
for band_index in range(n_bands):
    entropy_scores[:,:,band_index] = pca_list[band_index].transform(entropy_spectra[:,:,band_index])

# make an array to hold the explained variance
explained_variance = np.zeros((n_bands, n_components))
for band_index in range(n_bands):
    explained_variance[band_index,:] = pca_list[band_index].explained_variance_ratio_
    
# make an array to hold the loadings
loadings = np.zeros((n_bands, n_components, n_scales))
for band_index in range(n_bands):
    loadings[band_index,:,:] = pca_list[band_index].components_

# to associate these pc scores with things like field lithology and location, need to compare the sample names to the unique samples to get the indices
# we need to know where unique_samples is in the sample_names list
field_index = np.zeros(n_samples)
for i in range(n_samples):
    field_index[i] = np.where(np.array(field_names) == unique_samples[i])[0][0]
    
# also give each row a lithology index
lith_index = np.zeros(n_samples)
for i in range(n_samples):
    lith_index[i] = np.where(unique_liths == field_liths[int(field_index[i])])[0][0]

# %%
# make a bar chart for the one band showing the loadings for the first three components

fig, ax = plt.subplots(1,1, figsize=(15,5))
for component in range(3):
    ax.bar(np.linspace(0,len(unique_scales),len(unique_scales))+(component*0.1), loadings[0,component,:], width = 0.1)

ax.set_xlabel('Scale')
ax.set_ylabel('Loading')
ax.legend(['PC1', 'PC2', 'PC3'])

# adjust the ticks to be the scale values, but evenly spaced
ax.set_xticks(np.linspace(0,len(unique_scales),len(unique_scales)))
ax.set_xticklabels(unique_scales)

plt.show()

# %% make a plot showing the spectra for the top 3 and bottom 3 samples for each component

# make the figure
fig, ax = plt.subplots(1,3, figsize=(15,5))

# plot the highest and lowest scoring spectra for each component
line_styles = ['-', '--', '-.']
for i in range(3):
    # get the highest and lowest scoring spectra for this component
    pc_sort = np.argsort(entropy_scores[:,i,0])
    
    # plot the 3 highest and lowest scoring spectra for each component
    for j in range(3):
        ax[i].plot(unique_scales, original_entropy_spectra[pc_sort[-(j+1)],:,0], label='PC'+str(i+1)+' High', linestyle=line_styles[j], color='black')
        ax[i].plot(unique_scales, original_entropy_spectra[pc_sort[j],:,0], label='PC'+str(i+1)+' Low', linestyle=line_styles[j], color='red')

        # and label the spectra with their names
        ax[i].text(unique_scales[-1], original_entropy_spectra[pc_sort[-(j+1)],-1,0], unique_samples[pc_sort[-(j+1)]], fontsize=8)
        ax[i].text(unique_scales[-1], original_entropy_spectra[pc_sort[j],-1,0], unique_samples[pc_sort[j]], fontsize=8)

    # label the plot
    ax[i].set_title('PC'+str(i+1))
    ax[i].set_xlabel('Scale')
    ax[i].set_ylabel('Value')
    ax[i].legend()

    # make the x scale logarithmic
    ax[i].set_xscale('log')

plt.tight_layout()
plt.show()


# %% 

# %% Look at PCAs for anisotropy scale spectra

# first get all the unique scales that entropy was calculated at
# get the index of all rows where the metric name is entropy
all_metrics = im_metric_df.metric_name
anisotropy_inds = [x for x in range(len(all_metrics)) if 'rayleigh_anisotropy' in all_metrics[x]]

unique_scales = np.unique([im_metric_df.scale[x] for x in anisotropy_inds])

# and the unique bands it was calculated at
unique_bands = np.unique([im_metric_df.wavelength[x] for x in anisotropy_inds])

unique_samples = im_metric_df.sample_name.unique()

# make an array to hold the entropy spectra
n_samples = len(unique_samples)
n_scales = len(unique_scales)
n_bands = len(unique_bands)

anisotropy_spectra = np.zeros((n_samples, n_scales, n_bands))

# iterate through all rows of the im_metric_df
for index, row in im_metric_df.iterrows():
    # first check if this metric is entropy, otherwise skip
    if 'rayleigh_anisotropy' in row['metric_name']:
        # get the sample index
        sample_index = np.where(unique_samples == row['sample_name'])[0][0]
        # get the scale index
        scale_index = np.where(unique_scales == row['scale'])[0][0]
        # get the band index
        band_index = np.where(unique_bands == row['wavelength'])[0][0]
        # put the value in the array
        anisotropy_spectra[sample_index, scale_index, band_index] = row['value']

# normalize the spectra matrix to have a mean of 0 and a standard deviation of 1
original_anisotropy_spectra = anisotropy_spectra.copy()
anisotropy_spectra = (anisotropy_spectra - np.mean(anisotropy_spectra, axis=0))/np.std(anisotropy_spectra, axis=0)

# now we can do a PCA on each band
n_components = anisotropy_spectra.shape[1]
pca_list = []
for band_index in range(n_bands):
    pca = PCA(n_components=n_components)
    pca.fit(anisotropy_spectra[:,:,band_index])
    pca_list.append(pca)

# and make a new 3d array to hold the PCA reprjections
anisotropy_scores = np.zeros((n_samples, n_components, n_bands))
for band_index in range(n_bands):
    anisotropy_scores[:,:,band_index] = pca_list[band_index].transform(anisotropy_spectra[:,:,band_index])

# make an array to hold the explained variance
explained_variance = np.zeros((n_bands, n_components))
for band_index in range(n_bands):
    explained_variance[band_index,:] = pca_list[band_index].explained_variance_ratio_

# make an array to hold the loadings
loadings = np.zeros((n_bands, n_components, n_scales))
for band_index in range(n_bands):
    loadings[band_index,:,:] = pca_list[band_index].components_

# to associate these pc scores with things like field lithology and location, need to compare the sample names to the unique samples to get the indices
# we need to know where unique_samples is in the sample_names list
field_index = np.zeros(n_samples)
for i in range(n_samples):
    field_index[i] = np.where(np.array(field_names) == unique_samples[i])[0][0]

# also give each row a lithology index
lith_index = np.zeros(n_samples)
for i in range(n_samples):
    lith_index[i] = np.where(unique_liths == field_liths[int(field_index[i])])[0][0]

# %%

# make a bar chart for just the 1 band we have, with the first 3 components, with log scale
    
fig, ax = plt.subplots(1,1, figsize=(15,5))
for component in range(3):
    ax.bar(np.linspace(0,len(unique_scales),len(unique_scales))+(component*0.1), loadings[0,component,:], width = 0.1)

ax.set_xlabel('Scale')
ax.set_ylabel('Loading')
ax.legend(['PC1', 'PC2', 'PC3'])

# adjust the ticks to be the scale values, but evenly spaced
ax.set_xticks(np.linspace(0,len(unique_scales),len(unique_scales)))
ax.set_xticklabels(unique_scales)

plt.show()
# %% make a plot showing the spectra for the top 3 and bottom 3 samples for each component

# make the figure
fig, ax = plt.subplots(1,3, figsize=(15,5))

# plot the highest and lowest scoring spectra for each component
line_styles = ['-', '--', '-.']
for i in range(3):
    # get the highest and lowest scoring spectra for this component
    pc_sort = np.argsort(anisotropy_scores[:,i,0])
    
    # plot the 3 highest and lowest scoring spectra for each component
    for j in range(3):
        ax[i].plot(unique_scales, original_anisotropy_spectra[pc_sort[-(j+1)],:,0], label='PC'+str(i+1)+' High', linestyle=line_styles[j], color='black')
        ax[i].plot(unique_scales, original_anisotropy_spectra[pc_sort[j],:,0], label='PC'+str(i+1)+' Low', linestyle=line_styles[j], color='red')

        # and label the spectra with their names
        ax[i].text(unique_scales[-1], original_anisotropy_spectra[pc_sort[-(j+1)],-1,0], unique_samples[pc_sort[-(j+1)]], fontsize=8)
        ax[i].text(unique_scales[-1], original_anisotropy_spectra[pc_sort[j],-1,0], unique_samples[pc_sort[j]], fontsize=8)

    # label the plot
    ax[i].set_title('PC'+str(i+1))
    ax[i].set_xlabel('Scale')
    ax[i].set_ylabel('Value')
    ax[i].legend()

    # make the x scale logarithmic
    ax[i].set_xscale('log')

plt.tight_layout()
plt.show()

# %% Look at PCAs for point count fractions

# this df is mostly ready to go, just need to extract data, make some adjustments, and do a PCA
#pc_classes = point_count_df.columns[4:] 
pc_samples = point_count_df.sample_name

# going to manually select classes for now
pc_classes = ['Microb', 'Spar', 'Dol', 'Arch', 'Mi', 'ooid']

# extract the data into an array
pc_data = point_count_df[pc_classes].to_numpy()

# we don't want to the collumn thats None, so we'll remove it, redo the fractions so each row sums to 1, and then do a PCA
#none_inds = []
#for i in range(len(pc_classes)):
    #if pc_classes[i] is None:
        #none_inds.append(i)

#pc_classes = np.delete(pc_classes, none_inds[0])
#pc_data = np.delete(pc_data, none_inds[0], axis=1)

# before redoing fractions, replace nans with zeros
pc_data = np.nan_to_num(pc_data)

# make the rows sum to 1 again
pc_data = pc_data/np.sum(pc_data, axis=1)[:,None]

# now normalize and do a PCA
pc_data_original = pc_data.copy()
pc_data = (pc_data - np.mean(pc_data, axis=0))/np.std(pc_data, axis=0)

pca = PCA(n_components=len(pc_classes))
pca.fit(pc_data)
pc_scores = pca.transform(pc_data)
pc_loadings = pca.components_
explained_variance = pca.explained_variance_ratio_

# associate these samples with lithology and location
field_index = np.zeros(len(pc_samples))
for i in range(len(pc_samples)):
    field_index[i] = np.where(np.array(field_names) == pc_samples[i])[0][0]

lith_index = np.zeros(len(pc_samples))
for i in range(len(pc_samples)):
    lith_index[i] = np.where(unique_liths == field_liths[int(field_index[i])])[0][0]

# %% make a bar chart for the loadings for the first five components
    
fig, ax = plt.subplots(1,1, figsize=(15,5))
for component in range(6):
    ax.bar(np.linspace(0,len(pc_classes),len(pc_classes))+(component*0.1), pc_loadings[component], width = 0.1)

ax.set_xlabel('Class')
# set the x ticks to be the string names of the classes
ax.set_xticks(np.linspace(0,len(pc_classes),len(pc_classes)))
ax.set_xticklabels(pc_classes)
ax.set_ylabel('Loading')

# make a legend
ax.legend(['PC1', 'PC2', 'PC3', 'PC4', 'PC5', 'PC6'])
plt.show()

# %% list the top 5 samples for each component

pc1_sort = np.argsort(pc_scores[:,0])
pc2_sort = np.argsort(pc_scores[:,1])
pc3_sort = np.argsort(pc_scores[:,2])
pc4_sort = np.argsort(pc_scores[:,3])
pc5_sort = np.argsort(pc_scores[:,4])


print('Samples with the highest PC1 scores:')
for i in range(5):
    print(field_names[int(field_index[pc1_sort[-(i+1)]])])
print('Samples with the lowest PC1 scores:')
for i in range(5):
    print(field_names[int(field_index[pc1_sort[i]])])
print('Samples with the highest PC2 scores:')
for i in range(5):
    print(field_names[int(field_index[pc2_sort[-(i+1)]])])
print('Samples with the lowest PC2 scores:')
for i in range(5):
    print(field_names[int(field_index[pc2_sort[i]])])
print('Samples with the highest PC3 scores:')
for i in range(5):
    print(field_names[int(field_index[pc3_sort[-(i+1)]])])
print('Samples with the lowest PC3 scores:')
for i in range(5):
    print(field_names[int(field_index[pc3_sort[i]])])
print('Samples with the highest PC4 scores:')
for i in range(5):
    print(field_names[int(field_index[pc4_sort[-(i+1)]])])
print('Samples with the lowest PC4 scores:')
for i in range(5):
    print(field_names[int(field_index[pc4_sort[i]])])
print('Samples with the highest PC5 scores:')
for i in range(5):
    print(field_names[int(field_index[pc5_sort[-(i+1)]])])
print('Samples with the lowest PC5 scores:')
for i in range(5):
    print(field_names[int(field_index[pc5_sort[i]])])


# %%  make cross plots for all combinations of the first 5 components, color coded by lithology
    
fig, ax = plt.subplots(6,6, figsize=(15,15))
for i in range(6):
    for j in range(6):
        for lith in range(len(unique_liths)):
            ax[i,j].scatter(pc_scores[lith_index == lith, i], pc_scores[lith_index == lith, j], label=unique_liths[lith])

        ax[i,j].set_xlabel('PC'+str(i+1))
        ax[i,j].set_ylabel('PC'+str(j+1))
        if i == 0 and j == 0:
            ax[i,j].legend()

plt.tight_layout()

plt.show()
 
# %% make cross plot of any 2 PCs, color coded by lithology, samples labeled with their names

pc_to_plot = [1,2]

fig, ax = plt.subplots(1,1, figsize=(8,5))
for lith in range(len(unique_liths)):
    ax.scatter(pc_scores[lith_index == lith, pc_to_plot[0]], pc_scores[lith_index == lith, pc_to_plot[1]], label=unique_liths[lith])
    # put sample names on the plot, need to get integer index of the field_index
    good_inds = np.where(lith_index == lith)[0]
    for i in range(len(good_inds)):
        ax.text(pc_scores[good_inds[i], pc_to_plot[0]], pc_scores[good_inds[i], pc_to_plot[1]], field_names[int(field_index[good_inds[i]])], fontsize=5)

ax.set_xlabel('PC'+str(pc_to_plot[0]+1))
ax.set_ylabel('PC'+str(pc_to_plot[1]+1))
# put the legend outside the plot
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
# and make sure it still shows up
plt.tight_layout()


plt.show()
# %% For each PC, make a plot of the spectra for the top 3 and bottom 3 samples

# make the figure
fig, ax = plt.subplots(2, 3, figsize=(15,10))

# for each component
for i in range(6):
    # get the highest and lowest scoring spectra for this component
    pc_sort = np.argsort(pc_scores[:,i])
    for j in range(3):
        # plot the highest and lowest scoring spectra for each component
        ax[int(i/3), i%3].plot(np.linspace(0,len(pc_classes),len(pc_classes)), pc_data_original[pc_sort[-(j+1)],:], label='High')
        ax[int(i/3), i%3].plot(np.linspace(0,len(pc_classes),len(pc_classes)), pc_data_original[pc_sort[j],:], label='Low')

    # make the x ticks the string names of the classes
    ax[int(i/3), i%3].set_xticks(np.linspace(0,len(pc_classes),len(pc_classes)))
    ax[int(i/3), i%3].set_xticklabels(pc_classes)

    # label the plot
    ax[int(i/3), i%3].set_title('PC'+str(i+1))
    ax[int(i/3), i%3].set_xlabel('Class')
    ax[int(i/3), i%3].set_ylabel('Fraction')
    ax[int(i/3), i%3].legend()

    # ensure labels don't overlap
    plt.tight_layout()


plt.show()


# %% Look at PCAs for geochem data

# going to be kind of similar to the point count data, just extract data, normalize, and do a PCA
geochem_measurements = geochem_df.columns[5:]
geochem_samples = geochem_df.sample_name
geochem_phases = geochem_df.phase

geochem_data = geochem_df[geochem_measurements].to_numpy()

# we want completeness, so remove any columns that have over 10% nans
nan_fracs = np.sum(np.isnan(geochem_data), axis=0)/geochem_data.shape[0]
good_inds = np.where(nan_fracs < 0.1)
geochem_measurements = geochem_measurements[good_inds]
geochem_data = geochem_data[:,good_inds[0]]

# and now remove any rows that have nans
nan_rows = np.where(np.sum(np.isnan(geochem_data), axis=1) > 0)
geochem_data = np.delete(geochem_data, nan_rows, axis=0)
geochem_samples = np.delete(geochem_samples, nan_rows)
geochem_phases = np.delete(geochem_phases, nan_rows)

# last thing is to pull out the carbon isotopes, which are the first column into their own array, delete them from the data for now
carbon_isotopes = geochem_data[:,0]
geochem_data = np.delete(geochem_data, 0, axis=1)
geochem_measurements = np.delete(geochem_measurements, 0)

# now normalize and do a PCA, storing the standard deviations and means for later
geochem_data_original = geochem_data.copy()
geochem_data = (geochem_data - np.mean(geochem_data, axis=0))/np.std(geochem_data, axis=0)

pca = PCA(n_components=len(geochem_measurements))
pca.fit(geochem_data)
geochem_scores = pca.transform(geochem_data)
geochem_loadings = pca.components_
explained_variance = pca.explained_variance_ratio_

# bring in the phase names
phase_codes = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K']
phase_names = ['Ooid', 'Sparry mud/cement', 'Dolomitized mud', 'Calcite vein', 'Micrite', 'Regular archaeocyath', 'Irregular archaeocyath', 'Calcimicrobe', 'Shell', 'Microbial', 'Coralomorph']

# and store some stats about the original data, so we can recreate it later
original_means = np.mean(geochem_data_original, axis=0)
original_stds = np.std(geochem_data_original, axis=0)
original_mins = np.min(geochem_data_original, axis=0)
original_maxs = np.max(geochem_data_original, axis=0)
# %% plot end member geochem spectra for the first 4

# store the max and min values for each pc scores
pc_maxs = np.max(geochem_scores, axis=0)
pc_mins = np.min(geochem_scores, axis=0)

# make endmember spectra by multiplying the max and min scores by the loadings
max_endmembers = geochem_loadings*pc_maxs
min_endmembers = geochem_loadings*pc_mins

# make the figure
fig, ax = plt.subplots(1,1, figsize=(5,5))

# plot the end member spectra
for i in range(4):
    ax.plot(np.linspace(0,len(geochem_measurements),len(geochem_measurements)), max_endmembers[i,:], label='PC'+str(i+1), linewidth=2)   

# make the x ticks the string names of the classes
ax.set_xticks(np.linspace(0,len(geochem_measurements),len(geochem_measurements)))
ax.set_xticklabels(geochem_measurements)

# label the plot
ax.set_xlabel('Measurement')
ax.set_ylabel('Relative value')
ax.legend()

plt.show()

# %% visualize how the linear combination of the pcs can make back the original data

# this will be a 3 panel plot, with the original data, the contributions from the first 4 pcs, and the reconstructed spectra
sample_no = 10

# make the figure
fig, ax = plt.subplots(1,3, figsize=(15,5))

# plot the original data
ax[0].plot(np.linspace(0,len(geochem_measurements),len(geochem_measurements)), geochem_data[sample_no,:], label='Original', linewidth=2)

# plot the original data, but also with the contributions from the first 4 pcs, properly scaled by score
n_comps = 4
reconstructed = np.zeros((len(geochem_measurements), n_comps))
for i in range(n_comps):
    reconstructed[:,i] = geochem_loadings[i,:]*geochem_scores[sample_no,i]
    # round the score for this pc to 2 decimal places
    ax[1].plot(np.linspace(0,len(geochem_measurements),len(geochem_measurements)), reconstructed[:,i], label='PC'+str(i+1)+' score = '+str(np.round(geochem_scores[sample_no,i], 2)), linewidth=2)

# plot the reconstructed data, which is the sum of the contributions from the first 4 pcs
ax[2].plot(np.linspace(0,len(geochem_measurements),len(geochem_measurements)), geochem_data[sample_no,:], label='Original', linewidth=4, color = 'gray')
ax[2].plot(np.linspace(0,len(geochem_measurements),len(geochem_measurements)), np.sum(reconstructed[:,:2], axis=1), label='PC 1,2 reconstruction', linewidth=2)
ax[2].plot(np.linspace(0,len(geochem_measurements),len(geochem_measurements)), np.sum(reconstructed, axis=1), label='PC 1-4 reconstruction', linewidth=2)

    
# make the x ticks the string names of the classes
for i in range(3):
    ax[i].set_xticks(np.linspace(0,len(geochem_measurements),len(geochem_measurements)))
    ax[i].set_xticklabels(geochem_measurements)

    # label the plot
    ax[i].set_xlabel('Measurement')
    ax[i].set_ylabel('Relative value')
    ax[i].legend()

plt.show()


# %% make a crossplot of any 2 pcs, color coded by phase, and with dot sizes proportional to the misfit when reconstructing the original data with those 2 pcs

pc_to_plot = [0,1]

# make the misfits
misfits = np.zeros((len(geochem_samples), len(geochem_measurements)))
for i in range(len(geochem_samples)):
    for j in range(len(geochem_measurements)):
        misfits[i,j] = np.sum((np.dot(geochem_scores[i,0:2], geochem_loadings[0:2,:]) - geochem_data[i,:])**2)
        
# we may run out of colors before getting through phases, so list a few symbols to switch to once we run out of colors
symbols = ['o', 's', 'D', 'v', '^', '<', '>', 'p', 'P', '*', 'X']
n_colors = len(plt.rcParams['axes.prop_cycle'].by_key()['color'])

# make the cross plot
fig, ax = plt.subplots(1,1, figsize=(8,5))
count = 1
for phase in np.unique(geochem_phases):
    if count >= n_colors:
        ax.scatter(geochem_scores[geochem_phases == phase, pc_to_plot[0]], geochem_scores[geochem_phases == phase, pc_to_plot[1]], label=phase, s=10*misfits[geochem_phases == phase, 0], marker=symbols[count-n_colors])
    else:
        ax.scatter(geochem_scores[geochem_phases == phase, pc_to_plot[0]], geochem_scores[geochem_phases == phase, pc_to_plot[1]], label=phase, s=10*misfits[geochem_phases == phase, 0])
    count += 1

ax.set_xlabel('PC'+str(pc_to_plot[0]+1))
ax.set_ylabel('PC'+str(pc_to_plot[1]+1))

# make a legend, bring in the phase names, put it outside the plot, and make sure it still shows up
ax.legend(phase_names, loc='center left', bbox_to_anchor=(1, 0.5))

plt.tight_layout()

plt.show()

# %% make a crossplot of any 2 pcs, allow user to input vectors from the plot, and then make a nice crossplot with spectra reconstructed from those vectors

pc_to_plot = [2,3]

# make the misfits
misfits = np.zeros((len(geochem_samples), len(geochem_measurements)))
for i in range(len(geochem_samples)):
    for j in range(len(geochem_measurements)):
        misfits[i,j] = np.sum((np.dot(geochem_scores[i,0:2], geochem_loadings[pc_to_plot,:]) - geochem_data[i,:])**2)

# how many reconstructed spectra to show
n_recon = 5

# and we'll show them with a sequential color map
cmap = plt.get_cmap('viridis')
cmap_vals = np.linspace(0,1,n_recon)

# first just show the user the plot, no need to color code by phase or anything
fig, ax = plt.subplots(1,1, figsize=(8,5))
ax.scatter(geochem_scores[:, pc_to_plot[0]], geochem_scores[:, pc_to_plot[1]])
ax.set_xlabel('PC'+str(pc_to_plot[0]+1))
ax.set_ylabel('PC'+str(pc_to_plot[1]+1))

# then ask the user how many vectors they want to input
n_vectors = int(input('How many vectors would you like to input? '))
vectors = []

# now ginput n_vectors * 2 times, and store the vectors
print('Click on the plot to input vectors')
for i in range(n_vectors):
    vectors.append(plt.ginput(2))

# get the n_recon points in the cross plot along the vectors 
recon_points_x = np.zeros((n_vectors, n_recon))
recon_points_y = np.zeros((n_vectors, n_recon))
for i in range(n_vectors):
    # just need to linspace between end points of the vectors
    recon_points_x[i] = np.linspace(vectors[i][0][0], vectors[i][1][0], n_recon)
    recon_points_y[i] = np.linspace(vectors[i][0][1], vectors[i][1][1], n_recon)

# now that we have the recon points, use them in pc space to make spectra
recon_spectra = np.zeros((geochem_data.shape[1], n_recon, n_vectors))

for i in range(n_vectors):
    for j in range(n_recon):
        reconstructed_x = geochem_loadings[pc_to_plot[0],:]*recon_points_x[i,j]
        reconstructed_y = geochem_loadings[pc_to_plot[1],:]*recon_points_y[i,j]
        recon_spectra[:,j,i] = reconstructed_x + reconstructed_y

# and now make a pretty multipanel plot. The furthest left panel will be the color, size coded cross plot, and the rest will be the reconstructed spectra
fig, ax = plt.subplots(1, n_vectors+1, figsize=(5*(n_vectors+1),5)) 

# we may run out of colors before getting through phases, so list a few symbols to switch to once we run out of colors
symbols = ['o', 's', 'D', 'v', '^', '<', '>', 'p', 'P', '*', 'X']
n_colors = len(plt.rcParams['axes.prop_cycle'].by_key()['color'])

# make the color coded cross plot
count = 1
for phase in np.unique(geochem_phases):
    if count >= n_colors:
        ax[0].scatter(geochem_scores[geochem_phases == phase, pc_to_plot[0]], geochem_scores[geochem_phases == phase, pc_to_plot[1]], label=phase, s=10*misfits[geochem_phases == phase, 0], marker=symbols[count-n_colors])
    else:
        ax[0].scatter(geochem_scores[geochem_phases == phase, pc_to_plot[0]], geochem_scores[geochem_phases == phase, pc_to_plot[1]], label=phase, s=10*misfits[geochem_phases == phase, 0])
    count += 1

ax[0].set_xlabel('PC'+str(pc_to_plot[0]+1))
ax[0].set_ylabel('PC'+str(pc_to_plot[1]+1))
ax[0].legend(phase_names, loc='center left', bbox_to_anchor=(1, 0.5))

# also add the vectors to the plot, as arrows
for i in range(n_vectors):
    ax[0].arrow(vectors[i][0][0], vectors[i][0][1], vectors[i][1][0]-vectors[i][0][0], vectors[i][1][1]-vectors[i][0][1], head_width=0.1, head_length=0.1, fc='black', ec='black')

# now make the reconstructed spectra, we'll just have all 5 on 1 plot, but with a vertical offset
for i in range(n_vectors):
    for j in range(n_recon):
        ax[i+1].plot(np.linspace(0,len(geochem_measurements),len(geochem_measurements)), recon_spectra[:,j,i]+j*0.1, label='Recon '+str(j+1), color=cmap(cmap_vals[j]))
    ax[i+1].set_xticks(np.linspace(0,len(geochem_measurements),len(geochem_measurements)))
    ax[i+1].set_xticklabels(geochem_measurements)
    ax[i+1].set_xlabel('Measurement')
    ax[i+1].set_ylabel('Relative value')
    ax[i+1].legend()

plt.tight_layout()

plt.show()



# %%

# make 2 subplots, 1 for pc1 and carbon isotopes, and 1 for pc2 and carbon isotopes


# we may run out of colors before getting through phases, so list a few symbols to switch to once we run out of colors
symbols = ['o', 's', 'D', 'v', '^', '<', '>', 'p', 'P', '*', 'X']
n_colors = len(plt.rcParams['axes.prop_cycle'].by_key()['color'])

fig, ax = plt.subplots(1,2, figsize=(10,5))
count = 1
for phase in np.unique(geochem_phases):
    if count >= n_colors:
        ax[0].scatter(geochem_scores[geochem_phases == phase, 0], carbon_isotopes[geochem_phases == phase], label=phase, marker=symbols[count-n_colors])
        ax[1].scatter(geochem_scores[geochem_phases == phase, 1], carbon_isotopes[geochem_phases == phase], label=phase, marker=symbols[count-n_colors])
    else:
        ax[0].scatter(geochem_scores[geochem_phases == phase, 0], carbon_isotopes[geochem_phases == phase], label=phase)
        ax[1].scatter(geochem_scores[geochem_phases == phase, 1], carbon_isotopes[geochem_phases == phase], label=phase)
    count += 1

ax[0].set_xlabel('PC1')
ax[0].set_ylabel('Carbon isotopes')
ax[1].set_xlabel('PC2')
ax[1].set_ylabel('Carbon isotopes')
ax[0].legend(phase_names, loc='center left', bbox_to_anchor=(1, 0.5))

plt.tight_layout()

plt.show()



# %% make a 3d scatter plot of pc1, pc2, and carbon isotopes

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
for phase in np.unique(geochem_phases):
    ax.scatter(geochem_scores[geochem_phases == phase, 0], geochem_scores[geochem_phases == phase, 1], geochem_data[geochem_phases == phase, 0], label=phase)

ax.set_xlabel('PC1')
ax.set_ylabel('PC2')
ax.set_zlabel('Carbon isotopes')
ax.legend(phase_names, loc='center left', bbox_to_anchor=(1, 0.5))
plt.tight_layout()

plt.show()

# %%

# to make geographic plots, just need to take lat and long from the df and remove any rows that were removed from the data
geochem_lats = geochem_df.latitude
geochem_lons = geochem_df.longitude
geochem_lats = np.delete(geochem_lats, nan_rows)
geochem_lons = np.delete(geochem_lons, nan_rows)

# then make the plot
fig, ax = plt.subplots(1,3, figsize=(15,5))
for i in range(3):
    ax[i].scatter(geochem_lons, geochem_lats, c=geochem_scores[:,i])
    ax[i].set_xlabel('Longitude')
    ax[i].set_ylabel('Latitude')
    ax[i].set_title('PC'+str(i+1))
    # add a colorbar
    cbar = plt.colorbar(ax[i].scatter(geochem_lons, geochem_lats, c=geochem_scores[:,i]), ax=ax[i])
plt.show()

