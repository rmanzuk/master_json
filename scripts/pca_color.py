# script with sections to look at PCA of color percentile spectra
# written by R. A. Manzuk 03/19/2024
# last updated 03/19/2024

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
from json_processing import assemble_samples, select_gridded_im_metrics
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


# %% select the gridded im metrics
im_metric_df =  select_gridded_im_metrics(outcrop_data, desired_metrics=['percentile', 'rayleigh_anisotropy', 'entropy'], desired_scales=[1,0.5, 0.25, 0.125, 0.0625, 0.03125, 0.015625, 0.0078125, 0.00390625, 0.001953125])


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
 