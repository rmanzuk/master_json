# script with sections to look at PCA of anisotropy scale spectra
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