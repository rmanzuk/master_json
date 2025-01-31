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
import os # for file handling
import matplotlib.pyplot as plt # for plotting
import matplotlib # for color handling

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

# make fonts work out for Adobe Illustrator
plt.rcParams['pdf.fonttype'] = 42

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
# but then adjust the unique percentiles and n_percentiles
unique_percentiles = unique_percentiles[:-1]
n_percentiles = len(unique_percentiles)

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
percentile_explained_variance = np.zeros((n_bands, n_components))
for band_index in range(n_bands):
    percentile_explained_variance[band_index,:] = pca_list[band_index].explained_variance_ratio_

# make an array to hold the loadings
percentile_loadings = np.zeros((n_bands, n_components, n_components))
for band_index in range(n_bands):
    percentile_loadings[band_index,:,:] = pca_list[band_index].components_

# %% plot the loadings and spectra of high/low scorers for 590nm as an example



# which means band index 4
band_index = 4

# make 2 subplots. On the left, plot the loadings for the first 3 PCs
# on the right, plot the spectra of the samples with the highest and lowest scores for the first 3 PCs
# make a figure
fig, ax = plt.subplots(1,2, figsize=(6,2))

# plot the loadings, on a bar plot with the x axis as the percentiles
offset = 0.2
bar_width = 0.2
for pc_index in range(3):
    ax[0].bar(np.arange(n_percentiles) + offset*pc_index, percentile_loadings[band_index,pc_index,:], width=bar_width, label='PC %d, %.2f pct explained' % (pc_index+1, percentile_explained_variance[band_index,pc_index]*100))
ax[0].set_xlabel('Scale')
ax[0].set_ylabel('Loading')
ax[0].set_title('PC Loadings')
ax[0].legend()


# plot the spectra of the high and low scorers
# we'll need 3 line styles for the 3 PCs
line_styles = ['-', '--', ':']
for pc_index in range(3):
    # get the indices of the high and low scorers
    high_scorer = np.argmax(percentile_scores[:,pc_index,band_index])
    low_scorer = np.argmin(percentile_scores[:,pc_index,band_index])
    # plot the high scorer on a log scale as a black line of the appropriate style
    ax[1].plot(unique_percentiles, percentile_spectra[high_scorer,:,0], color='k', linestyle=line_styles[pc_index], label='PC %d High Scorer' % (pc_index+1))
    # plot the low scorer on a log scale as a gray dashed line of the appropriate style
    ax[1].plot(unique_percentiles, percentile_spectra[low_scorer,:,0], color='gray', linestyle=line_styles[pc_index], label='PC %d Low Scorer' % (pc_index+1))
    ax[1].set_xlabel('Scale')
    ax[1].set_ylabel('GLCM Contrast')
    ax[1].set_title('High and Low Scorer Spectra')
    ax[1].legend()
    # set the x axis to log

plt.tight_layout()


# # save the figure
export_path = '/Users/ryan/Dropbox (Princeton)/figures/reef_survey/pca_result_color/'
plt.savefig(export_path + 'pca_color_plots.pdf', dpi=300)

# print out the sample numbers of the 3 high and low scorers
print('High Scorer PC1: %s' % unique_samples[np.argmax(percentile_scores[:,0,band_index])])
print('High Scorer PC2: %s' % unique_samples[np.argmax(percentile_scores[:,1,band_index])])
print('High Scorer PC3: %s' % unique_samples[np.argmax(percentile_scores[:,2,band_index])])
print('Low Scorer PC1: %s' % unique_samples[np.argmin(percentile_scores[:,0,band_index])])
print('Low Scorer PC2: %s' % unique_samples[np.argmin(percentile_scores[:,1,band_index])])
print('Low Scorer PC3: %s' % unique_samples[np.argmin(percentile_scores[:,2,band_index])])

# what is the second highest scorer for PC1?
print('Second Highest Scorer PC1: %s' % unique_samples[np.argsort(percentile_scores[:,0,band_index])[-2]])