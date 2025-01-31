# script with sections to look at PCA of GLCM contrast
# written by R. A. Manzuk 06/18/2024
# last updated 06/18/2024

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
im_metric_df =  select_gridded_im_metrics(outcrop_data, desired_metrics=['percentile', 'rayleigh_anisotropy', 'entropy', 'glcm_contrast'], desired_scales=[1,0.5, 0.25, 0.125, 0.0625, 0.03125, 0.015625, 0.0078125, 0.00390625, 0.001953125])


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
contrast_inds = [x for x in range(len(all_metrics)) if 'glcm_contrast' in all_metrics[x]]

unique_scales = np.unique([im_metric_df.scale[x] for x in contrast_inds])

# and the unique bands it was calculated at
unique_bands = np.unique([im_metric_df.wavelength[x] for x in contrast_inds])

unique_samples = im_metric_df.sample_name.unique()

# make an array to hold the entropy spectra
n_samples = len(unique_samples)
n_scales = len(unique_scales)
n_bands = len(unique_bands)

contrast_spectra = np.zeros((n_samples, n_scales, n_bands))

# iterate through all rows of the im_metric_df
for index, row in im_metric_df.iterrows():
    # first check if this metric is entropy, otherwise skip
    if 'glcm_contrast' in row['metric_name']:
        # get the sample index
        sample_index = np.where(unique_samples == row['sample_name'])[0][0]
        # get the scale index
        scale_index = np.where(unique_scales == row['scale'])[0][0]
        # get the band index
        band_index = np.where(unique_bands == row['wavelength'])[0][0]
        # put the value in the array
        contrast_spectra[sample_index, scale_index, band_index] = row['value']

# normalize the spectra matrix to have a mean of 0 and a standard deviation of 1
original_contrast_spectra = contrast_spectra.copy()
contrast_spectra = (contrast_spectra - np.mean(contrast_spectra, axis=0))/np.std(contrast_spectra, axis=0)

# now we can do a PCA on each band
n_components = contrast_spectra.shape[1]
pca_list = []
for band_index in range(n_bands):
    pca = PCA(n_components=n_components)
    pca.fit(contrast_spectra[:,:,band_index])
    pca_list.append(pca)

# and make a new 3d array to hold the PCA reprojections
contrast_scores = np.zeros((n_samples, n_components, n_bands))
for band_index in range(n_bands):
    contrast_scores[:,:,band_index] = pca_list[band_index].transform(contrast_spectra[:,:,band_index])

# make an array to hold the explained variance
contrast_explained_variance = np.zeros((n_bands, n_components))
for band_index in range(n_bands):
    contrast_explained_variance[band_index,:] = pca_list[band_index].explained_variance_ratio_

# make an array to hold the loadings
contrast_loadings = np.zeros((n_bands, n_components, n_scales))
for band_index in range(n_bands):
    contrast_loadings[band_index,:,:] = pca_list[band_index].components_

# %% plot the loadings and spectra of high/low scorers

# make 2 subplots. On the left, plot the loadings for the first 3 PCs
# on the right, plot the spectra of the samples with the highest and lowest scores for the first 3 PCs
# make a figure
fig, ax = plt.subplots(1,2, figsize=(6,2))

# plot the loadings, on a bar plot with the x axis as the scales (log scale), and the different component bar charts slightly offset
offset = 0.2
bar_width = 0.2
for pc_index in range(3):
    for band_index in range(n_bands):
        # we can just do it with an even scale on the x, and then label it like it's log
        # put the % variance explained in the label, no need to put the band index
        ax[0].bar(np.arange(n_scales) + offset*pc_index, contrast_loadings[band_index,pc_index,:], width=bar_width, label='PC %d, %.2f pct explained' % (pc_index+1, contrast_explained_variance[band_index,pc_index]*100))
    ax[0].set_xlabel('Scale')
    ax[0].set_ylabel('Loading')
    ax[0].set_title('PC Loadings')
    ax[0].legend()

# set the x ticks as if they were log
ax[0].set_xticks(np.arange(n_scales))
ax[0].set_xticklabels(unique_scales)


# plot the spectra of the high and low scorers
# we'll need 3 line styles for the 3 PCs
line_styles = ['-', '--', ':']
for pc_index in range(3):
    # get the indices of the high and low scorers
    high_scorer = np.argmax(contrast_scores[:,pc_index,0])
    low_scorer = np.argmin(contrast_scores[:,pc_index,0])
    # plot the high scorer on a log scale as a black line of the appropriate style
    ax[1].plot(unique_scales, contrast_spectra[high_scorer,:,0], color='k', linestyle=line_styles[pc_index], label='PC %d High Scorer' % (pc_index+1))
    # plot the low scorer on a log scale as a gray dashed line of the appropriate style
    ax[1].plot(unique_scales, contrast_spectra[low_scorer,:,0], color='gray', linestyle=line_styles[pc_index], label='PC %d Low Scorer' % (pc_index+1))
    ax[1].set_xlabel('Scale')
    ax[1].set_ylabel('GLCM Contrast')
    ax[1].set_title('High and Low Scorer Spectra')
    ax[1].legend()
    # set the x axis to log
    ax[1].set_xscale('log')

plt.tight_layout()


# # save the figure
export_path = '/Users/ryan/Dropbox (Princeton)/figures/reef_survey/pca_result_contrast/'
plt.savefig(export_path + 'pca_contrast_plots.pdf', dpi=300)

# print out the sample numbers of the 3 high and low scorers
print('High Scorer PC1: %s' % unique_samples[np.argmax(contrast_scores[:,0,0])])
print('High Scorer PC2: %s' % unique_samples[np.argmax(contrast_scores[:,1,0])])
print('High Scorer PC3: %s' % unique_samples[np.argmax(contrast_scores[:,2,0])])
print('Low Scorer PC1: %s' % unique_samples[np.argmin(contrast_scores[:,0,0])])
print('Low Scorer PC2: %s' % unique_samples[np.argmin(contrast_scores[:,1,0])])
print('Low Scorer PC3: %s' % unique_samples[np.argmin(contrast_scores[:,2,0])])

#%% print out second highest scorer in pc2

print('Second Highest Scorer PC2: %s' % unique_samples[np.argsort(contrast_scores[:,1,0])[-16]])