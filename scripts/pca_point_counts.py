# script with sections to look at PCA of point count data
# written by R. A. Manzuk 03/19/2024
# last updated 03/19/2024

##########################################################################################
# package imports
##########################################################################################
# %%
import json # for json handling
import numpy as np # for array handling
from sklearn.decomposition import PCA # for PC
import matplotlib.pyplot as plt # for plotting
import matplotlib # for plotting
import os # for file handling

#%%
##########################################################################################
# local function imports
##########################################################################################
# %% 
from json_processing import assemble_samples, select_gridded_point_counts
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
# %% Look at PCAs for point count fractions

# this df is mostly ready to go, just need to extract data, make some adjustments, and do a PCA
#pc_classes = point_count_df.columns[4:] 
pc_samples = point_count_df.sample_name

# going to manually select classes for now
pc_classes = ['Microb', 'Spar', 'Dol', 'Arch', 'Mi', 'ooid']

# extract the data into an array
pc_data = point_count_df[pc_classes].to_numpy()

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
pc_explained_variance = pca.explained_variance_ratio_

# %% we won't be using the PCA of point counts for the final figure, and we need to justify that

# so we'll just make 2 bar plots, one of the explained variance in the dataset of the PCs
# and one of the explained variance in the dataset of the original data

# calculate the explained variance of the original data
original_explained_variance = np.var(pc_data_original, axis=0)/np.sum(np.var(pc_data_original, axis=0))

# and put it in order from largest to smallest
order_indices = np.argsort(original_explained_variance)[::-1]
original_explained_variance = original_explained_variance[order_indices]

#reorder the classes
pc_classes = np.array(pc_classes)[order_indices]

fig, ax = plt.subplots(1, 2, figsize=(6, 3))
ax[0].bar(np.arange(len(pc_explained_variance)), pc_explained_variance*100)
ax[0].set_xticks(np.arange(len(pc_explained_variance)))
ax[0].set_xticklabels(np.arange(1, len(pc_explained_variance)+1))
ax[0].set_xlabel('Principal Component')
ax[0].set_ylabel('Explained Variance (%)')
ax[0].set_title('Explained Variance of Principal Components')

ax[1].bar(np.arange(len(original_explained_variance)), original_explained_variance*100)
ax[1].set_xticks(np.arange(len(original_explained_variance)))
ax[1].set_xticklabels(pc_classes[order_indices])
ax[1].set_xlabel('Original Feature')
ax[1].set_ylabel('Explained Variance (%)')
ax[1].set_title('Explained Variance of Original Features')

# get a common set of y limits
max_pct = np.max([np.max(pc_explained_variance*100), np.max(original_explained_variance*100)])
ax[0].set_ylim([0, max_pct])
ax[1].set_ylim([0, max_pct])

plt.tight_layout()


# export the figure
save_path = '/Users/ryan/Dropbox (Princeton)/figures/reef_survey/pca_result_point_counts'
plt.savefig(os.path.join(save_path, 'explained_variance_bars.pdf'), dpi=300)
#plt.show()
