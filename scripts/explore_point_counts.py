# script with sections to explore point count data
# written by R. A. Manzuk 05/10/2024
# last updated 05/10/2024

##########################################################################################
# package imports
##########################################################################################
# %%
import matplotlib.pyplot as plt # for plotting
import matplotlib # for custom color maps
import json # for json handling
import numpy as np # for array operations
#%%
##########################################################################################
# local function imports
##########################################################################################
# %% 
from json_processing import assemble_samples, select_gridded_point_counts
from custom_plotting import display_point_counts

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

# define a path to save figures
out_path = '/Users/ryan/Dropbox (Princeton)/figures/reef_survey/point_count method/'


# %% pick out a sample an display its point counts

sample_num = 'smg_43'
all_sample_nums = [outcrop_data['grid_data'][0]['samples'][i]['sample_name'] for i in range(len(outcrop_data['grid_data'][0]['samples']))]
sample_index = all_sample_nums.index(sample_num)

# and just run the function
display_point_counts(outcrop_data['grid_data'][0]['samples'][sample_index], no_classes=True)

# and save the figure
plt.savefig(out_path + 'point_count_example.pdf', dpi=300)

# %% assemble all point count modalities from all samples, get rid of NaNs, so we can plot violins of abundances

# we'll just get the gridded point count data
point_count_df = select_gridded_point_counts(outcrop_data)

# extract an array of just the point count fraction parts of the df, turn NaNs to 0, and make each row sum to 1 again
good_cats = ['Mi','Dol','Fe', 'Stylo', 'Microb', 'Intra', 'UnID fos', 'Spar', 'ooid', 'Brach', 'Trilo', 'Arch', 'UnID other', 'Sal', 'Glau', 'echi', 'Qtz']
pc_data = point_count_df[good_cats].to_numpy()
pc_data = np.nan_to_num(pc_data)
pc_data = pc_data/np.sum(pc_data, axis=1)[:,None]

# and split the data based upon having a max greater than 0.2
abundance_mean_threshold = 0.02
abundant_classes = np.where(np.mean(pc_data, axis=0) > abundance_mean_threshold)[0]
less_abundant_classes = np.where(np.mean(pc_data, axis=0) <= abundance_mean_threshold)[0]
abundant_data = pc_data[:,abundant_classes]
less_abundant_data = pc_data[:,less_abundant_classes]
abundant_labels = np.array(good_cats)[abundant_classes]
less_abundant_labels = np.array(good_cats)[less_abundant_classes]

# get orders for the classes basued upon their mean abundance, reversed
abundant_order = np.argsort(np.mean(abundant_data, axis=0))[::-1]
less_abundant_order = np.argsort(np.mean(less_abundant_data, axis=0))[::-1]


# 2 subplots, one for the abundant classes, one for the less abundant classes
fig, axs = plt.subplots(1,2, figsize=(8,5))

# make a violin plot of the abundant classes, in order of median abundance
# make the violins light grey with black outlines
abundant_vp = axs[0].violinplot(abundant_data[:,abundant_order], showmedians=False, showextrema=False)
for pc in abundant_vp['bodies']:
    pc.set_facecolor('lightgrey')
    pc.set_edgecolor('black')
axs[0].set_xticks(np.arange(1,len(abundant_labels)+1))
axs[0].set_xticklabels(abundant_labels[abundant_order], rotation=90)
axs[0].set_ylabel('Fraction of point count')
axs[0].set_title('Abundant classes')
axs[0].set_ylim(0,1)
axs[0].set_yticks([0,0.25,0.5,0.75,1])

# make a violin plot of the less abundant classes, in order of median abundance
la_vp = axs[1].violinplot(less_abundant_data[:,less_abundant_order], showmedians=False, showextrema=False)
for pc in la_vp['bodies']:
    pc.set_facecolor('lightgrey')
    pc.set_edgecolor('black')
axs[1].set_xticks(np.arange(1,len(less_abundant_labels)+1))
axs[1].set_xticklabels(less_abundant_labels[less_abundant_order], rotation=90)
axs[1].set_title('Less abundant classes')
axs[1].set_ylim(0,0.2)
axs[1].set_yticks([0,0.05,0.1,0.15,0.2])

plt.tight_layout()

# and save the figure
plt.savefig(out_path + 'point_count_violins.pdf', dpi=300)

