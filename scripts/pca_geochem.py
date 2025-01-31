# script with sections to look at PCA of geochemical data
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
import matplotlib.colors # for color handling

#%%
##########################################################################################
# local function imports
##########################################################################################
# %% 
from json_processing import assemble_samples, select_gridded_geochem
from custom_plotting import pc_crossplot_vectors
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
geochem_df = select_gridded_geochem(outcrop_data, desired_metrics=['delta13c', 'delta18o', 'Li_Ca', 'Na_Ca', 'Mg_Ca', 'K_Ca', 'Mn_Ca', 'Fe_Ca', 'Sr_Ca'])


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
# %% Look at PCAs for geochem data

# going to be kind of similar to the point count data, just extract data, normalize, and do a PCA
geochem_measurements = geochem_df.columns[7:]
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
geochem_explained_variance = pca.explained_variance_ratio_

# bring in the phase names
phase_codes = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K']
phase_names = ['Ooid', 'Sparry mud/cement', 'Dolomitized mud', 'Calcite vein', 'Micrite', 'Regular archaeocyath', 'Irregular archaeocyath', 'Calcimicrobe', 'Shell', 'Microbial', 'Coralomorph']

# and store some stats about the original data, so we can recreate it later
original_means = np.mean(geochem_data_original, axis=0)
original_stds = np.std(geochem_data_original, axis=0)
original_mins = np.min(geochem_data_original, axis=0)
original_maxs = np.max(geochem_data_original, axis=0)

# %% get the min and max scores for pcs 1 and 2, but in the unnormalized space

# to avoid a few outliers, we'll just hand select the min and max scores
min_score_1 = -1.8
max_score_1 = 11
min_score_2 = -3
max_score_2 = 4.5

min_score_1_reproj = min_score_1*geochem_loadings[0,:]*original_stds + original_means
max_score_1_reproj = max_score_1*geochem_loadings[0,:]*original_stds + original_means
min_score_2_reproj = min_score_2*geochem_loadings[1,:]*original_stds + original_means
max_score_2_reproj = max_score_2*geochem_loadings[1,:]*original_stds + original_means

# we'll make a subplot for each of the geochem measurements, and plot arrows from the min to the max score for each pc
fig, axs = plt.subplots(1, len(min_score_1_reproj), figsize=(15, 5))
for i in range(len(min_score_1_reproj)):
    axs[i].arrow(1, min_score_1_reproj[i], 0, max_score_1_reproj[i]-min_score_1_reproj[i], head_width=0.1, head_length=0.1, fc='black', ec='black')
    axs[i].arrow(2, min_score_2_reproj[i], 0, max_score_2_reproj[i]-min_score_2_reproj[i], head_width=0.1, head_length=0.1, fc='black', ec='black')
    axs[i].set_title(geochem_measurements[i])

plt.tight_layout()

# save the figure
out_path = '/Users/ryan/Dropbox (Princeton)/figures/reef_survey/pca_result_geochem/'
plt.savefig(out_path+'geochem_pc_arrows.pdf', dpi=300)

# %% cross plot the first two pcs, colored by phase

# but we'll group archaeocyaths together, microbial with calcimicrobes, and coralomorphs with shells
grouped_codes = geochem_phases.copy()
grouped_codes[np.isin(grouped_codes, ['F', 'G'])] = 'F'
grouped_codes[np.isin(grouped_codes, ['H', 'J'])] = 'J'
grouped_codes[np.isin(grouped_codes, ['I', 'K'])] = 'I'

fig, ax = plt.subplots(1,1, figsize=(5,5))

new_unique_codes = np.unique(grouped_codes)

for point in range(geochem_scores.shape[0]):
    ax.scatter(geochem_scores[point,0], geochem_scores[point,1], color=muted_colors[np.where(new_unique_codes == grouped_codes[point])[0][0]], label=phase_names[phase_codes.index(grouped_codes[point])])
# scramble up the zorder so that the points don't overlap weirdly

ax.set_xlabel('PC1')
ax.set_ylabel('PC2')
ax.legend()

# set x and y limits
ax.set_xlim([-2,12])
ax.set_ylim([-4,6])

plt.tight_layout()

# save the figure
out_path = '/Users/ryan/Dropbox (Princeton)/figures/reef_survey/pca_result_geochem/'
#plt.savefig(out_path+'geochem_pc_crossplot.pdf', dpi=300)

plt.show()

# %% make box and whisker plots of the first two pcs, divided by phase

fig, axs = plt.subplots(1,2, figsize=(10,5))

# box plots, but don't show outliers, and orient them horizontally
axs[0].boxplot([geochem_scores[grouped_codes == x, 0] for x in new_unique_codes], labels=[phase_names[phase_codes.index(x)] for x in new_unique_codes], showfliers=False, vert=False)
axs[0].set_ylabel('PC1')
axs[0].set_title('PC1 by Phase')

axs[1].boxplot([geochem_scores[grouped_codes == x, 1] for x in new_unique_codes], labels=[phase_names[phase_codes.index(x)] for x in new_unique_codes], showfliers=False, vert=False)
axs[1].set_ylabel('PC2')
axs[1].set_title('PC2 by Phase')

plt.tight_layout()

plt.show()

# %% make a scatter plot of any pc against carbon isotopes, colored by phase

pc_num = 1

# we may run out of colors before getting through phases, so list a few symbols to switch to once we run out of colors
symbols = ['o', 's', 'D', 'v', '^', '<', '>', 'p', 'P', '*', 'X']
n_colors = len(plt.rcParams['axes.prop_cycle'].by_key()['color'])


fig, ax = plt.subplots(1,1, figsize=(8,5))
count = 1
for phase in np.unique(geochem_phases):
    if count >= n_colors:
        ax.scatter(geochem_scores[geochem_phases == phase, pc_num], carbon_isotopes[geochem_phases == phase], label=phase_names[phase_codes.index(phase)], marker=symbols[count-n_colors])
    else:
        ax.scatter(geochem_scores[geochem_phases == phase, pc_num], carbon_isotopes[geochem_phases == phase], label=phase_names[phase_codes.index(phase)])
    count += 1

# also get the correlation coefficient with all the data, and plot the line
correlation = np.corrcoef(geochem_scores[:,pc_num], carbon_isotopes)
m, b = np.polyfit(geochem_scores[:,pc_num], carbon_isotopes, 1)
ax.plot(np.linspace(min(geochem_scores[:,pc_num]), max(geochem_scores[:,pc_num]), 100), m*np.linspace(min(geochem_scores[:,pc_num]), max(geochem_scores[:,pc_num]), 100) + b, label='r = '+str(np.round(correlation[0,1], 2)), color='black', linestyle='--')

ax.set_xlabel('PC'+str(pc_num+1))
ax.set_ylabel('Carbon Isotopes')
# make a legend with the phase names and the correlation coefficient
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
#ax.legend(phase_names, loc='center left', bbox_to_anchor=(1, 0.5))

plt.tight_layout()

plt.show()

# %% make a pc crossplot to explore with vectors

# input the pcs to use and a couple other things
pc_nums = [1,2] # these are 1 indexed
n_vectors = 2
n_recon = 5

# all we should need for prep is a list of the phase names for each sample
phase_tags = geochem_phases
sample_phase_names = [phase_names[phase_codes.index(x)] for x in phase_tags]

# neet to turn geochem_measurements into a list of strings
geochem_labels = [x for x in geochem_measurements]

# run the function
pc_crossplot_vectors(geochem_scores, geochem_loadings, pc_nums, variable_labels=geochem_labels, phase_tags=sample_phase_names, n_vectors=n_vectors, n_recon=n_recon)
