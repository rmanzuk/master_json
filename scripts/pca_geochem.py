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

# %% for a given pc, plot the reconstructed spectrum ranging from the highest to lowest score

# parameters to set
pc_num = 0
num_spectra = 5

# get the scores spaced out
score_range = np.linspace(min(geochem_scores[:,pc_num]), max(geochem_scores[:,pc_num]), num_spectra)

# make a sequential colormap to pull from for each line, but cut off the first few colors because they are too light
cmap = plt.get_cmap('PuBuGn')
colors = [cmap(i) for i in np.linspace(0.1, 1, num_spectra)]

# make the figure
fig, ax = plt.subplots(1,1, figsize=(5,3))



# plot the end member spectra
for i in range(num_spectra):
    ax.plot(np.linspace(0,len(geochem_measurements),len(geochem_measurements)), geochem_loadings[pc_num,:]*score_range[i], label='PC'+str(i+1), linewidth=2, color=colors[i])  

# make the x ticks the string names of the classes
ax.set_xticks(np.linspace(0,len(geochem_measurements),len(geochem_measurements)))
# if the geochem measurement has an underscore, replace it with a /
geochem_labels = [x.replace('_', '/') for x in geochem_measurements]
# and if the label is delta18o, replace it with the delta symbol, and superscript the 18, capitalizing the O
geochem_labels = [x.replace('delta18o', '$\delta^{18}$O') for x in geochem_labels]
ax.set_xticklabels(geochem_labels)

# label the plot
ax.set_xlabel('Measurement')
ax.set_ylabel('Relative value')

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
