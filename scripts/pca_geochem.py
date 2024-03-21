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
import matplotlib # for color handling
import matplotlib.pyplot as plt # for plotting
import os # for file handling

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
