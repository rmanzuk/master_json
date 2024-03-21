# script with sections to look at PCA of point count data
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
