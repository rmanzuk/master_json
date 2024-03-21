# script with sections to assempble the facies vector from PCA of the gridded data, and explore a bit
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
import pandas as pd # for dataframes

#%%
##########################################################################################
# local function imports
##########################################################################################
# %% 
from json_processing import assemble_samples, select_gridded_geochem, select_gridded_im_metrics, select_gridded_point_counts
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


# %% select the gridded geochem data, im metrics, and point counts
geochem_df = select_gridded_geochem(outcrop_data, desired_metrics=['delta13c', 'delta18o', 'Li_Ca', 'Na_Ca', 'Mg_Ca', 'K_Ca', 'Mn_Ca', 'Fe_Ca', 'Sr_Ca'])

im_metric_df =  select_gridded_im_metrics(outcrop_data, desired_metrics=['percentile', 'rayleigh_anisotropy', 'entropy'], desired_scales=[1,0.5, 0.25, 0.125, 0.0625, 0.03125, 0.015625, 0.0078125, 0.00390625, 0.001953125])

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

# %% do PCA to get anisotropy components
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

# and to hold the names of the samples
anisotropy_names = []

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

        # if this is the first time we've seen this sample, get the name
        if row['sample_name'] not in anisotropy_names:
            anisotropy_names.append(row['sample_name'])


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
anisotropy_exp_var = np.zeros((n_bands, n_components))
for band_index in range(n_bands):
    anisotropy_exp_var[band_index,:] = pca_list[band_index].explained_variance_ratio_

# make an array to hold the loadings
anisotropy_loadings = np.zeros((n_bands, n_components, n_scales))
for band_index in range(n_bands):
    anisotropy_loadings[band_index,:,:] = pca_list[band_index].components_


# %% do PCA to get color components
    
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

# and separate ones to hold names 
percentile_names = []

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

        # if this is the first time we've seen this sample, get the name
        if row['sample_name'] not in percentile_names:
            percentile_names.append(row['sample_name'])

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
percentile_exp_var = np.zeros((n_bands, n_components))
for band_index in range(n_bands):
    percentile_exp_var[band_index,:] = pca_list[band_index].explained_variance_ratio_

# make an array to hold the loadings
percentile_loadings = np.zeros((n_bands, n_components, n_components))
for band_index in range(n_bands):
    percentile_loadings[band_index,:,:] = pca_list[band_index].components_

# %% do PCA to get geochem component

# going to be kind of similar to the point count data, just extract data, normalize, and do a PCA
geochem_measurements = geochem_df.columns[5:]
geochem_names = geochem_df.sample_name
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
geochem_names = np.delete(geochem_names, nan_rows)
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

# %% Look at PCAs for point count fractions

# this df is mostly ready to go, just need to extract data, make some adjustments, and do a PCA
#pc_classes = point_count_df.columns[4:] 
pc_names = point_count_df.sample_name.copy()
pc_names = pc_names.to_numpy()

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

# %% now assemple a dataframe with the scores for the components we select

# I'm just going to list out all the collumns we want to keep
facies_vector_fields = ['sample_name', 'field_lithology', 'lat', 'lon', 'msl', 'phase', 'carbon_isotopes', 'geochem_pc1', 'geochem_pc2', 
                        'anisotropy_pc1', 'anisotropy_pc2', 'anisotropy_pc3', 'pc_pc1', 'pc_pc2', 'pc_pc3', 'pc_pc4', 'pc_pc5']
# and systematically add in the first 3 pcs for each of the percentile spectra
band_strs = unique_bands.astype(int).astype(str)
for i in range(3):
    for band in band_strs:
        facies_vector_fields.append(band + '_percentile_pc' + str(i+1))


# now we need to match the samples in the other pcs to the samples in the geochem pcs
anisotropy_inds = []
pc_inds = []
percentile_inds = []
for i in range(len(geochem_names)):
    sample_name = geochem_names[i]
    if sample_name in anisotropy_names:
        anisotropy_inds.append(np.where(np.array(anisotropy_names) == sample_name)[0][0])
    else:
        anisotropy_inds.append(np.nan)
    if sample_name in pc_names:
        pc_inds.append(np.where(pc_names == sample_name)[0][0])
    else:
        pc_inds.append(np.nan)
    if sample_name in percentile_names:
        percentile_inds.append(np.where(np.array(percentile_names) == sample_name)[0][0])  
    else:
        percentile_inds.append(np.nan)

# check the other pcs to see if they have samples that the geochem pcs don't, and add them to the list
for i in range(len(anisotropy_names)):
    sample_name = anisotropy_names[i]
    if sample_name not in geochem_names:
        anisotropy_inds.append(i)
        geochem_names = np.append(geochem_names, [sample_name])
        print(sample_name)
        # now check the others, if they do have the sample, add the index, if not, add a nan
        if sample_name in pc_names:
            pc_inds.append(np.where(pc_names == sample_name)[0][0])
        else:
            pc_inds.append(np.nan)
        if sample_name in percentile_names:
            percentile_inds.append(np.where(np.array(percentile_names) == sample_name)[0][0])
        else:
            percentile_inds.append(np.nan)

for i in range(len(pc_names)):
    sample_name = pc_names[i]
    if sample_name not in geochem_names:
        pc_inds.append(i)
        geochem_names = np.append(geochem_names, [sample_name])
        print(sample_name)
        if sample_name in percentile_names:
            percentile_inds.append(np.where(np.array(percentile_names) == sample_name)[0][0])
        else:
            percentile_inds.append(np.nan)
        if sample_name in anisotropy_names:
            anisotropy_inds.append(np.where(np.array(anisotropy_names) == sample_name)[0][0])
        else:
            anisotropy_inds.append(np.nan)

for i in range(len(percentile_names)):
    sample_name = percentile_names[i]
    if sample_name not in geochem_names:
        percentile_inds.append(i)
        geochem_names = np.append(geochem_names, [sample_name])
        print(sample_name)
        if sample_name in anisotropy_names:
            anisotropy_inds.append(np.where(np.array(anisotropy_names) == sample_name)[0][0])
        else:
            anisotropy_inds.append(np.nan)
        if sample_name in pc_names:
            pc_inds.append(np.where(pc_names == sample_name)[0][0])
        else:
            pc_inds.append(np.nan)

field_liths = []
field_locs = np.zeros((len(geochem_names), 3))

# to get some of the field data, will be easiest to look at the outcrop data, but we need a list of the sample names in there
outcrop_sample_names = []
for sample in outcrop_data['grid_data'][0]['samples']:
    outcrop_sample_names.append(sample['sample_name'])  

# then add the field lithologies, lat, lon, and msl
for i in range(len(geochem_names)):
    outcrop_index = outcrop_sample_names.index(geochem_names[i])
    field_liths.append(outcrop_data['grid_data'][0]['samples'][outcrop_index]['field_lithology'])
    field_locs[i,0] = outcrop_data['grid_data'][0]['samples'][outcrop_index]['latitude']
    field_locs[i,1] = outcrop_data['grid_data'][0]['samples'][outcrop_index]['longitude']
    field_locs[i,2] = outcrop_data['grid_data'][0]['samples'][outcrop_index]['msl']

# then add the phase, which comes right from the geochem data, but need to add nans for the samples we tacked on
missing_phases = len(geochem_names) - len(geochem_phases)
geochem_phases = np.append(geochem_phases, np.full(missing_phases, np.nan))

# same for the carbon isotopes, and geochem pcs
missing_carbon = len(geochem_names) - len(carbon_isotopes)
carbon_isotopes = np.append(carbon_isotopes, np.full(missing_carbon, np.nan))
geochem_scores = np.append(geochem_scores, np.full((missing_carbon, geochem_scores.shape[1]), np.nan), axis=0)

# assemble the anisotropy array from first 3 pcs, looking at sample names for correspondence
anisotropy_scores_full = np.zeros((len(geochem_names), 3))
for i in range(len(geochem_names)):
    if not np.isnan(anisotropy_inds[i]):
        anisotropy_scores_full[i,:] = np.squeeze(anisotropy_scores[int(anisotropy_inds[i]),0:3])

# same for the pc scores
pc_scores_full = np.zeros((len(geochem_names), 5))
for i in range(len(geochem_names)):
    if not np.isnan(pc_inds[i]):
        pc_scores_full[i,:] = pc_scores[int(pc_inds[i]),0:5]

# and the percentile scores
percentile_scores_full = np.zeros((len(geochem_names), 3*len(unique_bands)))
for i in range(len(geochem_names)):
    if not np.isnan(percentile_inds[i]):
        percentile_scores_full[i,:] = np.reshape(percentile_scores[int(percentile_inds[i]),0:3,:], (1,3*len(unique_bands)))

# %%                              
# should be good to go to assemble the dataframe
facies_vector_df = np.column_stack((geochem_names, field_liths, field_locs, geochem_phases, carbon_isotopes, geochem_scores[:,:2], anisotropy_scores_full, pc_scores_full, percentile_scores_full))
facies_vector_df = pd.DataFrame(facies_vector_df, columns=facies_vector_fields)

# %% extract just the data ones and assemble a pairwise correlation matrix
data_fields = facies_vector_fields[6:]

for_correlation = facies_vector_df[data_fields].copy()

correlation_matrix = for_correlation.corr()

plt.imshow(correlation_matrix, cmap='viridis')
# label each pixel tick with the name of the field
plt.xticks(np.arange(len(data_fields)), data_fields, rotation=90)
plt.yticks(np.arange(len(data_fields)), data_fields)
plt.colorbar(label='Correlation Coefficient')
plt.title('Pairwise Correlation Matrix for Facies Vector Components')
plt.show()