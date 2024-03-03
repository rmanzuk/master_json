# script with sections to handle all little data input/updata tasks for jsons
# written by R. A. Manzuk 02/29/2024
# last updated 02/29/2024

##########################################################################################
# package imports
##########################################################################################
# %%
import json # for json handling
import numpy as np # for array handling
from sklearn.decomposition import PCA # for PCA
import matplotlib.pyplot as plt # for plotting
import os # for file handling

#%%
##########################################################################################
# local function imports
##########################################################################################
# %% 
from json_processing import assemble_samples, select_gridded_geochem, select_gridded_im_metrics, select_gridded_point_counts, data_audit# %% 
##########################################################################################
# script lines
##########################################################################################
# %%
# %% define paths, and read in the outcrop json, and assemble samples

outcrop_json_file = '/Users/ryan/Dropbox (Princeton)/code/master_json/stewarts_mill.json'
sample_json_dir = '/Users/ryan/Dropbox (Princeton)/code/master_json/stewarts_mill_grid_samples/'

with open(outcrop_json_file, 'r') as f:
    outcrop_data = json.load(f)

outcrop_data = assemble_samples(outcrop_data, sample_json_dir, data_type=['grid_data'], data_name=["Stewart's Mill Grid"])


# %% select the gridded geochem data, im metrics, and point counts
geochem_df = select_gridded_geochem(outcrop_data, desired_metrics=['delta13c', 'delta18o', 'Li_Ca', 'Na_Ca', 'Mg_Ca', 'K_Ca', 'Mn_Ca', 'Fe_Ca', 'Sr_Ca'])

im_metric_df =  select_gridded_im_metrics(outcrop_data, desired_metrics=['percentile', 'rayleigh_anisotropy', 'entropy'], desired_scales=[1,0.5, 0.25, 0.125, 0.0625, 0.03125, 0.015625, 0.0078125, 0.00390625])

point_count_df = select_gridded_point_counts(outcrop_data)

# and as a useful point of comparison, it may be helpul to have the field_lithology, latitude, longitude, and msl from each sample
all_sample_jsons = [os.path.join(sample_json_dir, x) for x in os.listdir(sample_json_dir) if x.endswith('.json')]
sample_names = []
field_liths = []
sample_lats = []
sample_lons = []
sample_msls = []
for sample_json in all_sample_jsons:
    with open(sample_json, 'r') as f:
        sample_data = json.load(f)
    sample_names.append(sample_data['sample_name'])
    field_liths.append(sample_data['field_lithology'])
    sample_lats.append(sample_data['latitude'])
    sample_lons.append(sample_data['longitude'])
    sample_msls.append(sample_data['msl'])

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

# make a 3D array to hold the data
n_samples = len(unique_samples)
n_percentiles = len(unique_percentiles)
n_bands = len(unique_bands)

percentile_spectra = np.zeros((n_samples, n_percentiles, n_bands))

# iterate through all rows of the im_metric_df
for index, row in im_metric_df.iterrows():
    # first check if this metric is a percentile, otherwise skip
    if 'percentile' in row['metric_name']:
        # get the sample index
        sample_index = np.where(unique_samples == row['sample_name'])[0][0]
        # get the percentile index
        percentile_index = np.where(unique_percentiles == int(row['metric_name'].split('_')[1]))[0][0]
        # get the band index
        band_index = np.where(unique_bands == row['wavelength'])[0][0]
        # put the value in the array
        percentile_spectra[sample_index, percentile_index, band_index] = row['value']

# normalize the spectra matrix to have a mean of 0 and a standard deviation of 1
percentile_spectra = (percentile_spectra - np.mean(percentile_spectra, axis=0))/np.std(percentile_spectra, axis=0)

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
explained_variance = np.zeros((n_bands, n_components))
for band_index in range(n_bands):
    explained_variance[band_index,:] = pca_list[band_index].explained_variance_ratio_

# make an array to hold the loadings
loadings = np.zeros((n_bands, n_components, n_percentiles))

# to associate these pc scores with things like field lithology and location, need to compare the sample names to the unique samples to get the indices
# we need to know where unique_samples is in the sample_names list
field_index = np.zeros(n_samples)
for i in range(n_samples):
    field_index[i] = np.where(np.array(sample_names) == unique_samples[i])[0][0]


# make a cross plot of the first two components for each band (on same figure)
# color code by field lithology
# first give each row in the array a lithology index
lith_index = np.zeros(n_samples)
for i in range(n_samples):
    lith_index[i] = np.where(unique_liths == field_liths[int(field_index[i])])[0][0]

# then make the plot
fig, ax = plt.subplots(3,3, figsize=(15,15))
for band_index in range(n_bands):
    for lith in range(len(unique_liths)):
        ax[int(band_index/3), band_index%3].scatter(percentile_scores[lith_index == lith, 0, band_index], percentile_scores[lith_index == lith, 1, band_index], label=unique_liths[lith])

    ax[int(band_index/3), band_index%3].set_title('Band: '+str(unique_bands[band_index]))
    ax[int(band_index/3), band_index%3].set_xlabel('PC1')   
    ax[int(band_index/3), band_index%3].set_ylabel('PC2')
    
    # make a legend on one of the blank plots
    if band_index == 0:
        ax[int(band_index/3), band_index%3].legend()
plt.show()

# make a bar chart for each band showing the loadings for the first three components
fig, ax = plt.subplots(3,3, figsize=(15,15))
for band_index in range(n_bands):
    for component in range(3):
        ax[int(band_index/3), band_index%3].bar(unique_percentiles[:-1]+(component), pca_list[band_index].components_[component])
    ax[int(band_index/3), band_index%3].set_title('Band: '+str(unique_bands[band_index]))
    ax[int(band_index/3), band_index%3].set_xlabel('Percentile')   
    ax[int(band_index/3), band_index%3].set_ylabel('Loading')

plt.show()

# make another set of cross plots for pc2 and 3
fig, ax = plt.subplots(3,3, figsize=(15,15))
for band_index in range(n_bands):
    for lith in range(len(unique_liths)):
        ax[int(band_index/3), band_index%3].scatter(percentile_scores[lith_index == lith, 1, band_index], percentile_scores[lith_index == lith, 2, band_index], label=unique_liths[lith])

    ax[int(band_index/3), band_index%3].set_title('Band: '+str(unique_bands[band_index]))
    ax[int(band_index/3), band_index%3].set_xlabel('PC2')   
    ax[int(band_index/3), band_index%3].set_ylabel('PC3')
    if band_index == 0:
        ax[int(band_index/3), band_index%3].legend()
plt.show()

# plot the samples geographically, color coded by pc1, 2, and 3
# first associeate the pc scores with the sample lat and lon
lat_inds = np.zeros(n_samples)
lon_inds = np.zeros(n_samples)
for i in range(n_samples):
    lat_inds[i] = sample_lats[int(field_index[i])]
    lon_inds[i] = sample_lons[int(field_index[i])]

# then make the plot
fig, ax = plt.subplots(1,3, figsize=(15,5))
for i in range(3):
    ax[i].scatter(lon_inds, lat_inds, c=percentile_scores[:,i,0])
    ax[i].set_xlabel('Longitude')
    ax[i].set_ylabel('Latitude')
    ax[i].set_title('PC'+str(i+1))
    # add a colorbar
    cbar = plt.colorbar(ax[i].scatter(lon_inds, lat_inds, c=percentile_scores[:,i,0]), ax=ax[i])
plt.show()
 
        
# %% Look at PCAs for entropy scale spectra

# first get all the unique scales that entropy was calculated at
# get the index of all rows where the metric name is entropy
all_metrics = im_metric_df.metric_name
entropy_inds = [x for x in range(len(all_metrics)) if 'entropy' in all_metrics[x]]

unique_scales = np.unique([im_metric_df.scale[x] for x in entropy_inds])

# and the unique bands it was calculated at
unique_bands = np.unique([im_metric_df.wavelength[x] for x in entropy_inds])

# make an array to hold the entropy spectra
n_samples = len(unique_samples)
n_scales = len(unique_scales)
n_bands = len(unique_bands)

entropy_spectra = np.zeros((n_samples, n_scales, n_bands))

# iterate through all rows of the im_metric_df
for index, row in im_metric_df.iterrows():
    # first check if this metric is entropy, otherwise skip
    if 'entropy' in row['metric_name']:
        # get the sample index
        sample_index = np.where(unique_samples == row['sample_name'])[0][0]
        # get the scale index
        scale_index = np.where(unique_scales == row['scale'])[0][0]
        # get the band index
        band_index = np.where(unique_bands == row['wavelength'])[0][0]
        # put the value in the array
        entropy_spectra[sample_index, scale_index, band_index] = row['value']

# normalize the spectra matrix to have a mean of 0 and a standard deviation of 1
entropy_spectra = (entropy_spectra - np.mean(entropy_spectra, axis=0))/np.std(entropy_spectra, axis=0)

# now we can do a PCA on each band
n_components = entropy_spectra.shape[1]
pca_list = []
for band_index in range(n_bands):
    pca = PCA(n_components=n_components)
    pca.fit(entropy_spectra[:,:,band_index])
    pca_list.append(pca)

# and make a new 3d array to hold the PCA reprjections
entropy_scores = np.zeros((n_samples, n_components, n_bands))
for band_index in range(n_bands):
    entropy_scores[:,:,band_index] = pca_list[band_index].transform(entropy_spectra[:,:,band_index])

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
    field_index[i] = np.where(np.array(sample_names) == unique_samples[i])[0][0]
    
# make a cross plot of the first two components for each band (on same figure)
# color code by field lithology
# first give each row in the array a lithology index
lith_index = np.zeros(n_samples)
for i in range(n_samples):
    lith_index[i] = np.where(unique_liths == field_liths[int(field_index[i])])[0][0]

# then make the plot
fig, ax = plt.subplots(3,3, figsize=(15,15))
for band_index in range(n_bands):
    for lith in range(len(unique_liths)):
        ax[int(band_index/3), band_index%3].scatter(entropy_scores[lith_index == lith, 0, band_index], entropy_scores[lith_index == lith, 1, band_index], label=unique_liths[lith])

    ax[int(band_index/3), band_index%3].set_title('Band: '+str(unique_bands[band_index]))
    ax[int(band_index/3), band_index%3].set_xlabel('PC1')   
    ax[int(band_index/3), band_index%3].set_ylabel('PC2')
    if band_index == 0:
        ax[int(band_index/3), band_index%3].legend()
plt.show()

# make a bar chart for each band showing the loadings for the first three components
fig, ax = plt.subplots(3,3, figsize=(15,15))
for band_index in range(n_bands):
    for component in range(3):
        ax[int(band_index/3), band_index%3].bar(unique_scales+(component*0.01), loadings[band_index,component,:], width = 0.01)
    ax[int(band_index/3), band_index%3].set_title('Band: '+str(unique_bands[band_index]))
    ax[int(band_index/3), band_index%3].set_xlabel('Scale')   
    ax[int(band_index/3), band_index%3].set_ylabel('Loading')
plt.show()

# make another set of cross plots for pc2 and 3
fig, ax = plt.subplots(3,3, figsize=(15,15))
for band_index in range(n_bands):
    for lith in range(len(unique_liths)):
        ax[int(band_index/3), band_index%3].scatter(entropy_scores[lith_index == lith, 1, band_index], entropy_scores[lith_index == lith, 2, band_index], label=unique_liths[lith])

    ax[int(band_index/3), band_index%3].set_title('Band: '+str(unique_bands[band_index]))
    ax[int(band_index/3), band_index%3].set_xlabel('PC2')   
    ax[int(band_index/3), band_index%3].set_ylabel('PC3')
    if band_index == 0:
        ax[int(band_index/3), band_index%3].legend()
plt.show()

# plot the samples geographically, color coded by pc1, 2, and 3
# first associeate the pc scores with the sample lat and lon
lat_inds = np.zeros(n_samples)
lon_inds = np.zeros(n_samples)
for i in range(n_samples):
    lat_inds[i] = sample_lats[int(field_index[i])]
    lon_inds[i] = sample_lons[int(field_index[i])]
    
# then make the plot
fig, ax = plt.subplots(1,3, figsize=(15,5))
for i in range(3):
    ax[i].scatter(lon_inds, lat_inds, c=entropy_scores[:,i,0])
    ax[i].set_xlabel('Longitude')
    ax[i].set_ylabel('Latitude')
    ax[i].set_title('PC'+str(i+1))
    # add a colorbar
    cbar = plt.colorbar(ax[i].scatter(lon_inds, lat_inds, c=entropy_scores[:,i,0]), ax=ax[i])
plt.show()


# %% Look at PCAs for anisotropy scale spectra

# first get all the unique scales that entropy was calculated at
# get the index of all rows where the metric name is entropy
all_metrics = im_metric_df.metric_name
anisotropy_inds = [x for x in range(len(all_metrics)) if 'rayleigh_anisotropy' in all_metrics[x]]

unique_scales = np.unique([im_metric_df.scale[x] for x in anisotropy_inds])

# and the unique bands it was calculated at
unique_bands = np.unique([im_metric_df.wavelength[x] for x in anisotropy_inds])

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
    field_index[i] = np.where(np.array(sample_names) == unique_samples[i])[0][0]

# make a cross plot of the first two components for each band (on same figure)
# color code by field lithology
# first give each row in the array a lithology index
lith_index = np.zeros(n_samples)
for i in range(n_samples):
    lith_index[i] = np.where(unique_liths == field_liths[int(field_index[i])])[0][0]

# then make the plot
fig, ax = plt.subplots(3,3, figsize=(15,15))
for band_index in range(n_bands):
    for lith in range(len(unique_liths)):
        ax[int(band_index/3), band_index%3].scatter(anisotropy_scores[lith_index == lith, 0, band_index], anisotropy_scores[lith_index == lith, 1, band_index], label=unique_liths[lith])

    ax[int(band_index/3), band_index%3].set_title('Band: '+str(unique_bands[band_index]))
    ax[int(band_index/3), band_index%3].set_xlabel('PC1')   
    ax[int(band_index/3), band_index%3].set_ylabel('PC2')
    if band_index == 0:
        ax[int(band_index/3), band_index%3].legend()
plt.show()

# make a bar chart for each band showing the loadings for the first three components
fig, ax = plt.subplots(3,3, figsize=(15,15))
for band_index in range(n_bands):
    for component in range(3):
        ax[int(band_index/3), band_index%3].bar(unique_scales+(component*0.01), loadings[band_index,component,:], width = 0.01)
    ax[int(band_index/3), band_index%3].set_title('Band: '+str(unique_bands[band_index]))
    ax[int(band_index/3), band_index%3].set_xlabel('Scale')   
    ax[int(band_index/3), band_index%3].set_ylabel('Loading')
plt.show()

# make another set of cross plots for pc2 and 3
fig, ax = plt.subplots(3,3, figsize=(15,15))

for band_index in range(n_bands):
    for lith in range(len(unique_liths)):
        ax[int(band_index/3), band_index%3].scatter(anisotropy_scores[lith_index == lith, 1, band_index], anisotropy_scores[lith_index == lith, 2, band_index], label=unique_liths[lith])

    ax[int(band_index/3), band_index%3].set_title('Band: '+str(unique_bands[band_index]))
    ax[int(band_index/3), band_index%3].set_xlabel('PC2')   
    ax[int(band_index/3), band_index%3].set_ylabel('PC3')
    if band_index == 0:
        ax[int(band_index/3), band_index%3].legend()
plt.show()

# plot the samples geographically, color coded by pc1, 2, and 3
# first associeate the pc scores with the sample lat and lon
lat_inds = np.zeros(n_samples)
lon_inds = np.zeros(n_samples)
for i in range(n_samples):
    lat_inds[i] = sample_lats[int(field_index[i])]
    lon_inds[i] = sample_lons[int(field_index[i])]

# then make the plot
fig, ax = plt.subplots(1,3, figsize=(15,5))
for i in range(3):
    ax[i].scatter(lon_inds, lat_inds, c=anisotropy_scores[:,i,0])
    ax[i].set_xlabel('Longitude')
    ax[i].set_ylabel('Latitude')
    ax[i].set_title('PC'+str(i+1))
    # add a colorbar
    cbar = plt.colorbar(ax[i].scatter(lon_inds, lat_inds, c=anisotropy_scores[:,i,0]), ax=ax[i])
plt.show()


# %% Look at PCAs for point count fractions

# this df is mostly ready to go, just need to extract data, make some adjustments, and do a PCA
pc_classes = point_count_df.columns[4:] 
pc_samples = point_count_df.sample_name

# extract the data into an array
pc_data = point_count_df[pc_classes].to_numpy()

# we don't want to the collumn thats None, so we'll remove it, redo the fractions so each row sums to 1, and then do a PCA
none_inds = []
for i in range(len(pc_classes)):
    if pc_classes[i] is None:
        none_inds.append(i)

pc_classes = np.delete(pc_classes, none_inds[0])
pc_data = np.delete(pc_data, none_inds[0], axis=1)

# before redoing fractions, replace nans with zeros
pc_data = np.nan_to_num(pc_data)

# make the rows sum to 1 again
pc_data = pc_data/np.sum(pc_data, axis=1)[:,None]

# now normalize and do a PCA
pc_data = (pc_data - np.mean(pc_data, axis=0))/np.std(pc_data, axis=0)

pca = PCA(n_components=len(pc_classes))
pca.fit(pc_data)
pc_scores = pca.transform(pc_data)
pc_loadings = pca.components_
explained_variance = pca.explained_variance_ratio_

# associate these samples with lithology and location
field_index = np.zeros(len(pc_samples))
for i in range(len(pc_samples)):
    field_index[i] = np.where(np.array(sample_names) == pc_samples[i])[0][0]

lith_index = np.zeros(len(pc_samples))
for i in range(len(pc_samples)):
    lith_index[i] = np.where(unique_liths == field_liths[int(field_index[i])])[0][0]

# make a cross plot of the first two components, color coded by lithology
fig, ax = plt.subplots(1,1, figsize=(5,5))
for lith in range(len(unique_liths)):
    ax.scatter(pc_scores[lith_index == lith, 0], pc_scores[lith_index == lith, 1], label=unique_liths[lith])

ax.set_xlabel('PC1')
ax.set_ylabel('PC2')
ax.legend()
plt.show()

# make a bar chart for each component showing the loadings for first three components
fig, ax = plt.subplots(1,1, figsize=(5,5))
for component in range(3):
    ax.bar(np.linspace(0,len(pc_classes),len(pc_classes))+(component*0.1), pc_loadings[component], width = 0.1) 
ax.set_xlabel('Class')
# set the x ticks to be the string names of the classes
ax.set_xticks(np.linspace(0,len(pc_classes),len(pc_classes)))
ax.set_xticklabels(pc_classes)
ax.set_ylabel('Loading')
plt.show()

# plot the samples geographically, color coded by pc1, 2, and 3
# first associeate the pc scores with the sample lat and lon
lat_inds = np.zeros(len(pc_samples))
lon_inds = np.zeros(len(pc_samples))
for i in range(len(pc_samples)):
    lat_inds[i] = sample_lats[int(field_index[i])]
    lon_inds[i] = sample_lons[int(field_index[i])]

# then make the plot
fig, ax = plt.subplots(1,3, figsize=(5,5))
for i in range(3):
    ax[i].scatter(lon_inds, lat_inds, c=pc_scores[:,i])
    ax[i].set_xlabel('Longitude')
    ax[i].set_ylabel('Latitude')
    ax[i].set_title('PC'+str(i+1))
    # add a colorbar
    cbar = plt.colorbar(ax[i].scatter(lon_inds, lat_inds, c=pc_scores[:,i]), ax=ax[i])
plt.show()


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

# now normalize and do a PCA
geochem_data = (geochem_data - np.mean(geochem_data, axis=0))/np.std(geochem_data, axis=0)

pca = PCA(n_components=len(geochem_measurements))
pca.fit(geochem_data)
geochem_scores = pca.transform(geochem_data)
geochem_loadings = pca.components_
explained_variance = pca.explained_variance_ratio_

# make a crossplot of the first two components, color coded by phase
fig, ax = plt.subplots(1,1, figsize=(5,5))
for phase in np.unique(geochem_phases):
    ax.scatter(geochem_scores[geochem_phases == phase, 0], geochem_scores[geochem_phases == phase, 1], label=phase)

ax.set_xlabel('PC1')
ax.set_ylabel('PC2')
ax.legend()
plt.show()

# make a bar chart for each component showing the loadings for first three components
fig, ax = plt.subplots(1,1, figsize=(5,5))
for component in range(3):
    ax.bar(np.linspace(0,len(geochem_measurements),len(geochem_measurements))+(component*0.1), geochem_loadings[component], width = 0.1)
ax.set_xlabel('Measurement')
# set the x ticks to be the string names of the classes
ax.set_xticks(np.linspace(0,len(geochem_measurements),len(geochem_measurements)))
ax.set_xticklabels(geochem_measurements)
ax.set_ylabel('Loading')
plt.show()

# plot pcs 2 and 3  
fig, ax = plt.subplots(1,1, figsize=(5,5))
for phase in np.unique(geochem_phases):
    ax.scatter(geochem_scores[geochem_phases == phase, 1], geochem_scores[geochem_phases == phase, 2], label=phase)

ax.set_xlabel('PC2')
ax.set_ylabel('PC3')
ax.legend()
plt.show()

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

