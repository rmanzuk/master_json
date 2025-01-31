# script with sections to investigate trends in carbon isotopes
# written by R. A. Manzuk 07/08/2024
# last updated 07/08/2024

##########################################################################################
# package imports
##########################################################################################
# %%
import json # for json handling
import numpy as np # for array handling
from sklearn.decomposition import PCA # for PCA
import matplotlib # for color handling
import matplotlib.pyplot as plt # for plotting
import pandas as pd # for data handling
import os # for file handling   
from scipy.cluster.hierarchy import dendrogram, linkage # for displaying clusters
from sklearn.metrics import r2_score # for calculating r2

#%%
##########################################################################################
# local function imports
##########################################################################################
# %% 
from json_processing import assemble_samples, select_gridded_im_metrics, select_gridded_point_counts, select_gridded_pa
from geospatial import latlong_to_utm, dip_correct_elev

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


# %% select the gridded im metrics, and point counts

im_metric_df =  select_gridded_im_metrics(outcrop_data, desired_metrics=['percentile', 'rayleigh_anisotropy', 'entropy', 'glcm_contrast'], desired_scales=[1,0.5, 0.25, 0.125, 0.0625, 0.03125, 0.015625, 0.0078125, 0.00390625, 0.001953125])

point_count_df = select_gridded_point_counts(outcrop_data)

presence_absence_df = select_gridded_pa(outcrop_data)

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
all_metrics_anisotropy = im_metric_df.metric_name
anisotropy_inds = [x for x in range(len(all_metrics_anisotropy)) if 'rayleigh_anisotropy' in all_metrics_anisotropy[x]]

unique_scales_anisotropy = np.unique([im_metric_df.scale[x] for x in anisotropy_inds])

# and the unique bands it was calculated at
unique_bands_anisotropy = np.unique([im_metric_df.wavelength[x] for x in anisotropy_inds])

unique_samples = im_metric_df.sample_name.unique()

# make an array to hold the entropy spectra
n_samples = len(unique_samples)
n_scales = len(unique_scales_anisotropy)
n_bands = len(unique_bands_anisotropy)

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
        scale_index = np.where(unique_scales_anisotropy == row['scale'])[0][0]
        # get the band index
        band_index = np.where(unique_bands_anisotropy == row['wavelength'])[0][0]
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

# %% do PCA to get glcm contrast components

# first get all the unique scales that entropy was calculated at
# get the index of all rows where the metric name is entropy
all_metrics_contrast = im_metric_df.metric_name
contrast_inds = [x for x in range(len(all_metrics_contrast)) if 'glcm_contrast' in all_metrics_contrast[x]]

unique_scales_contrast = np.unique([im_metric_df.scale[x] for x in contrast_inds])

# and the unique bands it was calculated at
unique_bands_contrast = np.unique([im_metric_df.wavelength[x] for x in contrast_inds])

unique_samples = im_metric_df.sample_name.unique()

# make an array to hold the entropy spectra
n_samples = len(unique_samples)
n_scales = len(unique_scales_contrast)
n_bands = len(unique_bands_contrast)

contrast_spectra = np.zeros((n_samples, n_scales, n_bands))

# and to hold the names of the samples
contrast_names = []

# iterate through all rows of the im_metric_df
for index, row in im_metric_df.iterrows():
    # first check if this metric is entropy, otherwise skip
    if 'glcm_contrast' in row['metric_name']:
        # get the sample index
        sample_index = np.where(unique_samples == row['sample_name'])[0][0]
        # get the scale index
        scale_index = np.where(unique_scales_contrast == row['scale'])[0][0]
        # get the band index
        band_index = np.where(unique_bands_contrast == row['wavelength'])[0][0]
        # put the value in the array
        contrast_spectra[sample_index, scale_index, band_index] = row['value']
        
        # if this is the first time we've seen this sample, get the name
        if row['sample_name'] not in contrast_names:
            contrast_names.append(row['sample_name'])

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

# %% do PCA to get color components
    
# first need to extract the color percentile spectra from the im_metric_df
# we'll make a 3D array that is n_samples x n_percentiles x n_bands, which we can do a PCA on each band

# get a list of the unique samples
unique_samples = im_metric_df.sample_name.unique()

# look in the metrics to find the unique percentiles, which are listed as 'percentile_XX'
all_metrics_percentile = im_metric_df.metric_name
percentile_metrics = all_metrics_percentile[all_metrics_percentile.str.contains('percentile')]
unique_percentiles = percentile_metrics.unique()

# and extract just the number and sort them
unique_percentiles = np.sort([int(x.split('_')[1]) for x in unique_percentiles])

# get the unique bands
unique_bands_percentile = im_metric_df.wavelength.unique()

# and flag if we want the normalized or unnormalized spectra
normalized = False

# make a 3D array to hold the dataf
n_samples = len(unique_samples)
n_percentiles = len(unique_percentiles)
n_bands = len(unique_bands_percentile)

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
        band_index = np.where(unique_bands_percentile == row['wavelength'])[0][0]
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



# %% Look at PCAs for point count fractions

# this df is mostly ready to go, just need to extract data, make some adjustments, and do a PCA
#pc_classes = point_count_df.columns[4:] 
pc_names = point_count_df.sample_name.copy()
pc_names = pc_names.to_numpy()

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

# %% make the facies vector for 'whole rock' metrics

whole_rock_fields = ['sample_name', 'field_lithology', 'lat', 'lon', 'msl', 'contrast_pc1', 'contrast_pc2', 'contrast_pc3', 'anisotropy_pc1', 'anisotropy_pc2', 'anisotropy_pc3', ]

# and systematically add in the first 3 pcs for each of the percentile spectra
band_strs = unique_bands_percentile.astype(int).astype(str)
for i in range(3):
    for band in band_strs:
        whole_rock_fields.append(band + '_percentile_pc' + str(i+1))

# and add in the original point count classes
for pc_class in pc_classes:
    whole_rock_fields.append(pc_class)

# add in the presence or absence classes
for pa_class in presence_absence_df.columns[4:]:
    whole_rock_fields.append(pa_class)

# we'll use anisotropy as the standard, and give the other pcs inds to match
contrast_inds = []
pc_inds = []
pa_inds = []
percentile_inds = []
for i in range(len(anisotropy_names)):
    sample_name = anisotropy_names[i]
    if sample_name in contrast_names:
        contrast_inds.append(np.where(np.array(contrast_names) == sample_name)[0][0])
    else:
        contrast_inds.append(np.nan)
    if sample_name in pc_names:
        pc_inds.append(np.where(pc_names == sample_name)[0][0])
    else:
        pc_inds.append(np.nan)
    if sample_name in presence_absence_df.sample_name.to_numpy():
        pa_inds.append(np.where(presence_absence_df.sample_name == sample_name)[0][0])
    else:
        pa_inds.append(np.nan)
    if sample_name in percentile_names:
        percentile_inds.append(np.where(np.array(percentile_names) == sample_name)[0][0])  
    else:
        percentile_inds.append(np.nan)

# check the other samples to see if they have samples that the anisotropy samples don't, and add them to the list (but not for the presence absence)
for i in range(len(pc_names)):
    sample_name = pc_names[i]
    if sample_name not in anisotropy_names:
        pc_inds.append(i)
        anisotropy_names = np.append(anisotropy_names, [sample_name])
        if sample_name in percentile_names:
            percentile_inds.append(np.where(np.array(percentile_names) == sample_name)[0][0])
        else:
            percentile_inds.append(np.nan)
    if sample_name in presence_absence_df.sample_name:
        pa_inds.append(np.where(presence_absence_df.sample_name == sample_name)[0][0])
    else:
        pa_inds.append(np.nan)
    if sample_name in contrast_names:
        contrast_inds.append(np.where(np.array(contrast_names) == sample_name)[0][0])
    else:
        contrast_inds.append(np.nan)

for i in range(len(presence_absence_df.sample_name)):
    sample_name = presence_absence_df.sample_name[i]
    if sample_name not in anisotropy_names:
        pa_inds.append(i)
        anisotropy_names = np.append(anisotropy_names, [sample_name])
        if sample_name in pc_names:
            pc_inds.append(np.where(pc_names == sample_name)[0][0])
        else:
            pc_inds.append(np.nan)
        if sample_name in percentile_names:
            percentile_inds.append(np.where(np.array(percentile_names) == sample_name)[0][0])
        else:
            percentile_inds.append(np.nan)
    if sample_name in contrast_names:
        contrast_inds.append(np.where(np.array(contrast_names) == sample_name)[0][0])
    else:
        contrast_inds.append(np.nan)

for i in range(len(percentile_names)):
    sample_name = percentile_names[i]
    if sample_name not in anisotropy_names:
        percentile_inds.append(i)
        anisotropy_names = np.append(anisotropy_names, [sample_name])
        if sample_name in pc_names:
            pc_inds.append(np.where(pc_names == sample_name)[0][0])
        else:
            pc_inds.append(np.nan)
        if sample_name in presence_absence_df.sample_name:
            pa_inds.append(np.where(presence_absence_df.sample_name == sample_name)[0][0])
        else:
            pa_inds.append(np.nan)
        if sample_name in contrast_names:
            contrast_inds.append(np.where(np.array(contrast_names) == sample_name)[0][0])
        else:
            contrast_inds.append(np.nan)

for i in range(len(contrast_names)):
    sample_name = contrast_names[i]
    if sample_name not in anisotropy_names:
        contrast_inds.append(i)
        anisotropy_names = np.append(anisotropy_names, [sample_name])
        if sample_name in pc_names:
            pc_inds.append(np.where(pc_names == sample_name)[0][0])
        else:
            pc_inds.append(np.nan)
        if sample_name in presence_absence_df.sample_name:
            pa_inds.append(np.where(presence_absence_df.sample_name == sample_name)[0][0])
        else:
            pa_inds.append(np.nan)
        if sample_name in percentile_names:
            percentile_inds.append(np.where(np.array(percentile_names) == sample_name)[0][0])
        else:
            percentile_inds.append(np.nan)

# bring in field lithologies and locations for the vector
field_liths = []
field_locs = np.zeros((len(anisotropy_names), 3))

# to get some of the field data, will be easiest to look at the outcrop data, but we need a list of the sample names in there
outcrop_sample_names = []
for sample in outcrop_data['grid_data'][0]['samples']:
    outcrop_sample_names.append(sample['sample_name'])  

# then add the field lithologies, lat, lon, and msl
for i in range(len(anisotropy_names)):
    outcrop_index = outcrop_sample_names.index(anisotropy_names[i])
    field_liths.append(outcrop_data['grid_data'][0]['samples'][outcrop_index]['field_lithology'])
    field_locs[i,0] = outcrop_data['grid_data'][0]['samples'][outcrop_index]['latitude']
    field_locs[i,1] = outcrop_data['grid_data'][0]['samples'][outcrop_index]['longitude']
    field_locs[i,2] = outcrop_data['grid_data'][0]['samples'][outcrop_index]['msl']

# assemble the anisotropy array from the first 3 pcs, which is just the anisotropy scores
anisotropy_scores_full = anisotropy_scores[:,:3,:]
# and squeeze it to get rid of the extra dimension
anisotropy_scores_full = np.squeeze(anisotropy_scores_full)
# check if it is too short because we added samples from other metrics
if anisotropy_scores_full.shape[0] < len(anisotropy_names):
    missing_rows = len(anisotropy_names) - anisotropy_scores_full.shape[0]
    anisotropy_scores_full = np.append(anisotropy_scores_full, np.full((missing_rows, 3), np.nan), axis=0)

# assemble the contrast array from the first 3 pcs
contrast_scores_full = contrast_scores[:,:3,:]
# and squeeze it to get rid of the extra dimension
contrast_scores_full = np.squeeze(contrast_scores_full)
# check if it is too short because we added samples from other metrics
if contrast_scores_full.shape[0] < len(anisotropy_names):
    missing_rows = len(anisotropy_names) - contrast_scores_full.shape[0]
    contrast_scores_full = np.append(contrast_scores_full, np.full((missing_rows, 3), np.nan), axis=0)

# assemble the pc array from the first 5 pcs
pc_scores_full = np.zeros((len(anisotropy_names), len(pc_classes)))
for i in range(len(anisotropy_names)):
    if not np.isnan(pc_inds[i]):
        pc_scores_full[i,:] = pc_data_original[int(pc_inds[i]),:]
    else:
        pc_scores_full[i,:] = np.nan

# assemble the presence absence array
pa_scores_full = np.zeros((len(anisotropy_names), presence_absence_df.shape[1] - 4))
for i in range(len(anisotropy_names)):
    if not np.isnan(pa_inds[i]):
        pa_scores_full[i,:] = np.squeeze(presence_absence_df.iloc[int(pa_inds[i]),4:])

# assemble the percentile array
percentile_scores_full = np.zeros((len(anisotropy_names), 3*len(unique_bands_percentile)))
for i in range(len(anisotropy_names)):
    if not np.isnan(percentile_inds[i]):
        percentile_scores_full[i,:] = np.reshape(percentile_scores[int(percentile_inds[i]),0:3,:], (1,3*len(unique_bands_percentile)))

# should be good to go to assemble the dataframe
whole_rock_vector = np.column_stack((anisotropy_names, field_liths, field_locs, contrast_scores_full, anisotropy_scores_full, percentile_scores_full, pc_scores_full, pa_scores_full))
whole_rock_vector = pd.DataFrame(whole_rock_vector, columns=whole_rock_fields)

# combining the dataframe turned the floats back into strings, so convert them back
data_cols = whole_rock_vector.columns[2:]
whole_rock_vector[data_cols] = whole_rock_vector[data_cols].astype(float)



# %% last thing before looking is to remove the samples from the other mound from all three vecotors

# I'll just list their sample names for now
to_remove = ['smg_167', 'smg_168', 'smg_169', 'smg_170', 'smg_171', 'smg_172', 'smg_173']

# and remove them
whole_rock_vector = whole_rock_vector[~whole_rock_vector.sample_name.isin(to_remove)]

# correct the row index
whole_rock_vector.index = range(len(whole_rock_vector))


# %% dip-correct the coordinates so we have relative stratigraphic elevation

# first need to correct lats and longs to utm
x_whole, y_whole = latlong_to_utm(whole_rock_vector.lat, whole_rock_vector.lon, zone=11, hemisphere='north')

# go into the outcrop data and get the regional strike and dip
regional_strike = outcrop_data['reg_strike_dip'][0]
regional_dip = outcrop_data['reg_strike_dip'][1]

# correct the strike to be a dip direction
regional_dip_dir = regional_strike + 90
if regional_dip_dir > 360:
    regional_dip_dir = regional_dip_dir - 360

# for some reason the whole rock vector has the msl as a string, so convert it to a float
whole_rock_vector.msl = whole_rock_vector.msl.astype(float)

# and now dip correct the elevations
x_corrected_whole, y_corrected_whole, z_corrected_whole = dip_correct_elev(x_whole, y_whole, whole_rock_vector.msl, regional_dip, regional_dip_dir)

# add the z corrected to each df as a new column called 'strat_height'
whole_rock_vector['strat_height'] = z_corrected_whole


# %% cluster based upon some metrics of the whole rock vector and compare to fossil occurrences

# make a numpy array of the data we want to cluster on
metrics_to_cluster = ['contrast_pc2', 'anisotropy_pc2', 'Microb', 'Dol', 'Arch', 'ooid', 'anisotropy_pc3', 'contrast_pc3', '365_percentile_pc1', '590_percentile_pc1']
cluster_data = whole_rock_vector[metrics_to_cluster].to_numpy()

# remove any rows with nans
nan_inds = np.unique(np.where(np.isnan(cluster_data))[0])
cluster_data = np.delete(cluster_data, nan_inds, axis=0)

# also get a set of sample names with the nans removed
sample_names = whole_rock_vector.sample_name.to_numpy()
sample_names = np.delete(sample_names, nan_inds)

# before clustering, we need to normalize the data
cluster_data = (cluster_data - np.mean(cluster_data, axis=0))/np.std(cluster_data, axis=0)

# now we can cluster
Z = linkage(cluster_data, 'ward')

# and plot the dendrogram
plt.figure(figsize=(25, 10))
plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('sample index')
plt.ylabel('distance')
R = dendrogram(Z, leaf_rotation=90., leaf_font_size=8., labels=sample_names)

# and make a bar plot of fossil occurrences for each sample in the order of the dendrogram
# first need to get the fossil occurrences
fossil_groups = ['trilobite','brachiopid','salterella','echinoderm','other_fossil']
fossil_occurrences = whole_rock_vector[fossil_groups].to_numpy()
fossil_occurrences = np.delete(fossil_occurrences, nan_inds, axis=0)

# and plot a stacked bar plot 
plt.figure()
plt.bar(range(len(sample_names)), fossil_occurrences[R['leaves'],0], label='trilobite')
plt.bar(range(len(sample_names)), fossil_occurrences[R['leaves'],1], bottom=fossil_occurrences[R['leaves'],0], label='brachiopid')
plt.bar(range(len(sample_names)), fossil_occurrences[R['leaves'],2], bottom=fossil_occurrences[R['leaves'],0] + fossil_occurrences[R['leaves'],1], label='salterella')
plt.bar(range(len(sample_names)), fossil_occurrences[R['leaves'],3], bottom=fossil_occurrences[R['leaves'],0] + fossil_occurrences[R['leaves'],1] + fossil_occurrences[R['leaves'],2], label='echinoderm')
plt.bar(range(len(sample_names)), fossil_occurrences[R['leaves'],4], bottom=fossil_occurrences[R['leaves'],0] + fossil_occurrences[R['leaves'],1] + fossil_occurrences[R['leaves'],2] + fossil_occurrences[R['leaves'],3], label='other_fossil')
plt.legend()
plt.show()

# %% Use groups I've identified to add a different heirarchy

group_sizes = [5, 7, 9, 4, 4, 3, 5, 7, 5, 2, 7, 5, 1, 4, 6, 2, 8, 4, 3, 4, 5, 3, 6, 4, 5, 3, 5, 4, 7, 4, 6, 6, 5, 7]
group_ranges = np.cumsum(group_sizes)

# make a stacked bar graph of the proportion of samples in each group that have each fossil
grouped_fossil_occurrences = np.zeros((len(group_sizes), fossil_occurrences.shape[1]))
for i in range(len(group_sizes)):
    these_inds = R['leaves'][group_ranges[i]-group_sizes[i]:group_ranges[i]]
    grouped_fossil_occurrences[i,:] = np.sum(fossil_occurrences[these_inds,:], axis=0)

grouped_fossil_proportions = np.zeros((len(group_sizes), fossil_occurrences.shape[1]))
for i in range(fossil_occurrences.shape[1]):
    grouped_fossil_proportions[:,i] = grouped_fossil_occurrences[:,i]/group_sizes

plt.figure()
plt.bar(range(len(group_sizes)), grouped_fossil_proportions[:,0], label='trilobite')
plt.bar(range(len(group_sizes)), grouped_fossil_proportions[:,1], bottom=grouped_fossil_proportions[:,0], label='brachiopid')
plt.bar(range(len(group_sizes)), grouped_fossil_proportions[:,2], bottom=grouped_fossil_proportions[:,0] + grouped_fossil_proportions[:,1], label='salterella')
plt.bar(range(len(group_sizes)), grouped_fossil_proportions[:,3], bottom=grouped_fossil_proportions[:,0] + grouped_fossil_proportions[:,1] + grouped_fossil_proportions[:,2], label='echinoderm')
plt.bar(range(len(group_sizes)), grouped_fossil_proportions[:,4], bottom=grouped_fossil_proportions[:,0] + grouped_fossil_proportions[:,1] + grouped_fossil_proportions[:,2] + grouped_fossil_proportions[:,3], label='other_fossil')
plt.legend()
plt.show()

# %% look in stratigraphic windows and plot the fossil occurrences

# get the strat heights with nan rows removed
strat_heights = whole_rock_vector.strat_height.to_numpy()
strat_heights = np.delete(strat_heights, nan_inds)

# and get moving window sums of the fossil occurrences
window_size = 10

# bin the strat heights by the window size
strat_bins = np.floor(strat_heights/window_size)
unique_bins = np.unique(strat_bins)

# make an array to hold the windowed fossil occurrences
windowed_fossils = np.zeros((len(unique_bins), fossil_occurrences.shape[1]))

for i in range(len(unique_bins)):
    window_inds = np.where(strat_bins == unique_bins[i])[0]
    # make the windowed fossils be the proportion of the samples in the window that have the fossil
    windowed_fossils[i,:] = np.sum(fossil_occurrences[window_inds,:], axis=0)/len(window_inds)


# and plot them
plt.figure()
plt.plot(windowed_fossils[:,0], unique_bins*window_size, label='trilobite', marker='o')
plt.plot(windowed_fossils[:,1], unique_bins*window_size, label='brachiopid', marker='o')
plt.plot(windowed_fossils[:,2], unique_bins*window_size, label='salterella', marker='o')
plt.plot(windowed_fossils[:,3], unique_bins*window_size, label='echinoderm', marker='o')
plt.plot(windowed_fossils[:,4], unique_bins*window_size, label='other_fossil', marker='o')
plt.legend()
plt.show()

# %% in the stratigraphic windows, look at the occurrences of the grouped facies

windowed_groups = np.zeros((len(unique_bins), len(group_sizes)))

# make a group index vector of same length as the number of samples
group_index = np.zeros(sum(group_sizes))
for i in range(len(group_sizes)):
    group_index[group_ranges[i]-group_sizes[i]:group_ranges[i]] = i

for i in range(len(unique_bins)):
    window_inds = np.where(strat_bins == unique_bins[i])[0]
    
    # now get the grouped facies that correspond to these samples
    for j in range(len(group_sizes)):
        # add up the ones that are in both this group and strat window
        group_leaves = np.where(group_index == j)[0]
        in_group = np.array(R['leaves'])[group_leaves]
        in_both = np.intersect1d(in_group, window_inds)
        windowed_groups[i,j] = len(in_both)

windowed_group_proportions = windowed_groups/np.sum(windowed_groups, axis=1)[:,None]

# multiply the proportion of each group by the proportion of the group that has each fossil
fossils_pred_facies = np.zeros((len(unique_bins), fossil_occurrences.shape[1])) 

for i in range(len(unique_bins)):
    this_window_pred_fossils = np.dot(windowed_group_proportions[i,:], grouped_fossil_proportions)
    fossils_pred_facies[i,:] = this_window_pred_fossils


# and plot them vs. stratigraphic height
plt.figure()
plt.plot(fossils_pred_facies[:,0], unique_bins*window_size, label='trilobite', marker='o')
plt.plot(fossils_pred_facies[:,1], unique_bins*window_size, label='brachiopid', marker='o')
plt.plot(fossils_pred_facies[:,2], unique_bins*window_size, label='salterella', marker='o')
plt.plot(fossils_pred_facies[:,3], unique_bins*window_size, label='echinoderm', marker='o')
plt.plot(fossils_pred_facies[:,4], unique_bins*window_size, label='other_fossil', marker='o')
plt.legend()    
plt.show()


# %% run an experiment where we randomly group the samples, and predict the fossils in the strat windows

n_experiments = 1000
n_groups = len(group_sizes)
n_fossils = fossil_occurrences.shape[1]

# make an array to hold the predicted fossils for each experiment
fossils_pred_exp = np.zeros((n_experiments, len(unique_bins), n_fossils))

for i in range(n_experiments):
    # randomly shuffle the group index
    to_shuffle = np.array(group_index).copy()
    np.random.shuffle(to_shuffle)

    windowed_groups = np.zeros((len(unique_bins), len(group_sizes)))
    
    # get the grouped facies that correspond to these samples
    for j in range(len(unique_bins)):
        window_inds = np.where(strat_bins == unique_bins[j])[0]
        for k in range(len(group_sizes)):
            # add up the ones that are in both this group and strat window
            group_leaves = np.where(to_shuffle == k)[0]
            in_group = np.array(R['leaves'])[group_leaves]
            in_both = np.intersect1d(in_group, window_inds)
            windowed_groups[j,k] = len(in_both)
    
        # multiply the proportion of each group by the proportion of the group that has each fossil
        windowed_group_proportions = windowed_groups/np.sum(windowed_groups, axis=1)[:,None]

    for k in range(len(unique_bins)):
        this_window_pred_fossils = np.dot(windowed_group_proportions[k,:], grouped_fossil_proportions)
        fossils_pred_exp[i,k,:] = this_window_pred_fossils

# get the mean squared error for each fossil in the experiments
mse_exp = np.zeros((n_experiments, n_fossils))
for i in range(n_experiments):
    for j in range(n_fossils):
        mse_exp[i,j] = np.mean((fossils_pred_exp[i,:,j] - windowed_fossils[:,j])**2)

# and get the mse for the facies model
mse_facies = np.zeros((n_fossils))
for i in range(n_fossils):
    mse_facies[i] = np.mean((fossils_pred_facies[:,i] - windowed_fossils[:,i])**2)
# %% make a histogram of the mses

# first get the percentile that the facies model falls within the experiments
percentile_facies = np.zeros((n_fossils))
for i in range(n_fossils):
    percentile_facies[i] = np.sum(mse_exp[:,i] > mse_facies[i])/n_experiments

# we'll do each histogram on a separate subplot
plt.figure()
axs = plt.subplots(n_fossils, 1, sharex=True)
for i in range(n_fossils):
    # first get the smoothened histogram of the experiments
    # bin it
    n_bins = 50
    hist, bin_edges = np.histogram(mse_exp[:,i], bins=n_bins)
    # smooth it
    hist_smooth = np.convolve(hist, np.ones(5)/5, mode='same')
    # and plot it
    axs[1][i].plot(bin_edges[0:-1], hist_smooth, label='experiments')

    # and add a vertical line for the facies model
    axs[1][i].axvline(mse_facies[i], color='k', linestyle='dashed', linewidth=1)

    # and add a text label for the percentile
    axs[1][i].text(0.8, 0.8, 'p = ' + str(percentile_facies[i]), horizontalalignment='center', verticalalignment='center', transform=axs[1][i].transAxes)

plt.tight_layout()
plt.show()

# %% cross plot the predicted fossils vs. the actual fossils

plt.figure()
plt.scatter(windowed_fossils[:,0], fossils_pred_facies[:,0], label='trilobite')
plt.scatter(windowed_fossils[:,1], fossils_pred_facies[:,1], label='brachiopid')
plt.scatter(windowed_fossils[:,2], fossils_pred_facies[:,2], label='salterella')
plt.scatter(windowed_fossils[:,3], fossils_pred_facies[:,3], label='echinoderm')
plt.scatter(windowed_fossils[:,4], fossils_pred_facies[:,4], label='other_fossil')
plt.legend()
plt.show()

# %%
