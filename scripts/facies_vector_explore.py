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
from scipy.cluster.hierarchy import dendrogram, linkage # for displaying clusters
import rasterio # for reading geotiffs

#%%
##########################################################################################
# local function imports
##########################################################################################
# %% 
from json_processing import assemble_samples, select_gridded_geochem, select_gridded_im_metrics, select_gridded_point_counts, select_gridded_pa
from geospatial import latlong_to_utm, dip_correct_elev, im_grid
from data_processing import random_sample_strat
from im_processing import sample_3channel  # for plotting
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


# %% select the gridded geochem data, im metrics, and point counts
geochem_df = select_gridded_geochem(outcrop_data, desired_metrics=['delta13c', 'delta18o', 'Li_Ca', 'Na_Ca', 'Mg_Ca', 'K_Ca', 'Mn_Ca', 'Fe_Ca', 'Sr_Ca'])

im_metric_df =  select_gridded_im_metrics(outcrop_data, desired_metrics=['percentile', 'rayleigh_anisotropy', 'entropy'], desired_scales=[1,0.5, 0.25, 0.125, 0.0625, 0.03125, 0.015625, 0.0078125, 0.00390625, 0.001953125])

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

# %% do PCA to get geochem component

# going to be kind of similar to the point count data, just extract data, normalize, and do a PCA
geochem_measurements = geochem_df.columns[7:]
geochem_names = geochem_df.sample_name
geochem_phases = geochem_df.phase
geochem_x_locs = geochem_df.im_loc_x
geochem_y_locs = geochem_df.im_loc_y

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
geochem_x_locs = np.delete(geochem_x_locs, nan_rows)
geochem_y_locs = np.delete(geochem_y_locs, nan_rows)

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

whole_rock_fields = ['sample_name', 'field_lithology', 'lat', 'lon', 'msl', 'anisotropy_pc1', 'anisotropy_pc2', 'anisotropy_pc3', 'pc_pc1', 'pc_pc2', 'pc_pc3', 'pc_pc4', 'pc_pc5']

# and add in the original point count classes
for pc_class in pc_classes:
    whole_rock_fields.append(pc_class)

# add in the presence or absence classes
for pa_class in presence_absence_df.columns[4:]:
    whole_rock_fields.append(pa_class)

# and systematically add in the first 3 pcs for each of the percentile spectra
band_strs = unique_bands_percentile.astype(int).astype(str)
for i in range(3):
    for band in band_strs:
        whole_rock_fields.append(band + '_percentile_pc' + str(i+1))

# we'll use anisotropy as the standard, and give the other pcs inds to match
pc_inds = []
pa_inds = []
percentile_inds = []
for i in range(len(anisotropy_names)):
    sample_name = anisotropy_names[i]
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
    anisotropy_scores_full = np.append(anisotropy_scores_full, np.full((missing_rows, 3, n_bands), np.nan), axis=0)

# assemble the pc array from the first 5 pcs
pc_scores_full = np.zeros((len(anisotropy_names), 5 + len(pc_classes)))
for i in range(len(anisotropy_names)):
    if not np.isnan(pc_inds[i]):
        pc_scores_full[i,:5] = np.squeeze(pc_scores[int(pc_inds[i]),0:5])
        pc_scores_full[i,5:] = pc_data_original[int(pc_inds[i]),:]
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
whole_rock_vector = np.column_stack((anisotropy_names, field_liths, field_locs, anisotropy_scores_full, pc_scores_full, pa_scores_full, percentile_scores_full))
whole_rock_vector = pd.DataFrame(whole_rock_vector, columns=whole_rock_fields)

# combining the dataframe turned the floats back into strings, so convert them back
data_cols = whole_rock_vector.columns[2:]
whole_rock_vector[data_cols] = whole_rock_vector[data_cols].astype(float)


# %% and assemple a phase-wise dataframe, which is really just geochem

phase_fields = ['sample_name', 'field_lithology', 'lat', 'lon', 'msl', 'phase','im_x_loc','im_y_loc', 'carbon_isotopes', 'geochem_pc1', 'geochem_pc2']

# need to redo field liths, and locations
field_liths = []
field_locs = np.zeros((len(geochem_names), 3))

# then add the field lithologies, lat, lon, and msl
for i in range(len(geochem_names)):
    outcrop_index = outcrop_sample_names.index(geochem_names[i])
    field_liths.append(outcrop_data['grid_data'][0]['samples'][outcrop_index]['field_lithology'])
    field_locs[i,0] = outcrop_data['grid_data'][0]['samples'][outcrop_index]['latitude']
    field_locs[i,1] = outcrop_data['grid_data'][0]['samples'][outcrop_index]['longitude']
    field_locs[i,2] = outcrop_data['grid_data'][0]['samples'][outcrop_index]['msl']

# this should be pretty much ready to assemble
phase_vector = np.column_stack((geochem_names, field_liths, field_locs, geochem_phases, geochem_x_locs, geochem_y_locs, carbon_isotopes, geochem_scores[:,:2]))
phase_vector = pd.DataFrame(phase_vector, columns=phase_fields)

# %% and then make a combined vector with the whole rock and phase-wise data combined

# the fields are just the whole phase fields plus the data fields from the whole rock vector
entire_fields = phase_fields + whole_rock_fields[5:]

# need to get indices for the samples in the whole rock vector with respect to the phase vector
phase_wise_names = phase_vector.sample_name

# make an array to hold the indices
whole_rock_inds = []

# iterate through the phase wise names and get where those samples are in the whole rock vector
for i in range(len(phase_wise_names)):
    if phase_wise_names[i] in whole_rock_vector.sample_name.to_numpy():
        whole_rock_inds.append(np.where(whole_rock_vector.sample_name == phase_wise_names[i])[0][0])
    else:
        whole_rock_inds.append(np.nan)

# and now just need to double check for names that are in the whole rock vector but not the phase vector
whole_rock_names = whole_rock_vector.sample_name
for i in range(len(whole_rock_names)):
    if whole_rock_names[i] not in phase_wise_names:
        whole_rock_inds.append(i)
        phase_wise_names = np.append(phase_wise_names, [whole_rock_names[i]])   

# assemble the combined vector
whole_rock_data = whole_rock_vector.iloc[whole_rock_inds,5:].to_numpy().copy()
phase_wise_data = phase_vector.iloc[:,5:].to_numpy().copy()

# and the whole rock data was made into strings for some reason, so convert it back to floats
whole_rock_data = whole_rock_data.astype(float)

# add nan rows to the bottom of the phase wise data to match the whole rock data
missing_rows = whole_rock_data.shape[0] - phase_wise_data.shape[0]
phase_wise_data = np.append(phase_wise_data, np.full((missing_rows, phase_wise_data.shape[1]), np.nan), axis=0)

# redo field lithologies and locations one more time
field_liths = []
field_locs = np.zeros((len(phase_wise_names), 3))

# then add the field lithologies, lat, lon, and msl
for i in range(len(phase_wise_names)):
    outcrop_index = outcrop_sample_names.index(phase_wise_names[i])
    field_liths.append(outcrop_data['grid_data'][0]['samples'][outcrop_index]['field_lithology'])
    field_locs[i,0] = outcrop_data['grid_data'][0]['samples'][outcrop_index]['latitude']
    field_locs[i,1] = outcrop_data['grid_data'][0]['samples'][outcrop_index]['longitude']
    field_locs[i,2] = outcrop_data['grid_data'][0]['samples'][outcrop_index]['msl']

# assemple the combined data
entire_vector = np.column_stack((phase_wise_names, field_liths, field_locs, phase_wise_data, whole_rock_data))
entire_vector = pd.DataFrame(entire_vector, columns=entire_fields)


# %% last thing before looking is to remove the samples from the other mound from all three vecotors

# I'll just list their sample names for now
to_remove = ['smg_167', 'smg_168', 'smg_169', 'smg_170', 'smg_171', 'smg_172', 'smg_173']

# and remove them
whole_rock_vector = whole_rock_vector[~whole_rock_vector.sample_name.isin(to_remove)]
phase_vector = phase_vector[~phase_vector.sample_name.isin(to_remove)]
entire_vector = entire_vector[~entire_vector.sample_name.isin(to_remove)]

# correct the row index
whole_rock_vector.index = range(len(whole_rock_vector))
phase_vector.index = range(len(phase_vector))
entire_vector.index = range(len(entire_vector))

# %% dip-correct the coordinates so we have relative stratigraphic elevation

# first need to correct lats and longs to utm
x_whole, y_whole = latlong_to_utm(whole_rock_vector.lat, whole_rock_vector.lon, zone=11, hemisphere='north')
x_phase, y_phase = latlong_to_utm(phase_vector.lat, phase_vector.lon, zone=11, hemisphere='north')
x_entire, y_entire = latlong_to_utm(entire_vector.lat, entire_vector.lon, zone=11, hemisphere='north')


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
x_corrected_phase, y_corrected_phase, z_corrected_phase = dip_correct_elev(x_phase, y_phase, phase_vector.msl, regional_dip, regional_dip_dir)
x_corrected_entire, y_corrected_entire, z_corrected_entire = dip_correct_elev(x_entire, y_entire, entire_vector.msl, regional_dip, regional_dip_dir)

# %% make a plot of carbon isotopes vs elevation, in this case make all phases gray except for micrite

fig, ax = plt.subplots(figsize=(6,10))
for phase in phase_codes:
    if phase == 'E':
        phase_inds = np.where(phase_vector.phase == phase)[0]
        ax.scatter(phase_vector.carbon_isotopes[phase_inds], z_corrected_phase[phase_inds], label=phase_names[phase_codes.index(phase)], color=cyan)
    else:
        phase_inds = np.where(phase_vector.phase == phase)[0]
        ax.scatter(phase_vector.carbon_isotopes[phase_inds], z_corrected_phase[phase_inds.tolist()], color='gray', alpha=0.2)    

# make the x label delta 13C, but with the delta symbol and 13 as a superscript and permil after
ax.set_xlabel(r'$\delta^{13}$C (‰)')
ax.set_ylabel('Strat height (m)')
ax.legend()
plt.rcParams['pdf.fonttype'] = 42
plt.tight_layout()
plt.show()


# %% randomly sample the carbon isotopes of micrite at desired intervals 100 times and plot the results

height_interval = 10
n_samplings = 30

micrite_carb_vals = phase_vector.carbon_isotopes[phase_vector.phase == 'E']
micrite_heights = z_corrected_phase[phase_vector.phase == 'E']

micrite_random_samples, bin_centers = random_sample_strat(micrite_carb_vals.to_numpy(), micrite_heights, height_interval, n_samplings)

# make a plot of the results
fig, ax = plt.subplots(figsize=(6,10))
# just put all carbon istopes in the background as light gray dots
ax.scatter(phase_vector.carbon_isotopes, z_corrected_phase, color='lightgray', alpha=0.2)
# then plot the random samplings as cyan lines
for i in range(n_samplings):
    ax.plot(micrite_random_samples[:,i], bin_centers, color=cyan, alpha=0.4)

ax.set_xlabel('Carbon Isotopes')
ax.set_ylabel('Strat height (m)')
plt.tight_layout()
plt.show()

# %% now make a plot of all carbon isotopes vs elevation, and allow user to input a boundary to separate some samples

fig, ax = plt.subplots(figsize=(4,6))
ax.scatter(phase_vector.carbon_isotopes, z_corrected_phase, color='lightgray', alpha=0.2)

# make a line to separate the samples
print('Click to place a separator line')
separator = plt.ginput(-1)

separator = np.array(separator)

# now we need to assign a separator value to each height in the z_corrected array, interpolating between the separator point above and below
separator_values = np.zeros(len(z_corrected_phase))
for i in range(len(z_corrected_phase)):
    # find the two separator points that this height is between
    above = np.where(separator[:,1] > z_corrected_phase[i])[0]
    below = np.where(separator[:,1] < z_corrected_phase[i])[0]
    if len(above) > 0 and len(below) > 0:
        above = above[-1]
        below = below[0]
        # interpolate between the two points
        separator_values[i] = separator[below,0] + (separator[above,0] - separator[below,0])*((z_corrected_phase[i] - separator[below,1])/(separator[above,1] - separator[below,1]))
    elif len(above) == 0:
        below = below[0]
        separator_values[i] = separator[below,0]
    elif len(below) == 0:
        above = above[-1]
        separator_values[i] = separator[above,0]

# and then separate
is_below = phase_vector.carbon_isotopes < separator_values

# make a plot of the results
fig, ax = plt.subplots(figsize=(6,10))
ax.scatter(phase_vector.carbon_isotopes[is_below], z_corrected_phase[is_below], color=rose, alpha=0.5, label='Below')
ax.scatter(phase_vector.carbon_isotopes[~is_below], z_corrected_phase[~is_below], color='gray', alpha=0.2, label='Above')

ax.set_xlabel('Carbon Isotopes')
ax.set_ylabel('Strat height (m)')
ax.legend()
plt.tight_layout()
plt.show()



# %% print out the inds of the micrite samples below the separator

micrite_inds = np.where(phase_vector.phase == 'E')[0]
micrite_inds_below = np.where(is_below & (phase_vector.phase == 'E'))[0]

print('Micrite samples below the separator:')
for i in range(len(micrite_inds_below)):
    print('Sample:', phase_vector.sample_name[micrite_inds_below[i]])
    print('Strat height:', z_corrected_phase[micrite_inds_below[i]])
    print('Carbon isotopes:', phase_vector.carbon_isotopes[micrite_inds_below[i]])


# %% look at the image of any rock sample below the separator, with the drill marks plotted

ind_desired = 820
row_ind = np.where(is_below)[0][ind_desired]

# get the sample name, and from that take the index in the outcrop data
sample_name = phase_vector.sample_name[row_ind]
outcrop_index = outcrop_sample_names.index(sample_name)

# we can extract the sample dict from the outcrop data, and make an rgb image
sample_dict = outcrop_data['grid_data'][0]['samples'][outcrop_index]
rgb_image = sample_3channel(sample_dict)

this_phase = phase_vector.phase[row_ind]

# and plot the image
fig, ax = plt.subplots(figsize=(8,8))
ax.imshow(rgb_image)
# and plot the drill mark indicated by the row_ind in the geochem data
drill_x = phase_vector.im_x_loc[row_ind]
drill_y = phase_vector.im_y_loc[row_ind]
drill_x = drill_x * rgb_image.shape[1]
drill_y = drill_y * rgb_image.shape[0]
ax.scatter(drill_x, drill_y, color='red', s=100, marker='x')

# make the title the phase
ax.set_title(phase_names[phase_codes.index(this_phase)])

# print the sample strat height and phase and sample name
print('Strat height:', z_corrected_phase[row_ind])
print('Phase:', this_phase)
print('Sample name:', sample_name)

plt.tight_layout()
plt.show()

# %% make a correction factor for carbon isotopes based upon the correlation between carbon isotopes and geochem_pc2

# first get the correlation coefficient, need to remove nans
good_inds = ~np.isnan(phase_vector.geochem_pc2.astype(float)) & ~np.isnan(phase_vector.carbon_isotopes.astype(float))
correlation = np.corrcoef(phase_vector.geochem_pc2.astype(float)[good_inds], phase_vector.carbon_isotopes.astype(float)[good_inds])

# get the equation of the line
m = correlation[0,1]*np.std(phase_vector.carbon_isotopes.astype(float)[good_inds])/np.std(phase_vector.geochem_pc2.astype(float)[good_inds])
b = np.mean(phase_vector.carbon_isotopes.astype(float)[good_inds]) - m*np.mean(phase_vector.geochem_pc2.astype(float)[good_inds])

# now apply the correction
correction_factor = m*phase_vector.geochem_pc2.astype(float) + b
corrected_carbon_isotopes = phase_vector.carbon_isotopes.astype(float) - correction_factor

# %% replot the carbon isotopes vs elevation, but with the corrected values, still color coded by if they are below the separator

fig, ax = plt.subplots(figsize=(6,10))

ax.scatter(corrected_carbon_isotopes[is_below], z_corrected_phase[is_below], color=rose, alpha=0.5, label='Below')
ax.scatter(corrected_carbon_isotopes[~is_below], z_corrected_phase[~is_below], color='gray', alpha=0.2, label='Above')

ax.set_xlabel('Corrected Carbon Isotopes')
ax.set_ylabel('Strat height (m)')
ax.legend()
plt.tight_layout()
plt.show()

# %% replot the carbon isotopes vs elevation, but with the corrected values, gray if they are above, color coded by phase if below

fig, ax = plt.subplots(figsize=(8,10))

symbols = ['o', 's', 'D', 'v', '^', '<', '>', 'p', 'P', '*', 'X']
n_colors = len(plt.rcParams['axes.prop_cycle'].by_key()['color'])

ax.scatter(corrected_carbon_isotopes[~is_below], z_corrected_phase[~is_below], color='gray', alpha=0.2)

count = 0
for phase in phase_codes:
    if count < n_colors:
        phase_inds = np.where(phase_vector.phase == phase)[0]
        below_inds = np.where(is_below & (phase_vector.phase == phase))[0]
        ax.scatter(corrected_carbon_isotopes[below_inds], z_corrected_phase[below_inds], label=phase_names[phase_codes.index(phase)], color=plt.rcParams['axes.prop_cycle'].by_key()['color'][count], marker=symbols[0])

    else:
        phase_inds = np.where(phase_vector.phase == phase)[0]
        below_inds = np.where(is_below & (phase_vector.phase == phase))[0]
        ax.scatter(corrected_carbon_isotopes[below_inds], z_corrected_phase[below_inds], label=phase_names[phase_codes.index(phase)], color=plt.rcParams['axes.prop_cycle'].by_key()['color'][count-n_colors], marker=symbols[1])
    
    count += 1

ax.set_xlabel('Corrected Carbon Isotopes')
ax.set_ylabel('Strat height (m)')
# place the legend outside the plot
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()


# %% with the idea that being below the separator is most common in allochthonous settings, plot c isotopes vs. some bedded variable

# to access the entire vector, need to add falses to the bottom to match the length of the entire vector
missing_rows = entire_vector.shape[0] - phase_vector.shape[0]
is_below_full = np.append(is_below, np.full(missing_rows, False))

# make a plot of carbon isotopes vs microbial fraction, color coded by if they are below the separator
x_var = 'carbon_isotopes'
y_var = 'anisotropy_pc2'
phases = ['E', 'F', 'G', 'I']
phase_colors = [cyan, wine, teal, purple]

fig, ax = plt.subplots(figsize=(7,4))
#ax.scatter(entire_vector[x_var][entire_vector.phase == phase], entire_vector[y_var][entire_vector.phase == phase], color='lightgray', alpha=0.2)
for phase in phases:
    below_inds = np.where(is_below_full & (entire_vector.phase == phase))[0]
    above_inds = np.where(~is_below_full & (entire_vector.phase == phase))[0]
    ax.scatter(entire_vector[x_var][below_inds], entire_vector[y_var][below_inds], label=phase_names[phase_codes.index(phase)], color=phase_colors[phases.index(phase)])
    ax.scatter(entire_vector[x_var][above_inds], entire_vector[y_var][above_inds], color='gray', alpha=0.2)

ax.set_xlabel(x_var)
ax.set_ylabel(y_var)
# set y limits -4 to 4
ax.set_ylim(-4,4)
# put the legend outside the plot
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

# %% there are a couple of outliers in the the below samples in terms of aniso_pc2, so let's look at those

phases = ['E', 'F', 'G', 'I']

# get the inds of the below
below_inds = np.where(is_below_full)[0]

# and the inds of any sample that is in the phases we are looking at
phase_inds = np.zeros(len(entire_vector), dtype=bool)
for phase in phases:
    phase_inds = phase_inds | (entire_vector.phase == phase)

# get the inds of the below samples that are in the phases we are looking at
below_phase_inds = np.where(is_below_full & phase_inds)[0]

# print out the 2 highest and 2 lowest aniso_pc2 values and sample names
print('Highest aniso_pc2 values:')
for i in range(2):
    ind = np.argmax(entire_vector.anisotropy_pc2[below_phase_inds])
    print('Sample:', entire_vector.sample_name[below_phase_inds[ind]])
    print('Aniso_pc2:', entire_vector.anisotropy_pc2[below_phase_inds[ind]])
    below_phase_inds = np.delete(below_phase_inds, ind)

print('Lowest aniso_pc2 values:')
for i in range(2):
    ind = np.argmin(entire_vector.anisotropy_pc2[below_phase_inds])
    print('Sample:', entire_vector.sample_name[below_phase_inds[ind]])
    print('Aniso_pc2:', entire_vector.anisotropy_pc2[below_phase_inds[ind]])
    below_phase_inds = np.delete(below_phase_inds, ind)






# %% show the histogram of any variable, color coded by presence/absence of certain fossils

# fossils we are checking for
to_check = ['trilobite', 'brachiopid', 'salterella']

# metric we are looking at
histo_metric = 'anisotropy_pc2'

# do the logical thing to get the inds of the samples that have the fossils
has_them = np.zeros((len(whole_rock_vector), len(to_check)))
for i in range(len(to_check)):
    has_them[:,i] = whole_rock_vector[to_check[i]] == 1.

# make into a bool
has_them = has_them.astype(bool)

# note which have all
has_all = np.all(has_them, axis=1)

# make a subplot for each fossil, plus one more for all
fig, ax = plt.subplots(len(to_check)+1, 1, figsize=(8,6), sharex=True)
for i in range(len(to_check)):
    ax[i].hist(whole_rock_vector[histo_metric][has_them[:,i]], color='gray', alpha=0.5, bins=20, label='Has ' + to_check[i])
    ax[i].hist(whole_rock_vector[histo_metric][~has_them[:,i]], color=cyan, alpha=0.5, bins=20, label='Does not have ' + to_check[i])
    ax[i].legend()
    ax[i].set_title(to_check[i])

# and the last one for all
ax[-1].hist(whole_rock_vector[histo_metric][has_all], color='gray', alpha=0.5, bins=20, label='Has all')
ax[-1].hist(whole_rock_vector[histo_metric][~has_all], color=cyan, alpha=0.5, bins=20, label='Does not have all')
ax[-1].legend()
ax[-1].set_title('All')

ax[-1].set_xlabel(histo_metric)
plt.tight_layout()
plt.show()

# %% looking at the histograms, we maybe see some rules emerge, so let's define them and apply them

ooids_thresh = 0.01
archaeo_thresh1 = 0.01
archaeo_thresh2 = 0.2
dol_thresh1 = 0.01
dol_thresh1 = 0.2
microb_thresh1 = 0.01
microb_thresh2 = 0.6
aniso2_thresh1 = -2
aniso2_thresh2 = 2

# make a data array that is just the whole rock data for the metrics we are looking at, remove any nans
data_fields = ['ooid', 'Arch', 'Dol', 'Microb', 'anisotropy_pc2']
data_array = whole_rock_vector[data_fields].to_numpy()
to_remove = np.where(np.isnan(data_array).any(axis=1))
data_array = np.delete(data_array, to_remove, axis=0)

# apply the rules
pass_ooids = data_array[:,0] > ooids_thresh
pass_arch1 = data_array[:,1] > archaeo_thresh1
pass_arch2 = data_array[:,1] < archaeo_thresh2
pass_dol1 = data_array[:,2] > dol_thresh1
pass_dol2 = data_array[:,2] < dol_thresh1
pass_microb1 = data_array[:,3] > microb_thresh1
pass_microb2 = data_array[:,3] < microb_thresh2
pass_aniso1 = data_array[:,4] > aniso2_thresh1
pass_aniso2 = data_array[:,4] < aniso2_thresh2

# and combine them into one array, and group by the rocks that pass the same rules
pass_array = np.column_stack((pass_ooids, pass_arch1, pass_arch2,  pass_microb1, pass_microb2, pass_dol1, pass_dol2))

# to accompany the pass array, make an array with the fossils presence-absence
# fossils we are checking for
to_check = ['trilobite', 'brachiopid', 'salterella']

# do the logical thing to get the inds of the samples that have the fossils
has_them = np.zeros((len(whole_rock_vector), len(to_check)))
for i in range(len(to_check)):
    has_them[:,i] = whole_rock_vector[to_check[i]] == 1.

# make into a bool
has_them = has_them.astype(bool)

# note which have all
has_all = np.all(has_them, axis=1)

# take away the same rows as the data array
has_them = np.delete(has_them, to_remove, axis=0)
has_all = np.delete(has_all, to_remove)
fossil_occurrences = np.column_stack((has_them, has_all))

# make an array of all the unique combinations, and an equivalent fossil counting array
unique_combinations = np.unique(pass_array, axis=0)
fossil_counts = np.zeros((len(unique_combinations),4))

# make a bar plot of the number of samples in each group
n_samples = np.zeros(len(unique_combinations))
for i in range(len(unique_combinations)):
    pass_logical = np.all(pass_array == unique_combinations[i], axis=1)
    pass_inds = np.where(pass_logical)[0]
    n_samples[i] = np.sum(pass_logical)
    fossil_counts[i,:] = fossil_counts[i,:] + np.sum(fossil_occurrences[pass_inds].astype(int), axis=0)

fig, ax = plt.subplots(figsize=(8,6))
# make bars for both the unique combinations and the fossils
bar_width = 0.35
bar_inds = np.arange(len(unique_combinations))
ax.bar(bar_inds, n_samples, bar_width, label='Unique Combinations')
ax.bar(bar_inds + bar_width, fossil_counts[:,1], bar_width, label='Fossil Counts')
ax.set_xticks(np.arange(len(unique_combinations)))

# make the x labels the combination of 0 and 1 it corresponds to
x_labels = []
for i in range(len(unique_combinations)):
    bool_to_int = unique_combinations[i].astype(int)
    x_labels.append(''.join([str(x) for x in bool_to_int]))

ax.set_xticklabels(x_labels)
# make the x ticks vertical
plt.xticks(rotation=90)
ax.set_ylabel('Number of Samples')

# make a legend
ax.legend()

plt.tight_layout()
plt.show()

# %% cluster the samples based upon the one hot encoded data from the rules aboove

# for any characters that we gave 2 thresholds, we need to combine them into one character.
# being below the threshold is 0, being between the thresholds is 1, and being above both thresholds is 2
pass_array = np.zeros((data_array.shape[0], 5))
for i in range(data_array.shape[0]):
    if pass_ooids[i]:
        pass_array[i,0] = 2
    elif data_array[i,0] > ooids_thresh:
        pass_array[i,0] = 1
    if pass_arch1[i]:
        pass_array[i,1] = 0
    elif pass_arch2[i]:
        pass_array[i,1] = 2
    else:
        pass_array[i,1] = 1
    if pass_dol1[i]:
        pass_array[i,2] = 0
    elif pass_dol2[i]:
        pass_array[i,2] = 2
    else:
        pass_array[i,2] = 1
    if pass_microb1[i]:
        pass_array[i,3] = 0
    elif pass_microb2[i]:
        pass_array[i,3] = 2
    else:
        pass_array[i,3] = 1
    if pass_aniso1[i]:
        pass_array[i,4] = 0
    elif pass_aniso2[i]:
        pass_array[i,4] = 2
    else:
        pass_array[i,4] = 1


# do the linkage
linkage_matrix = linkage(pass_array, method='ward')


# make a dendrogram
dendro = dendrogram(linkage_matrix,truncate_mode='level', p=2)

# based upon the leaves of the dendrogram, and the linkage matrix, we can make cluster assignments
leaves = dendro['leaves']

# initiate the cluster list with the cluster number for each leaf
cluster_list = [[leaf] for leaf in leaves]

# loop through each leaf to get the cluster list 
for i in range(len(leaves)):    
    still_going = True

    # quick check if the leaf is already just a single sample
    if len(cluster_list[i]) == 1 and cluster_list[i][0] <= len(linkage_matrix):
        continue
    # otherwise we gotta keep going
    while still_going:
        # get the cluster numbers available for this leaf
        these_clusters = cluster_list[i].copy()

        # if it's a single sample, we know we should just split it and move on
        if len(these_clusters) == 1:
            linkage_ind = these_clusters[0] - len(pass_array)
            cluster_list[i] = [linkage_matrix[linkage_ind,0], linkage_matrix[linkage_ind,1]]
            # and go back through the while loop
            continue

        # otherwise, we need to scan for clusters to split
        to_split = []
        for j in range(len(these_clusters)):
            if these_clusters[j] > len(linkage_matrix):
                to_split.append(j)

        # if there are no clusters to split, we are done
        if len(to_split) == 0:
            still_going = False
            continue

        # otherwise, we need to split each of the to_split clusters
        for j in to_split:
            # get the cluster number
            cluster_num = these_clusters[j]

            # get the linkage index
            linkage_ind = cluster_num - len(pass_array)

            # get the two clusters that result from the split
            cluster1 = linkage_matrix[int(linkage_ind),0]
            cluster2 = linkage_matrix[int(linkage_ind),1]

            # remove the old cluster and add the new ones
            cluster_list[i].remove(cluster_num)
            cluster_list[i].append(cluster1)
            cluster_list[i].append(cluster2)

# list the sample names we've included
sample_names = whole_rock_vector.sample_name.to_numpy()
sample_names = np.delete(sample_names, to_remove)

# turn the cluster list into a list of sample names
cluster_names = []
for i in range(len(cluster_list)):
    these_cluster_names = []
    for j in range(len(cluster_list[i])):
        if cluster_list[i][j] < len(pass_array):
            these_cluster_names.append(sample_names[int(cluster_list[i][j])])
    cluster_names.append(these_cluster_names)

# for each cluster, go through the sample names and gather the fossil occurrences for each
cluster_fossil_counts = []
reduced_sample_names = whole_rock_vector.sample_name.to_numpy()
reduced_sample_names = np.delete(reduced_sample_names, to_remove)
for i in range(len(cluster_names)):
    these_cluster_fossils = np.zeros(4)
    for j in range(len(cluster_names[i])):
        sample_ind = np.where(reduced_sample_names == cluster_names[i][j])[0][0]
        these_cluster_fossils = these_cluster_fossils + fossil_occurrences[sample_ind].astype(int)
    cluster_fossil_counts.append(these_cluster_fossils)

# now show the dendrogram with a bar plot of the proportion of samples that have all fossils
fig, ax = plt.subplots(2,1, figsize=(8,8))
dendro = dendrogram(linkage_matrix, ax=ax[0], truncate_mode='level', p=2)
bar_inds = np.arange(len(cluster_fossil_counts))
n_samples = np.zeros(len(cluster_fossil_counts))
for i in range(len(cluster_fossil_counts)):
    n_samples[i] = np.sum(cluster_fossil_counts[i])
proportion_all = np.zeros(len(cluster_fossil_counts))
for i in range(len(cluster_fossil_counts)):
    proportion_all[i] = cluster_fossil_counts[i][3]/n_samples[i]
bar_width = 0.35
ax[1].bar(bar_inds, proportion_all, bar_width)
ax[1].set_xticks(np.arange(len(cluster_fossil_counts)))
ax[1].set_xticklabels(np.arange(len(cluster_fossil_counts)))
ax[1].set_xlabel('Cluster Number')
ax[1].set_ylabel('Proportion of Samples with All Fossils')
plt.tight_layout()
plt.show()

# %% make a map of presence/absence of all fossils

# load in the dem from the outcrop data
dem_file = outcrop_data['dem_file']

# get the boundaries of the dem to make a grid
with rasterio.open(dem_file) as src:
    bounds = src.bounds 
    dem_im = src.read(1)

# the bounds are in utm, so make 2 pairs that represent the corners and convert to lat long
bound_lat = [bounds.bottom, bounds.top]
bound_lon = [bounds.left, bounds.right]
bound_x, bound_y = latlong_to_utm(bound_lat, bound_lon, zone=11, hemisphere='north')

# make a grid of the dem
im_bounds = (bound_x[0], bound_x[1], bound_y[0], bound_y[1])
im_width = dem_im.shape[1]
im_height = dem_im.shape[0]
x_grid, y_grid = im_grid(im_bounds, [im_width, im_height])

# use the right whole rock x and y locations to understand max extents of the samples
sample_x = np.delete(x_whole, to_remove)
sample_y = np.delete(y_whole, to_remove)

# get the max and min of the sample locations, and add a buffer if we want
buffer_size = 100 # in meters
x_min = np.min(sample_x) - buffer_size
x_max = np.max(sample_x) + buffer_size
y_min = np.min(sample_y) - buffer_size
y_max = np.max(sample_y) + buffer_size

# find the row and column inds in the dem that correspond to the max and min values
x_min_ind = np.argmin(np.abs(x_grid[0,:] - x_min))
x_max_ind = np.argmin(np.abs(x_grid[0,:] - x_max))
y_min_ind = np.argmin(np.abs(y_grid[:,0] - y_min))
y_max_ind = np.argmin(np.abs(y_grid[:,0] - y_max))

# crop the dem to the max and min values
dem_im_cropped = dem_im[y_max_ind:y_min_ind, x_min_ind:x_max_ind]
x_grid_cropped = x_grid[y_max_ind:y_min_ind, x_min_ind:x_max_ind]
y_grid_cropped = y_grid[y_max_ind:y_min_ind, x_min_ind:x_max_ind]

# any vlues below 0 are nans
dem_im_cropped[dem_im_cropped < 0] = np.nan

# and we need the pixel indices of the sample locations
im_scale = (x_grid[0,1] - x_grid[0,0])
sample_x_inds = np.round((sample_x - x_grid_cropped[0,0])/im_scale).astype(int)
sample_y_inds = np.round((sample_y - y_grid_cropped[0,0])/im_scale).astype(int)
# make the y_inds the right way around
sample_y_inds = np.abs(sample_y_inds)
#sample_y_inds = dem_im_cropped.shape[0] - sample_y_inds

# plot the dem
fig, ax = plt.subplots(figsize=(8,6))
ax.imshow(dem_im_cropped, cmap='gray')
ax.set_xlabel('easting (m)')
ax.set_ylabel('northing (m)')
# make a colorbar
cbar = plt.colorbar(ax.imshow(dem_im_cropped, cmap='gray'))
cbar.set_label('Elevation (m)')

# make the ticks so they make sense with the utm coordinates
x_ticks = np.linspace(0, dem_im_cropped.shape[1], 5)
x_tick_labels = np.round(x_grid_cropped[0,0] + x_ticks*im_scale).astype(int)
ax.set_xticks(x_ticks)
ax.set_xticklabels(x_tick_labels)

y_ticks = np.linspace(0, dem_im_cropped.shape[0], 5)
y_tick_labels = np.round(y_grid_cropped[0,0] + y_ticks*im_scale).astype(int)
# gotta flip the y labels
y_tick_labels = np.flip(y_tick_labels)
ax.set_yticks(y_ticks)
ax.set_yticklabels(y_tick_labels)

# plot the sample locations that have all fossils as small, o's
sample_y_inds_all = sample_y_inds[has_all]
sample_x_inds_all = sample_x_inds[has_all]
ax.scatter(sample_x_inds_all, sample_y_inds_all, color='black', s=10, label='All Fossils', marker='o')

# plot the sample locations that do not have all fossils, as small x's
sample_y_inds_not_all = sample_y_inds[~has_all]
sample_x_inds_not_all = sample_x_inds[~has_all]
ax.scatter(sample_x_inds_not_all, sample_y_inds_not_all, color='gray', s=10, label='All Fossils', marker='x')

ax.legend()
plt.tight_layout()
plt.show()

# %% make a map of the clusters


# plot the dem
fig, ax = plt.subplots(figsize=(8,6))
ax.imshow(dem_im_cropped, cmap='gray')
ax.set_xlabel('easting (m)')
ax.set_ylabel('northing (m)')
# make a colorbar
cbar = plt.colorbar(ax.imshow(dem_im_cropped, cmap='gray'))
cbar.set_label('Elevation (m)')

# make the ticks so they make sense with the utm coordinates
x_ticks = np.linspace(0, dem_im_cropped.shape[1], 5)
x_tick_labels = np.round(x_grid_cropped[0,0] + x_ticks*im_scale).astype(int)
ax.set_xticks(x_ticks)
ax.set_xticklabels(x_tick_labels)

y_ticks = np.linspace(0, dem_im_cropped.shape[0], 5)
y_tick_labels = np.round(y_grid_cropped[0,0] + y_ticks*im_scale).astype(int)
# gotta flip the y labels
y_tick_labels = np.flip(y_tick_labels)
ax.set_yticks(y_ticks)
ax.set_yticklabels(y_tick_labels)

# loop through the clusters and plot the sample locations
for i in range(len(cluster_list)):
    # need to turn the cluster list into ints
    cluster_list_int = [int(x) for x in cluster_list[i]]
    sample_y_inds_cluster = sample_y_inds[cluster_list_int]
    sample_x_inds_cluster = sample_x_inds[cluster_list_int]
    ax.scatter(sample_x_inds_cluster, sample_y_inds_cluster, s=50, label='Cluster ' + str(i))

ax.legend(loc='upper left')
plt.tight_layout()
plt.show()

# %% make the same map, but leave the symbol unfilled if it does not have all fossils

# plot the dem
fig, ax = plt.subplots(figsize=(8,6))
ax.imshow(dem_im_cropped, cmap='gray')
ax.set_xlabel('easting (m)')
ax.set_ylabel('northing (m)')
# make a colorbar
cbar = plt.colorbar(ax.imshow(dem_im_cropped, cmap='gray'))
cbar.set_label('Elevation (m)')
# make the ticks so they make sense with the utm coordinates
x_ticks = np.linspace(0, dem_im_cropped.shape[1], 5)
x_tick_labels = np.round(x_grid_cropped[0,0] + x_ticks*im_scale).astype(int)
ax.set_xticks(x_ticks)
ax.set_xticklabels(x_tick_labels)

y_ticks = np.linspace(0, dem_im_cropped.shape[0], 5)
y_tick_labels = np.round(y_grid_cropped[0,0] + y_ticks*im_scale).astype(int)
# gotta flip the y labels
y_tick_labels = np.flip(y_tick_labels)
ax.set_yticks(y_ticks)
ax.set_yticklabels(y_tick_labels)

# loop through the clusters and plot the sample locations
for i in range(len(cluster_list)):
    # need to turn the cluster list into ints
    cluster_list_int = [int(x) for x in cluster_list[i]]
    sample_y_inds_cluster = sample_y_inds[cluster_list_int]
    sample_x_inds_cluster = sample_x_inds[cluster_list_int]
    # get the inds of the samples that have all fossils
    has_all_cluster = np.all(fossil_occurrences[cluster_list_int].astype(bool), axis=1)
    # plot the samples that have all fossils
    ax.scatter(sample_x_inds_cluster[has_all_cluster], sample_y_inds_cluster[has_all_cluster], s=50, label='Cluster ' + str(i), color=muted_colors[i])
    # plot the samples that do not have all fossils, not filled
    ax.scatter(sample_x_inds_cluster[~has_all_cluster], sample_y_inds_cluster[~has_all_cluster], s=50, color=muted_colors[i], facecolors='none')

ax.legend(loc='upper left')
plt.tight_layout()
plt.show()

