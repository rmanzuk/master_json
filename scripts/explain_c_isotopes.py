# script with sections to investigate trends in carbon isotopes
# written by R. A. Manzuk 04/15/2024
# last updated 06/26/2024

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

#%%
##########################################################################################
# local function imports
##########################################################################################
# %% 
from json_processing import assemble_samples, select_gridded_geochem, select_gridded_im_metrics, select_gridded_point_counts
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

# select the gridded geochem data, and look at that first
geochem_df = select_gridded_geochem(outcrop_data, desired_metrics=['delta13c', 'delta18o', 'Li_Ca', 'Na_Ca', 'Mg_Ca', 'K_Ca', 'Mn_Ca', 'Fe_Ca', 'Sr_Ca'])

# %% do PCA to get geochem components

# going to be kind of similar to the point count data, just extract data, normalize, and do a PCA
geochem_measurements = geochem_df.columns[7:]
geochem_names = geochem_df.sample_name
geochem_phases = geochem_df.phase
geochem_x_locs = geochem_df.im_loc_x
geochem_y_locs = geochem_df.im_loc_y
geochem_lat = geochem_df.latitude
geochem_long = geochem_df.longitude
geochem_msl = geochem_df.msl

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
geochem_lat = np.delete(geochem_lat, nan_rows)
geochem_long = np.delete(geochem_long, nan_rows)
geochem_msl = np.delete(geochem_msl, nan_rows)

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

# %% start by looking at things stratigraphically, so need to dip correct the elevations

# remove samples in the adjacent mound
to_remove = ['smg_167', 'smg_168', 'smg_169', 'smg_170', 'smg_171', 'smg_172', 'smg_173']
to_remove_inds = np.where(np.isin(geochem_names, to_remove))[0]
geochem_data = np.delete(geochem_data, to_remove_inds, axis=0)
geochem_names = np.delete(geochem_names, to_remove_inds)
geochem_phases = np.delete(geochem_phases, to_remove_inds)
geochem_x_locs = np.delete(geochem_x_locs, to_remove_inds)
geochem_y_locs = np.delete(geochem_y_locs, to_remove_inds)
geochem_lat = np.delete(geochem_lat, to_remove_inds)
geochem_long = np.delete(geochem_long, to_remove_inds)
geochem_msl = np.delete(geochem_msl, to_remove_inds)
carbon_isotopes = np.delete(carbon_isotopes, to_remove_inds)
geochem_scores = np.delete(geochem_scores, to_remove_inds, axis=0)

# first need to correct lats and longs to utm
x_geochem, y_geochem = latlong_to_utm(geochem_lat, geochem_long, zone=11, hemisphere='north')

# go into the outcrop data and get the regional strike and dip
regional_strike = outcrop_data['reg_strike_dip'][0]
regional_dip = outcrop_data['reg_strike_dip'][1]

# correct the strike to be a dip direction
regional_dip_dir = regional_strike + 90
if regional_dip_dir > 360:
    regional_dip_dir = regional_dip_dir - 360

# do the dip correction
x_corrected_geochem, y_corrected_geochem, z_corrected_geochem = dip_correct_elev(x_geochem, y_geochem, geochem_msl, regional_dip, regional_dip_dir)

# %% plot carbon isotopes vs. geochem pc2, color coded by phase

# but we'll group archaeocyaths together, microbial with calcimicrobes, and coralomorphs with shells
grouped_codes = geochem_phases.copy()
grouped_codes[np.isin(grouped_codes, ['F', 'G'])] = 'F'
grouped_codes[np.isin(grouped_codes, ['H', 'J'])] = 'J'
grouped_codes[np.isin(grouped_codes, ['I', 'K'])] = 'I'


ig, ax = plt.subplots(1,1, figsize=(5,5))

new_unique_codes = np.unique(grouped_codes)

# do it point by point so things are scrambled in z order
for point in range(geochem_scores.shape[0]):
    ax.scatter(carbon_isotopes[point], geochem_scores[point,1], color=muted_colors[new_unique_codes.tolist().index(grouped_codes[point])], label=phase_names[phase_codes.index(grouped_codes[point])])

ax.set_xlabel(r'$\delta^{13}$C (‰)')
ax.set_ylabel('PC2')
#ax.legend()

# get the correlation line,
correlation = np.corrcoef(carbon_isotopes, geochem_scores[:,1])

# and plot the line
m,b = np.polyfit(geochem_scores[:,1], carbon_isotopes, 1)
y_eval = np.array([-4, 6])
x_eval = m*y_eval + b
ax.plot(x_eval, y_eval, color='black')
# set x and y limits
ax.set_xlim(-4,2)
ax.set_ylim(-4,6)


plt.tight_layout()

# save the figure
out_path = '/Users/ryan/Dropbox (Princeton)/figures/reef_survey/neomorphism_carbon_isotopes_correction/'
#plt.savefig(out_path+'carbon_isotopes_vs_geochem_pc2.pdf', dpi=300)
plt.show()


# %% make 2 subplots, one of carbon isotopes vs strat height, and one wiht the same things but where carbon isotopes are corrected by the correlation

# correct the carbon isotopes
correlation = np.corrcoef(geochem_scores[:,1], carbon_isotopes)

# get the equation of the line
m = correlation[0,1]*np.std(carbon_isotopes)/np.std(geochem_scores[:,1])
b = np.mean(carbon_isotopes) - m*np.mean(geochem_scores[:,1])

# now apply the correction
correction_factor = m*geochem_scores[:,1] + b
corrected_carbon_isotopes = carbon_isotopes - correction_factor

fig, ax = plt.subplots(1,2, figsize=(5,5))
# and plot all phases gray, except for the micrite in blue
for phase in phase_codes:
    if phase == 'E':
        ax[0].scatter(carbon_isotopes[geochem_phases == phase], z_corrected_geochem[geochem_phases == phase], color=cyan, alpha=0.5, label=phase_names[phase_codes.index(phase)])
        ax[1].scatter(corrected_carbon_isotopes[geochem_phases == phase], z_corrected_geochem[geochem_phases == phase], color=cyan, alpha=0.5, label=phase_names[phase_codes.index(phase)])
    else:
        ax[0].scatter(carbon_isotopes[geochem_phases == phase], z_corrected_geochem[geochem_phases == phase], color='gray', alpha=0.2)
        ax[1].scatter(corrected_carbon_isotopes[geochem_phases == phase], z_corrected_geochem[geochem_phases == phase], color='gray', alpha=0.2)

ax[0].set_xlabel(r'$\delta^{13}$C (‰)')
ax[0].set_ylabel('Strat height (m)')
ax[0].legend()
ax[1].set_xlabel('Corrected $\delta^{13}$C (‰)')
ax[1].set_ylabel('Strat height (m)')
ax[1].legend()

# get the correlations between carbon isotopes and strat height for both
raw_corr = np.corrcoef(carbon_isotopes, z_corrected_geochem)
corr_corr = np.corrcoef(corrected_carbon_isotopes, z_corrected_geochem)

raw_m, raw_b = np.polyfit(carbon_isotopes, z_corrected_geochem, 1)
corr_m, corr_b = np.polyfit(corrected_carbon_isotopes, z_corrected_geochem, 1)

x_eval = np.array([-4, 2])
y_eval_raw = raw_m*x_eval + raw_b
y_eval_corr = corr_m*x_eval + corr_b

ax[0].plot(x_eval, y_eval_raw, color='black')
ax[1].plot(x_eval, y_eval_corr, color='black')

# set x limits
ax[0].set_xlim(-4,2)
ax[1].set_xlim(-4,2)

# set y limits
ax[0].set_ylim(-5, 150)
ax[1].set_ylim(-5, 150)

plt.tight_layout()

# save the figure
out_path = '/Users/ryan/Dropbox (Princeton)/figures/reef_survey/neomorphism_carbon_isotopes_correction/'
#plt.savefig(out_path+'carbon_isotopes_vs_strat_height.pdf', dpi=300)

plt.show()
# %% make histograms of before and after correction

fig, ax = plt.subplots(1,2, figsize=(5,2))
# put micrite in blue, and one for all other phases behind it in gray
non_micrite_inds = np.where(geochem_phases != 'E')[0]

ax[0].hist(carbon_isotopes[non_micrite_inds], bins=20, color='gray', alpha=0.5, label='All other phases')
ax[0].hist(carbon_isotopes[geochem_phases == 'E'], bins=20, color=cyan, alpha=0.5, label='Micrite')

ax[1].hist(corrected_carbon_isotopes[non_micrite_inds], bins=20, color='gray', alpha=0.5, label='All other phases')
ax[1].hist(corrected_carbon_isotopes[geochem_phases == 'E'], bins=20, color=cyan, alpha=0.5, label='Micrite')

ax[0].set_xlabel(r'$\delta^{13}$C (‰)')
ax[0].set_ylabel('Count')
ax[0].legend()
ax[1].set_xlabel('Corrected $\delta^{13}$C (‰)')
ax[1].set_ylabel('Count')
ax[1].legend()

ax[0].set_xlim(-4,2)
ax[1].set_xlim(-4,2)

plt.tight_layout()

# save the figure
out_path = '/Users/ryan/Dropbox (Princeton)/figures/reef_survey/neomorphism_carbon_isotopes_correction/'
#plt.savefig(out_path+'carbon_isotopes_histograms.pdf', dpi=300)

plt.show()

# %% because the low population of carbon isotopes isn't corrected for by geochem alteration, look at image metrics

im_metric_df =  select_gridded_im_metrics(outcrop_data, desired_metrics=['percentile', 'rayleigh_anisotropy', 'entropy', 'glcm_contrast'], desired_scales=[1,0.5, 0.25, 0.125, 0.0625, 0.03125, 0.015625, 0.0078125, 0.00390625, 0.001953125])

point_count_df = select_gridded_point_counts(outcrop_data)

# extract the point count classes we'll focus on
pc_classes = ['Microb', 'Spar', 'Dol', 'Arch', 'Mi', 'ooid']
pc_data = point_count_df[pc_classes].to_numpy()
pc_names = point_count_df.sample_name.copy()
pc_names = pc_names.to_numpy()

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

# %% make the facies vector for 'whole rock' metrics

whole_rock_fields = ['sample_name', 'field_lithology', 'lat', 'lon', 'msl', 'anisotropy_pc1', 'anisotropy_pc2', 'anisotropy_pc3', 'contrast_pc1', 'contrast_pc2', 'contrast_pc3']

# and add in the original point count classes
for pc_class in pc_classes:
    whole_rock_fields.append(pc_class)

# and systematically add in the first 3 pcs for each of the percentile spectra
band_strs = unique_bands_percentile.astype(int).astype(str)
for i in range(3):
    for band in band_strs:
        whole_rock_fields.append(band + '_percentile_pc' + str(i+1))

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
        pc_scores_full[i,:] = pc_data[int(pc_inds[i]),:]
    else:
        pc_scores_full[i,:] = np.nan

# assemble the percentile array
percentile_scores_full = np.zeros((len(anisotropy_names), 3*len(unique_bands_percentile)))
for i in range(len(anisotropy_names)):
    if not np.isnan(percentile_inds[i]):
        percentile_scores_full[i,:] = np.reshape(percentile_scores[int(percentile_inds[i]),0:3,:], (1,3*len(unique_bands_percentile)))

# should be good to go to assemble the dataframe
whole_rock_vector = np.column_stack((anisotropy_names, field_liths, field_locs, anisotropy_scores_full, contrast_scores_full, pc_scores_full, percentile_scores_full))
whole_rock_vector = pd.DataFrame(whole_rock_vector, columns=whole_rock_fields)

# combining the dataframe turned the floats back into strings, so convert them back
data_cols = whole_rock_vector.columns[2:]
whole_rock_vector[data_cols] = whole_rock_vector[data_cols].astype(float)

# %% from the facies vector, remove the same samples that were removed from the other metrics

# first remove the samples that were removed from the other metrics
to_remove = ['smg_167', 'smg_168', 'smg_169', 'smg_170', 'smg_171', 'smg_172', 'smg_173']
to_remove_inds = np.where(np.isin(whole_rock_vector['sample_name'], to_remove))[0]
whole_rock_vector = whole_rock_vector.drop(to_remove_inds)

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

# %% give a list of indices for the carbon isoptope data that correspond to where the whole rock data is

c_to_wr_inds = []  
for i in range(len(carbon_isotopes)):
    if geochem_names[i] in whole_rock_vector['sample_name'].to_list():
        # append the index of where the sample is in the whole rock vector
        c_to_wr_inds.append(whole_rock_vector['sample_name'].to_list().index(geochem_names[i]))


# %% now crossplot any variable and the carbon isotopes, color coded by phase

var_to_plot = 'Microb'

fig, axs = plt.subplots(1,2, figsize=(10,5))

for i in range(len(corrected_carbon_isotopes)):
    # and we're only going to plot 'original' phases, so codes a, e, f, j, i, skip otherwise
    if geochem_phases[i] in ['A', 'E', 'F', 'J', 'I']:
        axs[0].scatter(corrected_carbon_isotopes[i], whole_rock_vector[var_to_plot][c_to_wr_inds[i]], color=muted_colors[new_unique_codes.tolist().index(grouped_codes[i])])

# and we'll make a second subplot, where we plot the variance in carbon isotopes vs. the variance in the variable
# recollect the points
points_to_plot = []
for i in range(len(corrected_carbon_isotopes)):
    if geochem_phases[i] in ['A', 'E', 'F', 'J', 'I']:
        points_to_plot.append([corrected_carbon_isotopes[i], whole_rock_vector[var_to_plot][c_to_wr_inds[i]]])

points_to_plot = np.array(points_to_plot)

# bin the points by the variable such that there are a certain number of points in each bin
points_in_bin = 20

# gotta sort the points by the variable
sorted_points = points_to_plot[np.argsort(points_to_plot[:,1])]
variance_list = []
bin_centers = []
for i in range(0, len(sorted_points), points_in_bin):
    bin_points = sorted_points[i:i+points_in_bin]
    variance_list.append(np.var(bin_points[:,0]))
    bin_centers.append(np.mean(bin_points[:,1]))

# smooth the variance list before plotting
variance_list = np.convolve(variance_list, np.ones(3)/3, mode='same')
axs[1].plot(variance_list, bin_centers, color='black')

# make the y axes the same
axs[1].set_ylim(axs[0].get_ylim())

plt.show()


# %% get some statistics for measurements within each unique sample

# first get the unique samples
unique_samples = np.unique(geochem_names)   

# make a list to hold lists of carbon isotopes for each unique sample, and phase
samplewise_carbon_isotopes = []
samplewise_phases = []
for sample in unique_samples:
    samplewise_carbon_isotopes.append(corrected_carbon_isotopes[geochem_names == sample])
    samplewise_phases.append(geochem_phases[geochem_names == sample])

# get the median, mean, standard deviation, min, and max for each sample
sample_medians = np.zeros(len(unique_samples))
sample_means = np.zeros(len(unique_samples))
sample_stds = np.zeros(len(unique_samples))
sample_mins = np.zeros(len(unique_samples))
sample_maxs = np.zeros(len(unique_samples))
sample_max_phase = []
sample_min_phase = []
n_phases = np.zeros(len(unique_samples))
for i in range(len(unique_samples)):
    sample_medians[i] = np.median(samplewise_carbon_isotopes[i])
    sample_means[i] = np.mean(samplewise_carbon_isotopes[i])
    sample_stds[i] = np.std(samplewise_carbon_isotopes[i])
    sample_mins[i] = np.min(samplewise_carbon_isotopes[i])
    sample_maxs[i] = np.max(samplewise_carbon_isotopes[i])

    # get the phase that corresponds to the min and max
    sample_max_phase.append(samplewise_phases[i][np.where(samplewise_carbon_isotopes[i] == sample_maxs[i])[0][0]])
    sample_min_phase.append(samplewise_phases[i][np.where(samplewise_carbon_isotopes[i] == sample_mins[i])[0][0]])

    # get the number of phases
    n_phases[i] = len(np.unique(samplewise_phases[i]))

# group the min and max phases as we did before
sample_max_phase = np.array(sample_max_phase)   
sample_min_phase = np.array(sample_min_phase)

sample_max_phase[np.isin(sample_max_phase, ['F', 'G'])] = 'F'
sample_max_phase[np.isin(sample_max_phase, ['H', 'J'])] = 'J'
sample_max_phase[np.isin(sample_max_phase, ['I', 'K'])] = 'I'

sample_min_phase[np.isin(sample_min_phase, ['F', 'G'])] = 'F'
sample_min_phase[np.isin(sample_min_phase, ['H', 'J'])] = 'J'
sample_min_phase[np.isin(sample_min_phase, ['I', 'K'])] = 'I'


# %% plot the mins vs the mins and the maxes vs the mins

fig, ax = plt.subplots(1,1, figsize=(8,3))

# scatter the maxes as filled diamonds
for i in range(len(unique_samples)):
    ax.scatter(sample_mins[i], sample_maxs[i], color=muted_colors[new_unique_codes.tolist().index(sample_max_phase[i])], marker='D')

# for the mins, we'll color code them by phase
for i in range(len(unique_samples)):
    ax.scatter(sample_mins[i], sample_mins[i], color=muted_colors[new_unique_codes.tolist().index(sample_min_phase[i])])

# and we'll also go through each pair and plot a thin gray line connecting them
for i in range(len(unique_samples)):
    ax.plot([sample_mins[i], sample_mins[i]], [sample_mins[i], sample_maxs[i]], color='gray', alpha=0.5)

# for the samples with the 5 lowest mins, we'll also plot their other samples as small squares, color coded by phase
min_inds = np.argsort(sample_mins)
for i in range(6):
    this_sample = unique_samples[min_inds[i]]
    this_sample_inds = np.where(geochem_names == this_sample)[0]
    for j in range(len(this_sample_inds)):
        ax.scatter(sample_mins[min_inds[i]], corrected_carbon_isotopes[this_sample_inds[j]], color=muted_colors[new_unique_codes.tolist().index(grouped_codes[this_sample_inds[j]])], marker='s', s=10)

ax.set_xlabel('Min corrected $\delta^{13}$C (‰)')
ax.set_ylabel('Max corrected $\delta^{13}$C (‰)')

# also print the r squared for the relationship of mins vs. maxes
r_squared = np.corrcoef(sample_mins, sample_maxs)[0,1]**2
ax.text(0.05, 0.95, 'R$^2$ = ' + str(np.round(r_squared, 2)), transform=ax.transAxes)

plt.tight_layout()
plt.show()


# %% for each phase, print out the % of times it is the max or min divided by the % that class is of the whole population

# get the total number of each phase
n_phases_total = np.zeros(len(new_unique_codes))
for i in range(len(new_unique_codes)):
    n_phases_total[i] = np.sum(np.isin(grouped_codes, new_unique_codes[i]))

# get the number of times each phase is the max and min
n_phases_max = np.zeros(len(new_unique_codes))
n_phases_min = np.zeros(len(new_unique_codes))
for i in range(len(new_unique_codes)):
    n_phases_max[i] = np.sum(np.isin(sample_max_phase, new_unique_codes[i]))
    n_phases_min[i] = np.sum(np.isin(sample_min_phase, new_unique_codes[i]))

# turn to fractions
frac_phases_max = n_phases_max/np.sum(n_phases_max)
frac_phases_min = n_phases_min/np.sum(n_phases_min)
frac_phases = n_phases_total/np.sum(n_phases_total)

# for each class print out the fraction of times it is the max divided by the fraction of times it is in the population
for i in range(len(new_unique_codes)):
    print('Phase ' + new_unique_codes[i] + ' is the max ' + str(np.round(frac_phases_max[i]/frac_phases[i], 2)) + ' times more than it is in the population')
    print('Phase ' + new_unique_codes[i] + ' is the min ' + str(np.round(frac_phases_min[i]/frac_phases[i], 2)) + ' times more than it is in the population')

# %% scatter within sample standard deviation vs. minimum

fig, ax = plt.subplots(1,1, figsize=(5,5))

ax.scatter(sample_stds, sample_mins, color='black',facecolors='none')

# and add in the correlation line with r squared
m, b = np.polyfit(sample_stds, sample_mins, 1)
x_eval = np.array([0, 1.25])
y_eval = m*x_eval + b
ax.plot(x_eval, y_eval, color='black')
r_squared = np.corrcoef(sample_stds, sample_mins)[0,1]**2
ax.text(0.05, 0.95, 'R$^2$ = ' + str(np.round(r_squared, 2)), transform=ax.transAxes)



ax.set_xlabel('Standard deviation of corrected $\delta^{13}$C (‰)')
ax.set_ylabel('Min corrected $\delta^{13}$C (‰)')
plt.tight_layout()
plt.show()

# %% scatter any of the group statistics against a whole rock metric

# make sure the names are the same, get  a vector of indices for the group stat names that correspond to the whole rock vector
gr_to_ww_inds = []
for i in range(len(unique_samples)):
    if unique_samples[i] in whole_rock_vector['sample_name'].to_list():
        gr_to_ww_inds.append(whole_rock_vector['sample_name'].to_list().index(unique_samples[i]))   

# get the whole rock metric to plot
whole_rock_metric = 'Dol'

fig, ax = plt.subplots(1,1, figsize=(5,5))

ax.scatter(whole_rock_vector[whole_rock_metric][gr_to_ww_inds], sample_stds, color='black')

ax.set_xlabel(whole_rock_metric)
ax.set_ylabel('Standard deviation of corrected $\delta^{13}$C (‰)')
plt.show()

# %% threshold based upon minimum C isotope value, exclude samples with class D 

min_c_thresh = -0.8
below_thresh_inds = np.where(sample_mins < min_c_thresh)[0]

# remove any of the below threshold samples that are class D
to_remove = []
for i in range(len(below_thresh_inds)):
    if sample_min_phase[below_thresh_inds[i]] == 'D':
        to_remove.append(i)

below_thresh_inds = np.delete(below_thresh_inds, to_remove)

# a histogram of any variables of interest for the below threshold samples, and another for the rest
var_to_hist = 'Microb'

fig, ax = plt.subplots(2,1, figsize=(5,5))

# first plot the below threshold samples
ax[0].hist(whole_rock_vector[var_to_hist][np.array(gr_to_ww_inds)[below_thresh_inds]], bins=20, color='red', alpha=0.5, label='Below threshold')
ax[0].set_title('Below threshold')
ax[0].set_xlabel(var_to_hist)
ax[0].set_ylabel('Count')

# then plot the above threshold samples
ax[1].hist(whole_rock_vector[var_to_hist][gr_to_ww_inds], bins=20, color='blue', alpha=0.5, label='Above threshold')
ax[1].set_title('Above threshold')
ax[1].set_xlabel(var_to_hist)
ax[1].set_ylabel('Count')

# use the same x limits for both
all_min = np.min(whole_rock_vector[var_to_hist][gr_to_ww_inds])
all_max = np.max(whole_rock_vector[var_to_hist][gr_to_ww_inds])
ax[0].set_xlim(all_min, all_max)
ax[1].set_xlim(all_min, all_max)


plt.tight_layout()
plt.show()

