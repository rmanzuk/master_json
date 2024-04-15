# script with sections to investigate trends in carbon isotopes
# written by R. A. Manzuk 04/15/2024
# last updated 04/15/2024

##########################################################################################
# package imports
##########################################################################################
# %%
import json # for json handling
import numpy as np # for array handling
from sklearn.decomposition import PCA # for PCA
import matplotlib # for color handling
import matplotlib.pyplot as plt # for plotting

#%%
##########################################################################################
# local function imports
##########################################################################################
# %% 
from json_processing import assemble_samples, select_gridded_geochem, select_gridded_im_metrics
from geospatial import latlong_to_utm, dip_correct_elev
from data_processing import random_sample_strat

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

# %% make a plot of carbon isotopes vs elevation, in this case make all phases gray except for micrite

fig, ax = plt.subplots(figsize=(6,10))
for phase in phase_codes:
    if phase == 'E':
        phase_inds = np.where(geochem_phases == phase)[0]
        ax.scatter(carbon_isotopes[phase_inds], z_corrected_geochem[phase_inds], label=phase_names[phase_codes.index(phase)], color=cyan)
    else:
        phase_inds = np.where(geochem_phases == phase)[0]
        ax.scatter(carbon_isotopes[phase_inds], z_corrected_geochem[phase_inds.tolist()], color='gray', alpha=0.2)    

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

micrite_carb_vals = carbon_isotopes[geochem_phases == 'E']
micrite_heights = z_corrected_geochem[geochem_phases == 'E']

micrite_random_samples, bin_centers = random_sample_strat(micrite_carb_vals, micrite_heights, height_interval, n_samplings)

# make a plot of the results
fig, ax = plt.subplots(figsize=(6,10))
# just put all carbon istopes in the background as light gray dots
ax.scatter(carbon_isotopes, z_corrected_geochem, color='lightgray', alpha=0.2)
# then plot the random samplings as cyan lines
for i in range(n_samplings):
    ax.plot(micrite_random_samples[:,i], bin_centers, color=cyan, alpha=0.4)

ax.set_xlabel('Carbon Isotopes')
ax.set_ylabel('Strat height (m)')
plt.tight_layout()
plt.show()
# %% now make a plot of all carbon isotopes vs elevation, and allow user to input a boundary to separate some samples

fig, ax = plt.subplots(figsize=(4,6))
ax.scatter(carbon_isotopes, z_corrected_geochem, color='lightgray', alpha=0.2)

# make a line to separate the samples
print('Click to place a separator line')
separator = plt.ginput(-1)

separator = np.array(separator)

# now we need to assign a separator value to each height in the z_corrected array, interpolating between the separator point above and below
separator_values = np.zeros(len(z_corrected_geochem))
for i in range(len(z_corrected_geochem)):
    # find the two separator points that this height is between
    above = np.where(separator[:,1] > z_corrected_geochem[i])[0]
    below = np.where(separator[:,1] < z_corrected_geochem[i])[0]
    if len(above) > 0 and len(below) > 0:
        above = above[-1]
        below = below[0]
        # interpolate between the two points
        separator_values[i] = separator[below,0] + (separator[above,0] - separator[below,0])*((z_corrected_geochem[i] - separator[below,1])/(separator[above,1] - separator[below,1]))
    elif len(above) == 0:
        below = below[0]
        separator_values[i] = separator[below,0]
    elif len(below) == 0:
        above = above[-1]
        separator_values[i] = separator[above,0]

# and then separate
is_below = carbon_isotopes < separator_values

# make a plot of the results
fig, ax = plt.subplots(figsize=(6,10))
ax.scatter(carbon_isotopes[is_below], z_corrected_geochem[is_below], color=rose, alpha=0.5, label='Below')
ax.scatter(carbon_isotopes[~is_below], z_corrected_geochem[~is_below], color='gray', alpha=0.2, label='Above')

ax.set_xlabel('Carbon Isotopes')
ax.set_ylabel('Strat height (m)')
ax.legend()
plt.tight_layout()
plt.show()

# %% make a correction factor for carbon isotopes based upon the correlation between carbon isotopes and geochem_pc2

# first get the correlation coefficient
correlation = np.corrcoef(geochem_scores[:,1], carbon_isotopes)

# get the equation of the line
m = correlation[0,1]*np.std(carbon_isotopes)/np.std(geochem_scores[:,1])
b = np.mean(carbon_isotopes) - m*np.mean(geochem_scores[:,1])

# now apply the correction
correction_factor = m*geochem_scores[:,1] + b
corrected_carbon_isotopes = carbon_isotopes - correction_factor

# %% replot the carbon isotopes vs elevation, but with the corrected values, still color coded by if they are below the separator

fig, ax = plt.subplots(figsize=(6,10))

ax.scatter(corrected_carbon_isotopes[is_below], z_corrected_geochem[is_below], color=rose, alpha=0.5, label='Below')
ax.scatter(corrected_carbon_isotopes[~is_below], z_corrected_geochem[~is_below], color='gray', alpha=0.2, label='Above')

ax.set_xlabel('Corrected Carbon Isotopes')
ax.set_ylabel('Strat height (m)')
ax.legend()
plt.tight_layout()
plt.show()

# %% replot the carbon isotopes vs elevation, but with the corrected values, gray if they are above, color coded by phase if below

fig, ax = plt.subplots(figsize=(8,10))

symbols = ['o', 's', 'D', 'v', '^', '<', '>', 'p', 'P', '*', 'X']
n_colors = len(plt.rcParams['axes.prop_cycle'].by_key()['color'])

ax.scatter(corrected_carbon_isotopes[~is_below], z_corrected_geochem[~is_below], color='gray', alpha=0.2)

count = 0
for phase in phase_codes:
    if count < n_colors:
        phase_inds = np.where(geochem_phases == phase)[0]
        below_inds = np.where(is_below & (geochem_phases == phase))[0]
        ax.scatter(corrected_carbon_isotopes[below_inds], z_corrected_geochem[below_inds], label=phase_names[phase_codes.index(phase)], color=plt.rcParams['axes.prop_cycle'].by_key()['color'][count], marker=symbols[0])

    else:
        phase_inds = np.where(geochem_phases == phase)[0]
        below_inds = np.where(is_below & (geochem_phases == phase))[0]
        ax.scatter(corrected_carbon_isotopes[below_inds], z_corrected_geochem[below_inds], label=phase_names[phase_codes.index(phase)], color=plt.rcParams['axes.prop_cycle'].by_key()['color'][count-n_colors], marker=symbols[1])
    
    count += 1

ax.set_xlabel('Corrected Carbon Isotopes')
ax.set_ylabel('Strat height (m)')
# place the legend outside the plot
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

# %% because the low population of carbon isotopes isn't corrected for by geochem alteration, look at anisotropy

im_metric_df =  select_gridded_im_metrics(outcrop_data, desired_metrics=['rayleigh_anisotropy'], desired_scales=[1,0.5, 0.25, 0.125, 0.0625, 0.03125, 0.015625, 0.0078125, 0.00390625, 0.001953125])

# do PCA to get anisotropy components
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

# %% use sample names to find the anisotropy indices that correspond to the carbon isotopes
      
c_ind_anisotropy = []
for sample in geochem_names:
    if sample in anisotropy_names:
        c_ind_anisotropy.append(np.where(np.array(anisotropy_names) == sample)[0][0])
    else:
        c_ind_anisotropy.append(None)

# %% make a plot of carbon isotopes vs an anisotropy pc, if the point is below the separator, color by phase, otherwise gray
        
# pick an anisotropy pc to use
anisotropy_pc = 3 # 1 indexed

# only going to do this for certain phases
phases = ['E', 'F', 'G', 'I']
phase_colors = [cyan, wine, teal, purple]

fig, ax = plt.subplots(figsize=(7,4))
#ax.scatter(entire_vector[x_var][entire_vector.phase == phase], entire_vector[y_var][entire_vector.phase == phase], color='lightgray', alpha=0.2)
for phase in phases:
    below_inds = np.where(is_below & (geochem_phases == phase))[0]
    above_inds = np.where(~is_below & (geochem_phases == phase))[0]
    # make c_ind_anisotropy a numpy array so we can index it
    c_ind_anisotropy = np.array(c_ind_anisotropy)
    ax.scatter(carbon_isotopes[below_inds], anisotropy_scores[c_ind_anisotropy[below_inds], anisotropy_pc-1, 0], color=phase_colors[phases.index(phase)], label=phase_names[phase_codes.index(phase)])
    ax.scatter(carbon_isotopes[above_inds], anisotropy_scores[c_ind_anisotropy[above_inds], anisotropy_pc-1, 0], color='gray', alpha=0.2)

ax.set_xlabel(r'$\delta^{13}$C (‰)')
ax.set_ylabel('Anisotropy PC' + str(anisotropy_pc))
# set y limits -4 to 4
ax.set_ylim(-4,4)
# put the legend outside the plot
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()
