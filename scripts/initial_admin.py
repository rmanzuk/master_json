# script to perform initial admin tasks on an outcrop json and sample jsons
# written by R. A. Manzuk 02/14/2024
# last updated 02/19/2024

##########################################################################################
# package imports
##########################################################################################
# %%

import json # for json handling
#%%
##########################################################################################
# local function imports
##########################################################################################
# %% 
from json_processing import assemble_samples, data_audit
from custom_plotting import display_point_counts
# %% 
##########################################################################################
# script lines
##########################################################################################
# %% define paths, and read in the outcrop json

outcrop_json_file = '/Users/ryan/Dropbox (Princeton)/code/master_json/stewarts_mill.json'
sample_json_dir = '/Users/ryan/Dropbox (Princeton)/code/master_json/stewarts_mill_grid_samples/'

# load in the outcrop json
with open(outcrop_json_file, 'r') as f:
    outcrop_data = json.load(f)

# %% assemble the missing samples into the outcrop json

outcrop_data = assemble_samples(outcrop_data, sample_json_dir, data_type=['grid_data'], data_name=["Stewart's Mill Grid"])

# and audit the data, but just the grid samples
data_audit(outcrop_data, 'grid_data')

# %% for a particular sample, display the point counts, and certain points to highlight

sample_set = "Stewart's Mill Grid"
sample_name = 'smg_100'
star_points = [195]

# put together a list of all sample names
all_sample_sets = [outcrop_data['grid_data'][i]['name'] for i in range(len(outcrop_data['grid_data']))]

# check if the sample set is in the list
if sample_set in all_sample_sets:
    # get the index of the sample set
    sample_set_index = all_sample_sets.index(sample_set)
    # get a list of the sample names
    sample_names = [outcrop_data['grid_data'][sample_set_index]['samples'][i]['sample_name'] for i in range(len(outcrop_data['grid_data'][sample_set_index]['samples']))]
    # get the sample
    sample_index = sample_names.index(sample_name)
    sample = outcrop_data['grid_data'][sample_set_index]['samples'][sample_index]
    # display the point counts
    display_point_counts(sample, star_points=star_points)

