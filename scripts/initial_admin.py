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
from json_processing import assemble_samples, select_gridded_geochem, select_gridded_im_metrics, select_gridded_point_counts
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

# %% select the gridded geochem data, im metrics, and point counts
geochem_df = select_gridded_geochem(outcrop_data)

im_metric_df =  select_gridded_im_metrics(outcrop_data)

point_count_df = select_gridded_point_counts(outcrop_data)