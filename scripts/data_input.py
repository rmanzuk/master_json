# script with sections to handle all little data input/updata tasks for jsons
# written by R. A. Manzuk 02/16/2024
# last updated 02/16/2024

##########################################################################################
# package imports
##########################################################################################
# %%
import os # for file handling
import json # for json handling

#%%
##########################################################################################
# local function imports
##########################################################################################
# %% 
from file_handing import get_file_list
from json_processing import get_point_count_fracs, give_geochem_ids
# %% 
##########################################################################################
# script lines
##########################################################################################
# %% section to turn point counts into fractions and store in sample jsons

# get a list of all the json files
input_json_dir = '/Users/ryan/Dropbox (Princeton)/code/master_json/stewarts_mill_grid_samples/'
json_files = get_file_list(input_json_dir, '*.json')

# output directory for the json files (can be the same as the input)
output_json_dir = input_json_dir

# iterate through the json files
for json_file in json_files:

    # load in the json file
    with open(json_file, 'r') as f:

        original_data = json.load(f)

        # make a copy of the data
        data_copy = original_data.copy()

        # check that it has point counts and not point count fractions
        if 'point_counts' in data_copy.keys() and 'point_count_fracs' not in data_copy.keys():

            # then run functoin to convert to fractions
            point_count_fracs = get_point_count_fracs(data_copy['point_counts'])

            # put the point count fractions in the json, 
            data_copy['point_count_fracs'] = point_count_fracs

            # write the data_copy to a new json file
            out_file = os.path.join(output_json_dir,json_file.split('/')[-1])
            with open(out_file, 'w') as outfile:
                json.dump(data_copy, outfile,indent=4)


# %% section for point count fraction of samples within the outcrop json file
                
outcrop_json = '/Users/ryan/Dropbox (Princeton)/code/master_json/stewarts_mill.json'

# load in the json file
with open(outcrop_json, 'r') as f:

    original_data = json.load(f)

    # make a copy of the data
    data_copy = original_data.copy()

    # iterate through all strat_data entries and the samples within
    for strat_data in data_copy['strat_data']:
        for sample in strat_data['samples']:
            if 'point_counts' in sample.keys() and len(sample['point_counts']) > 0 and 'point_count_fracs' not in sample.keys():
                sample['point_count_fracs'] = get_point_count_fracs(sample['point_counts'])

    # and iterate through grid_data
    for grid_data in data_copy['grid_data']:
        for sample in grid_data['samples']:
            if 'point_counts' in sample.keys() and len(sample['point_counts']) > 0 and 'point_count_fracs' not in sample.keys():
                sample['point_count_fracs'] = get_point_count_fracs(sample['point_counts'])


# write the data_copy to a new json file
out_file = os.path.join(outcrop_json.split('/')[-1])
with open(out_file, 'w') as outfile:
    json.dump(data_copy, outfile,indent=4)

