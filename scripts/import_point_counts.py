# script to update all point count data from the csvs to the jsons
# written by R. A. Manzuk 02/14/2024
# last updated 02/14/2024

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
from file_handing import jmicro_to_dict, get_file_list
from json_processing import get_point_count_fracs
# %% 
##########################################################################################
# script lines
##########################################################################################
# %% define paths

csv_dir = '/Users/ryan/Dropbox (Princeton)/reef_survey_project/nevada/data_spreadsheets/point_counts/smg_excel_exports/'
input_json_dir = '/Users/ryan/Dropbox (Princeton)/code/master_json/stewarts_mill_grid_samples/'

# and an output directory for the json files (can be the same as the input)
output_json_dir = input_json_dir

# make the output directory if it doesn't exist
if not os.path.exists(output_json_dir):
    os.makedirs(output_json_dir)

# %% iterate through the csvs and update the jsons

# get a list of all the csv files
csv_files = get_file_list(csv_dir, '*.csv')

# iterate through the csv files
for csv_file in csv_files:

    # get the sample number
    sample_num = csv_file.split('/')[-1][:-4]

    # get the corresponding json file
    json_file = get_file_list(input_json_dir, sample_num+'.json')

    # only proceed if there is a match
    if len(json_file) > 0:

        # convert the csv to a dictionary
        csv_dict = jmicro_to_dict(csv_file, 14204*0.35277778, 10652*0.35277778, delimiter=';')

        # load in the json file
        with open(json_file[0], 'r') as f:

            original_data = json.load(f)

            # make a copy of the data
            data_copy = original_data.copy()

            # clear the point_counts field, replacing it with the csv_dict
            data_copy['point_counts'] = csv_dict

        # write the data_copy to a new json file
        out_file = os.path.join(output_json_dir,json_file[0].split('/')[-1])

        with open(out_file, 'w') as outfile:
            json.dump(data_copy, outfile,indent=4)

# %% also going to put some lines here to turn point counts into fractions and store in jsons

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


# %% lines for point count fraction of samples within the outcrop json file
                
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
