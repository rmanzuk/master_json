# set of functions for working with json files for analysis
# written by R. A. Manzuk 12/15/2023
# last updated 12/15/2023

##########################################################################################
# package imports
##########################################################################################
import numpy as np # for array handling
import pdb # for debugging
import json # for json handling
import pandas as pd # for dataframes
##########################################################################################
# function definitions
##########################################################################################

# ----------------------------------------------------------------------------------------
# get the indices of where an item is in a list of strings
def string_find(solo_string, string_list):
    """
    a function to get the indices of where an item is in a list of strings

    Keyword arguments:
    solo_string -- the string to search for
    string_list -- the list of strings to search in

    Returns:
    match_inds -- a list of the indices of the matches
    """
    # set up empty match list and counter for going through the list
    match_inds = []
    i = 0

    # loop through and add the index if a match
    while i < len(string_list):
        if solo_string == string_list[i]:
            match_inds.append(i)
        i += 1

    # return the final match indices
    return match_inds

# ----------------------------------------------------------------------------------------
def get_lithology(strat_data, strat_cols):
    """
    A function to get the lithology of each sample in a stratigraphic column, based on
    the likeness code of the column and the likeness codes of the samples. This function 
    searches the column and applies the lithology of the bed the sample falls within.

    Keyword arguments:
    strat_data -- a dictionary containing stratigraphic samples
    strat_cols -- a list of dictionaries containing stratigraphic columns

    Returns:
    None, but it updates the strat_data dictionary with the field_lithology key
    """
    # start by looking for the likeness code
    this_likeness = strat_data['likeness_code']

    col_likenesses = []
    for col in strat_cols:
        col_likenesses.append(col['likeness_code'])

    # find the column with the same likeness code
    match_ind = string_find(this_likeness,col_likenesses)

    # only proceed if we got a match
    if len(match_ind) == 1:
        # get the column
        this_col = strat_cols[match_ind[0]]

        # take out the bed heights so we can get cumulative thicknesses
        thickness_list = [sub['thickness'] for sub in this_col['beds']]
        litho_list = [sub['lithology'] for sub in this_col['beds']]
        cu_thicknesses = np.cumsum(thickness_list)

        # go through the sample heights and match to a lithology
        for samp in strat_data['samples']:
            # get the height of the sample
            samp_height = samp['strat_height']

            # find the index of the first thickness that is greater than the sample height
            match_ind = np.where(cu_thicknesses > samp_height)[0][0]

            # get the lithology
            samp['field_lithology'] = litho_list[match_ind]
    else:
        print('No match found for likeness code ' + this_likeness + ' in stratigraphic columns')

# ----------------------------------------------------------------------------------------
def unique_likenesses(input_json):
    """
    A function to get all of the unique likeness codes in a json file

    Keyword arguments:
    input_json -- the json file to search, loaded as a dict

    Returns:
    all_likeness -- a list of all the unique likeness codes in the json file
    """
    # set up an empty list to catch all the likeness codes
    all_likeness = []
    # loop through all strat columns
    for iter in input_json['strat_columns']:
        # check if we already have this likeness code
        if iter['likeness_code'] not in all_likeness:
            # if not, add it to the list
            all_likeness.append(iter['likeness_code'])

    # do the same for strat data and grid data
    for iter in input_json['strat_data']:
        if iter['likeness_code'] not in all_likeness:
            all_likeness.append(iter['likeness_code'])

    for iter in input_json['grid_data']:
        if iter['likeness_code'] not in all_likeness:
            all_likeness.append(iter['likeness_code'])

    # just return the list at the end
    return all_likeness

# ----------------------------------------------------------------------------------------
def assemble_samples(outcrop_json, sample_json_dir, data_type=None, data_name=None):
    """
    A function to fill in missing data from an outcrop json by reading in the sample jsons
    and adding the data to the outcrop json

    Keyword arguments:
    outcrop_json -- the json file to fill in, as a dictionary
    sample_json_dir -- the directory containing the sample jsons
    data_type -- list of datatypes to fill in, if None, fill in all
    data_name -- list of data names to fill in, if None, fill in all

    Returns:
    outcrop_json -- the updated outcrop json
    """

    # if we didn't specify a data type, fill in all
    if data_type is None:
        data_type = ['strat_columns', 'strat_data', 'grid_data']

    # loop through all the data types
    for data in data_type:
        # if we didn't specify a data name, fill in all
        if data_name is None:
            data_name = [sub['name'] for sub in outcrop_json[data]]

        # store all the data names so we can choose their inds later
        all_data_names = [sub['name'] for sub in outcrop_json[data]]

        # loop through all the data names
        for name in data_name:
    
            # get the index of the data we want to fill in
            data_ind = string_find(name,all_data_names)[0]

            # loop through all the samples, and add the data to the outcrop json
            for samp in outcrop_json[data][data_ind]['samples']:
                # get the keys and check if "sample_json_file" is in them
                samp_keys = samp.keys()
                
                if 'sample_json_file' in samp_keys:
                    # if it is, read in the sample json
                    with open(sample_json_dir + samp['sample_json_file'], 'r') as f:
                        sample_data = json.load(f)
                    
                    # and replace the sample with the new data
                    samp_ind = outcrop_json[data][data_ind]['samples'].index(samp)
                    outcrop_json[data][data_ind]['samples'][samp_ind] = sample_data

    # return the updated outcrop json
    return outcrop_json

# ----------------------------------------------------------------------------------------
def get_point_count_fracs(point_counts_list):
    """
    function to get the fraction of all points counted that are of each class, returning
    a list of dictionaries with the fractions

    Keyword arguments:
    point_counts_list -- a list of dictionaries containing point count data, could be either
    individual observations with locations or bulk counts

    Returns:
    point_count_fracs -- a list of dictionaries containing the fractions of each point count
    class
    """

    # to start, we need to knwo which cases we are dealing with, so we will check the first
    # entry in the list to see if it has x_position as one of its keys
    if 'x_position' in point_counts_list[0].keys():
        # if it does, we have individual observations with locations
        # set up a list to catch the counts
        point_count_fracs = []

        # get the total number of points counted
        total_points = len(point_counts_list)

        # get the unique classes
        unique_classes = list(set([sub['class'] for sub in point_counts_list]))

        # loop through the unique classes and get the fraction
        for u_class in unique_classes:
            # get the count of the class
            class_count = len([sub for sub in point_counts_list if sub['class'] == u_class])

            # get the fraction
            class_frac = class_count/total_points

            # add to the dictionary
            point_count_fracs.append({'class':u_class, 'unit': 'fraction_points', 'value': class_frac})

    # for the other case check if unit is in the keys, and if the entry for unit is 'points_counted'
    elif 'unit' in point_counts_list[0].keys() and point_counts_list[0]['unit'] == 'points_counted':
        # if it doesn't, we have bulk counts
        # set up a list to catch the counts
        point_count_fracs = []

        # get the total number of points counted
        total_points = sum([sub['value'] for sub in point_counts_list])

        # get the unique classes
        unique_classes = list(set([sub['class'] for sub in point_counts_list]))

        # loop through the unique classes and get the fraction
        for u_class in unique_classes:
            # get the count of the class
            class_count = sum([sub['value'] for sub in point_counts_list if sub['class'] == u_class])

            # get the fraction
            class_frac = class_count/total_points

            # add to the dictionary
            point_count_fracs.append({'class':u_class, 'unit': 'fraction_points', 'value': class_frac})

    # return the list of dictionaries
    return point_count_fracs


# ----------------------------------------------------------------------------------------
def select_geospatial_data(outcrop_json, desired_metrics=None):
    """
    A function to go through an outcrop json and select only data with gps coordinates, 
    returning them in a datframe

    Keyword arguments:
    outcrop_json -- the json file to fill in, as a dictionary
    desired_metrics -- a list of metrics to return, if None, the function will ask for input

    Returns:
    out_df -- a dataframe containing the desired metrics, with GPS coordinates
    """

    # start by asking for input if we didn't get a list of metrics
    if desired_metrics is None:
        
        # we will scrape through the json to find all the metrics to list them
        all_metrics = []

        # loop through all the strat data 
        for strat in outcrop_json['strat_data']:
            for samp in strat['samples']:
                # go into the geochem_measurements and add all the metrics to the list
                for meas in samp['geochem_measurements']:
                    if meas['measurement_name'] not in all_metrics:
                        all_metrics.append(meas['measurement_name'])

                # do the same for point count percents
                # WRITE THIS PART ONCE POINT COUNT PERCENTS ARE IN THE JSON
                # for meas in samp['point_count_percent']:
        
        # loop through all the grid data
        for grid in outcrop_json['grid_data']:
            for samp in grid['samples']:
                # go into the geochem_measurements and add all the metrics to the list
                for meas in samp['geochem_measurements']:
                    if meas['measurement_name'] not in all_metrics:
                        all_metrics.append(meas['measurement_name'])

                # do the same for point count percents
                # WRITE THIS PART ONCE POINT COUNT PERCENTS ARE IN THE JSON
                # for meas in samp['point_count_percent']:
                        
        # ask for input
        print('The available metrics are:')

        # print each metric on a new line with an index
        for i in range(len(all_metrics)):
            print(str(i) + ': ' + all_metrics[i])

        # get the input
        desired_metrics = input('Please enter the indices of the metrics you would like to select, separated by commas: ')

        # split the input into a list of indices
        desired_metrics = desired_metrics.split(',')

        # convert the indices to integers
        desired_metrics = [int(sub) for sub in desired_metrics]

        # use the indices to get the desired metrics
        desired_metrics = [all_metrics[sub] for sub in desired_metrics]

    # set up an empty dataframe to catch the data
    out_df = pd.DataFrame()

    # fields will be sample name, latitude, longitude, msl, phase, and the desired metrics
    out_df['sample_name'] = []
    out_df['latitude'] = []
    out_df['longitude'] = []
    out_df['msl'] = []
    out_df['phase'] = []
    for metric in desired_metrics:
        out_df[metric] = []

    # now pretty much same loops as before, but adding to the dataframe
    # loop through all the strat data
    for strat in outcrop_json['strat_data']:
        for samp in strat['samples']:

            # if the sample has gps coordinates
            if 'latitude' in samp.keys():

                # only proceed if the sample has at least one of the desired metrics
                if len([sub for sub in samp['geochem_measurements'] if sub['measurement_name'] in desired_metrics]) > 0:
                    
                    # now we need to determine if the any of the geochem measeaurements have image locations
                    # these will serve as unique identifiers for the sub-samples. Otherwise, phase will be used
                    # set up a vector to catch measurements with image locations
                    image_locations = [sub for sub in samp['geochem_measurements'] if 'image_location' in sub.keys()]

                    # and those without
                    no_image_locations = [sub for sub in samp['geochem_measurements'] if 'image_location' not in sub.keys()]

                    # for those with image locations, get all the image locations so we can make a unique list
                    if len(image_locations) > 0:
                        all_image_locations = []
                        for meas in image_locations:
                            all_image_locations.append(meas['im_loc'][0])

                        # make a unique list
                        unique_image_locations = list(set(all_image_locations))
