# set of functions for working with json files for analysis
# written by R. A. Manzuk 12/15/2023
# last updated 12/19/2023

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
def select_gridded_geochem(outcrop_json, desired_metrics=None):
    """
    A function to go through an outcrop json and select only geochem data with gps coordinates, 
    returning them in a dataframe

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
                # only continue with sample if it has geochem measurements
                if 'geochem_measurements' in samp.keys() and len(samp['geochem_measurements']) > 0:
                    # if there's another list nested in the geochem measurements, we need to go deeper
                    if type(samp['geochem_measurements'][0]) == list:
                        for sub_meas in samp['geochem_measurements']:
                            for meas in sub_meas:
                                if meas['measurement_name'] not in all_metrics:
                                    all_metrics.append(meas['measurement_name'])
                    elif type(samp['geochem_measurements'][0]) == dict:
                        for meas in samp['geochem_measurements']:
                            if meas['measurement_name'] not in all_metrics:
                                all_metrics.append(meas['measurement_name'])
        
        # loop through all the grid data
        for grid in outcrop_json['grid_data']:
            for samp in grid['samples']:
                # only continue with sample if it has geochem measurements
                if 'geochem_measurements' in samp.keys() and len(samp['geochem_measurements']) > 0:
                    # if there's another list nested in the geochem measurements, we need to go deeper
                    if type(samp['geochem_measurements'][0]) == list:
                        for sub_meas in samp['geochem_measurements']:
                            for meas in sub_meas:
                                if meas['measurement_name'] not in all_metrics:
                                    all_metrics.append(meas['measurement_name'])
                    elif type(samp['geochem_measurements'][0]) == dict:
                        for meas in samp['geochem_measurements']:
                            if meas['measurement_name'] not in all_metrics:
                                all_metrics.append(meas['measurement_name'])
                        
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
    out_df['im_loc_y'] = []
    out_df['im_loc_x'] = []
    for metric in desired_metrics:
        out_df[metric] = []

    # now pretty much same loops as before, but adding to the dataframe
    # loop through all the strat data
    for strat in outcrop_json['strat_data']:
        for samp in strat['samples']:

            # if the sample has gps coordinates and geochem measurements
            if 'latitude' in samp.keys() and 'geochem_measurements' in samp.keys() and len(samp['geochem_measurements']) > 0:

                # again need to check if there's another list nested in the geochem measurements
                if type(samp['geochem_measurements'][0]) == list:
                    for sub_meas in samp['geochem_measurements']:
                        # make a new dataframe to hold the data
                        new_df = pd.DataFrame({'sample_name': samp['sample_name'], 'latitude': samp['latitude'], 'longitude': samp['longitude'], 'msl': samp['msl'], 'phase': sub_meas[0]['phase']}, index=[0])
                        # if the sample has image locations, add them to the dataframe
                        if 'im_loc' in sub_meas[0].keys() and len(sub_meas[0]['im_loc']) > 0:
                            new_df['im_loc_y'] = sub_meas[0]['im_loc'][0]
                            new_df['im_loc_x'] = sub_meas[0]['im_loc'][1]
                        for meas in sub_meas:
                            if meas['measurement_name'] in desired_metrics:
                                # add the measurement to the dataframe
                                new_df[meas['measurement_name']] = meas['value']
                        # concatenate the new dataframe to the output dataframe        
                        out_df = pd.concat([out_df, new_df], ignore_index=True)
                elif type(samp['geochem_measurements'][0]) == dict:
                    # make a new dataframe to hold the data
                    new_df = pd.DataFrame({'sample_name': samp['sample_name'], 'latitude': samp['latitude'], 'longitude': samp['longitude'], 'msl': samp['msl'], 'phase': samp['geochem_measurements'][0]['phase']}, index=[0])
                    # if the sample has image locations, add them to the dataframe
                    if 'im_loc' in samp['geochem_measurements'][0].keys() and len(samp['geochem_measurements'][0]['im_loc']) > 0:
                        new_df['im_loc_y'] = samp['geochem_measurements'][0]['im_loc'][0]
                        new_df['im_loc_x'] = samp['geochem_measurements'][0]['im_loc'][1]
                    for meas in samp['geochem_measurements']:
                        if meas['measurement_name'] in desired_metrics:
                            # add the measurement to the dataframe
                            new_df[meas['measurement_name']] = meas['value']
                    # concatenate the new dataframe to the output dataframe
                    out_df = pd.concat([out_df, new_df], ignore_index=True)
    # loop through all the grid data
    for grid in outcrop_json['grid_data']:
        for samp in grid['samples']:

            # if the sample has gps coordinates and geochem measurements
            if 'latitude' in samp.keys() and 'geochem_measurements' in samp.keys() and len(samp['geochem_measurements']) > 0:

                # again need to check if there's another list nested in the geochem measurements
                if type(samp['geochem_measurements'][0]) == list:
                    for sub_meas in samp['geochem_measurements']:
                        # make a new dataframe to hold the data
                        new_df = pd.DataFrame({'sample_name': samp['sample_name'], 'latitude': samp['latitude'], 'longitude': samp['longitude'], 'msl': samp['msl'], 'phase': sub_meas[0]['phase']}, index=[0])
                        # if the sample has image locations, add them to the dataframe
                        if 'im_loc' in sub_meas[0].keys() and len(sub_meas[0]['im_loc']) > 0:
                            new_df['im_loc_y'] = sub_meas[0]['im_loc'][0]
                            new_df['im_loc_x'] = sub_meas[0]['im_loc'][1]
                        for meas in sub_meas:
                            if meas['measurement_name'] in desired_metrics:
                                # add the measurement to the dataframe
                                new_df[meas['measurement_name']] = meas['value']
                        # concatenate the new dataframe to the output dataframe        
                        out_df = pd.concat([out_df, new_df], ignore_index=True)
                elif type(samp['geochem_measurements'][0]) == dict:
                    # make a new dataframe to hold the data
                    new_df = pd.DataFrame({'sample_name': samp['sample_name'], 'latitude': samp['latitude'], 'longitude': samp['longitude'], 'msl': samp['msl'], 'phase': samp['geochem_measurements'][0]['phase']}, index=[0])
                    # if the sample has image locations, add them to the dataframe
                    if 'im_loc' in samp['geochem_measurements'][0].keys() and len(samp['geochem_measurements'][0]['im_loc']) > 0:
                        new_df['im_loc_y'] = samp['geochem_measurements'][0]['im_loc'][0]
                        new_df['im_loc_x'] = samp['geochem_measurements'][0]['im_loc'][1]
                    for meas in samp['geochem_measurements']:
                        if meas['measurement_name'] in desired_metrics:
                            # add the measurement to the dataframe
                            new_df[meas['measurement_name']] = meas['value']
                    # concatenate the new dataframe to the output dataframe
                    out_df = pd.concat([out_df, new_df], ignore_index=True)
    # return the dataframe
    return out_df
                    
# ----------------------------------------------------------------------------------------
def select_gridded_point_counts(outcrop_json):
    """
    A function to go through an outcrop json and select only point count percentages with gps coordinates,
    returning them in a dataframe
    
    Keyword arguments:
    outcrop_json -- the json file to fill in, as a dictionary
    
    Returns:
    out_df -- a dataframe containing the point count data
    """

    # first, we need to know all the classes available to input
    all_classes = []

    # find them by looping through all samples, and if they have gps coordinates and point count fractions, 
    # add new classes to the list
    for strat in outcrop_json['strat_data']:
        for samp in strat['samples']:
            if 'latitude' in samp.keys() and 'point_count_fracs' in samp.keys():
                for frac in samp['point_count_fracs']:
                    if frac['class'] not in all_classes:
                        all_classes.append(frac['class'])

    for grid in outcrop_json['grid_data']:
        for samp in grid['samples']:
            if 'latitude' in samp.keys() and 'point_count_fracs' in samp.keys():
                for frac in samp['point_count_fracs']:
                    if frac['class'] not in all_classes:
                        all_classes.append(frac['class'])

    # set up an empty dataframe to catch the data with classes sample name, latitude, longitude, msl, and the classes
    out_df = pd.DataFrame()
    out_df['sample_name'] = []
    out_df['latitude'] = []
    out_df['longitude'] = []
    out_df['msl'] = []
    for class_name in all_classes:
        out_df[class_name] = []

    # now loop through all the strat data and grid data, and add the data to the dataframe 
    for strat in outcrop_json['strat_data']:
        for samp in strat['samples']:
            if 'latitude' in samp.keys() and 'point_count_fracs' in samp.keys() and len(samp['point_count_fracs']) > 0:
                new_row = {'sample_name': samp['sample_name'], 'latitude': samp['latitude'], 'longitude': samp['longitude'], 'msl': samp['msl']}
                for frac in samp['point_count_fracs']:
                    new_row[frac['class']] = frac['value']
                new_df = pd.DataFrame(new_row, index=[0])
                out_df = pd.concat([out_df, new_df], ignore_index=True)

    for grid in outcrop_json['grid_data']:
        for samp in grid['samples']:
            if 'latitude' in samp.keys() and 'point_count_fracs' in samp.keys() and len(samp['point_count_fracs']) > 0:
                new_row = {'sample_name': samp['sample_name'], 'latitude': samp['latitude'], 'longitude': samp['longitude'], 'msl': samp['msl']}
                for frac in samp['point_count_fracs']:
                    new_row[frac['class']] = frac['value']
                new_df = pd.DataFrame(new_row, index=[0])
                out_df = pd.concat([out_df, new_df], ignore_index=True)
                
    # return the dataframe
    return out_df

# ----------------------------------------------------------------------------------------
def select_gridded_pa(outcrop_json):
    """
    A function to go through an outcrop json and select presence/absence data with gps coordinates,
    returning them in a dataframe.

    Keyword arguments:
    outcrop_json -- the json file to fill in, as a dictionary

    Returns:
    out_df -- a dataframe containing the presence/absence data
    """

    # first we need to know all the classes available to input
    all_classes = []

    # find them by looping through all samples, and if they have gps coordinates and presence/absence data,
    # add new classes to the list
    for strat in outcrop_json['strat_data']:
        for samp in strat['samples']:
            if 'latitude' in samp.keys() and 'presence_absence' in samp.keys():
                for pa in samp['presence_absence'].keys():
                    if pa not in all_classes:
                        all_classes.append(pa)

    for grid in outcrop_json['grid_data']:
        for samp in grid['samples']:
            if 'latitude' in samp.keys() and 'presence_absence' in samp.keys():
                for pa in samp['presence_absence'].keys():
                    if pa not in all_classes:
                        all_classes.append(pa)

    # set up an empty dataframe to catch the data with classes sample name, latitude, longitude, msl, and the classes
    out_df = pd.DataFrame()
    out_df['sample_name'] = []
    out_df['latitude'] = []
    out_df['longitude'] = []
    out_df['msl'] = []
    for class_name in all_classes:
        out_df[class_name] = []

    # now loop through all the strat data and grid data, and add the data to the dataframe
    for strat in outcrop_json['strat_data']:
        for samp in strat['samples']:
            if 'latitude' in samp.keys() and 'presence_absence' in samp.keys():
                new_row = {'sample_name': samp['sample_name'], 'latitude': samp['latitude'], 'longitude': samp['longitude'], 'msl': samp['msl']}
                for pa in samp['presence_absence'].keys():
                    new_row[pa] = samp['presence_absence'][pa]
                # if any of the classes are missing, add them with a value of False
                for cl in all_classes:
                    if cl not in new_row.keys():
                        new_row[cl] = False

                new_df = pd.DataFrame(new_row, index=[0])
                out_df = pd.concat([out_df, new_df], ignore_index=True)

    for grid in outcrop_json['grid_data']:
        for samp in grid['samples']:
            if 'latitude' in samp.keys() and 'presence_absence' in samp.keys():
                new_row = {'sample_name': samp['sample_name'], 'latitude': samp['latitude'], 'longitude': samp['longitude'], 'msl': samp['msl']}
                for pa in samp['presence_absence'].keys():
                    new_row[pa] = samp['presence_absence'][pa]
                # if any of the classes are missing, add them with a value of False
                for cl in all_classes:
                    if cl not in new_row.keys():
                        new_row[cl] = False

                new_df = pd.DataFrame(new_row, index=[0])
                out_df = pd.concat([out_df, new_df], ignore_index=True)

    # return the dataframe
    return out_df

# ----------------------------------------------------------------------------------------
def select_gridded_im_metrics(outcrop_json, desired_metrics=None, desired_scales=None):
    """
    A function to go through an outcrop json and select only image metrics with gps coordinates,
    returning them in a datframe
    
    Keyword arguments:
    outcrop_json -- the json file to fill in, as a dictionary
    
    Returns:
    out_df -- a dataframe containing the image metrics
    """

    # if metrics and scales are not specified, we will ask for input
    if desired_metrics is None:
        # first, we need to know all the metrics available to input, and all the scales
        all_metrics = []

        # find them by looping through all samples, and if they have gps coordinates and image metrics, 
        # add new metrics and scales to the lists
        for strat in outcrop_json['strat_data']:
            for samp in strat['samples']:
                if 'latitude' in samp.keys() and 'images' in samp.keys() and len(samp['images']) > 0:
                    for im in samp['images']:
                        if 'metrics' in im.keys() and len(im['metrics']) > 0:
                            for metric in im['metrics']:
                                if metric['metric'] not in all_metrics:
                                    all_metrics.append(metric['metric'])

        for grid in outcrop_json['grid_data']:
            for samp in grid['samples']:
                if 'latitude' in samp.keys() and 'images' in samp.keys() and len(samp['images']) > 0:
                    for im in samp['images']:
                        if 'metrics' in im.keys() and len(im['metrics']) > 0:
                            for metric in im['metrics']:
                                if metric['metric'] not in all_metrics:
                                    all_metrics.append(metric['metric'])

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

    # do the same for scales
    if desired_scales is None:
        # first, we need to know all the scales available to input
        all_scales = []

        # find them by looping through all samples, and if they have gps coordinates and image metrics, 
        # add new metrics and scales to the lists
        for strat in outcrop_json['strat_data']:
            for samp in strat['samples']:
                if 'latitude' in samp.keys() and 'images' in samp.keys() and len(samp['images']) > 0:
                    for im in samp['images']:
                        if len(im['metrics']) > 0:
                            for metric in im['metrics']:
                                if metric['metric'] in desired_metrics and metric['scale'] not in all_scales:
                                    all_scales.append(metric['scale'])

        for grid in outcrop_json['grid_data']:
            for samp in grid['samples']:
                if 'latitude' in samp.keys() and 'images' in samp.keys() and len(samp['images']) > 0:
                    for im in samp['images']:
                        if 'metrics' in im.keys() and len(im['metrics']) > 0:
                            for metric in im['metrics']:
                                if metric['metric'] in desired_metrics and metric['scale'] not in all_scales:
                                    all_scales.append(metric['scale'])

        # ask for input
        print('The available scales are:')

        # print each scale on a new line with an index
        for i in range(len(all_scales)):
            print(str(i) + ': ' + str(all_scales[i]))

        # get the input
        desired_scales = input('Please enter the indices of the scales you would like to select, separated by commas: ')

        # split the input into a list of indices
        desired_scales = desired_scales.split(',')

        # convert the indices to integers
        desired_scales = [int(sub) for sub in desired_scales]

        # use the indices to get the desired scales
        desired_scales = [all_scales[sub] for sub in desired_scales]

    # ready to go, set up an empty dataframe to catch the data with classes sample name, latitude, longitude, msl, wavelength, metric_name, scale, and the values
    out_df = pd.DataFrame()
    out_df['sample_name'] = []
    out_df['latitude'] = []
    out_df['longitude'] = []
    out_df['msl'] = []
    out_df['wavelength'] = []
    out_df['metric_name'] = []
    out_df['scale'] = []
    out_df['normalized'] = []
    out_df['value'] = []

    # now loop through all the strat data and grid data, and add the data to the dataframe
    for strat in outcrop_json['strat_data']:
        for samp in strat['samples']:
            if 'latitude' in samp.keys() and 'images' in samp.keys() and len(samp['images']) > 0:
                for im in samp['images']:
                    if 'metrics' in im.keys() and len(im['metrics']) > 0:
                        for metric in im['metrics']:
                            if metric['metric'] in desired_metrics and metric['scale'] in desired_scales:
                                # slitghly different case if the metric is percentile, because then we need the percentile value
                                if metric['metric'] == 'percentile':
                                    new_row = {'sample_name': samp['sample_name'], 'latitude': samp['latitude'], 'longitude': samp['longitude'], 'msl': samp['msl'], 'wavelength': im['wavelength'], 'metric_name': [metric['metric']+'_'+metric['percentile']], 'scale': metric['scale'], 'normalized': metric['normalized'], 'value': metric['value']}
                                else:
                                    new_row = {'sample_name': samp['sample_name'], 'latitude': samp['latitude'], 'longitude': samp['longitude'], 'msl': samp['msl'], 'wavelength': im['wavelength'], 'metric_name': metric['metric'], 'scale': metric['scale'], 'normalized': metric['normalized'], 'value': metric['value']}
                                new_df = pd.DataFrame(new_row, index=[0])
                                out_df = pd.concat([out_df, new_df], ignore_index=True)

    for grid in outcrop_json['grid_data']:
        for samp in grid['samples']:
            if 'latitude' in samp.keys() and 'images' in samp.keys() and len(samp['images']) > 0:
                for im in samp['images']:
                    if 'metrics' in im.keys() and len(im['metrics']) > 0:
                        for metric in im['metrics']:
                            if metric['metric'] in desired_metrics and metric['scale'] in desired_scales:
                                # slitghly different case if the metric is percentile, because then we need the percentile value
                                if metric['metric'] == 'percentile':
                                    new_row = {'sample_name': samp['sample_name'], 'latitude': samp['latitude'], 'longitude': samp['longitude'], 'msl': samp['msl'], 'wavelength': im['wavelength'], 'metric_name': [metric['metric']+'_'+metric['percentile']], 'scale': metric['scale'], 'normalized': metric['normalized'], 'value': metric['value']}
                                else:
                                    new_row = {'sample_name': samp['sample_name'], 'latitude': samp['latitude'], 'longitude': samp['longitude'], 'msl': samp['msl'], 'wavelength': im['wavelength'], 'metric_name': metric['metric'], 'scale': metric['scale'], 'normalized': metric['normalized'], 'value': metric['value']}
                                new_df = pd.DataFrame(new_row, index=[0])
                                out_df = pd.concat([out_df, new_df], ignore_index=True)

    # return the dataframe
    return out_df

# ----------------------------------------------------------------------------------------
def select_gridded_bow(outcrop_json):
    """
    A function to go through an outcrop json and select only bag of words data with gps coordinates,
    returning them in a dataframe

    Keyword arguments:
    outcrop_json -- the json file to extract words from, as a dictionary

    Returns:
    out_df -- a dataframe containing the bag of words data
    """

    # set up an empty dataframe to catch the data with classes sample name, latitude, longitude, msl, and the values
    out_df = pd.DataFrame()
    out_df['sample_name'] = []
    out_df['latitude'] = []
    out_df['longitude'] = []
    out_df['msl'] = []
    out_df['bow'] = []

    # now loop through all the strat data and grid data, and add the data to the dataframe
    for strat in outcrop_json['strat_data']:
        for samp in strat['samples']:
            if 'latitude' in samp.keys() and 'bag_of_words' in samp.keys() and len(samp['bag_of_words']) > 0:
                new_row = {'sample_name': samp['sample_name'], 'latitude': samp['latitude'], 'longitude': samp['longitude'], 'msl': samp['msl'], 'bow': samp['bag_of_words']}
                new_df = pd.DataFrame([new_row], index=[0])
                out_df = pd.concat([out_df, new_df], ignore_index=True)

    for grid in outcrop_json['grid_data']:
        for samp in grid['samples']:
            if 'latitude' in samp.keys() and 'bag_of_words' in samp.keys() and len(samp['bag_of_words']) > 0:
                new_row = {'sample_name': samp['sample_name'], 'latitude': samp['latitude'], 'longitude': samp['longitude'], 'msl': samp['msl'], 'bow': samp['bag_of_words']}
                new_df = pd.DataFrame([new_row], index=[0])
                out_df = pd.concat([out_df, new_df], ignore_index=True)

    # return the dataframe
    return out_df

# ----------------------------------------------------------------------------------------
def data_audit(outcrop_json, sample_set):
    """
    A simple function to go through the samples in an outcrop json and check for missing data

    Keyword arguments:
    outcrop_json -- the json file to look through, as a dictionary
    sample_set -- indicating which types of samples to look at, options are:
        'strat_data'
        'grid_data'
        'all'

    Returns:
    None, but prints out the names of samples missing data
    """

    # we'll look for missing images, geochem, bag of words and point count data, set up empty lists to catch the names
    missing_images = []
    missing_geochem = []
    missing_point_count_fracs = []
    missing_point_counts = []
    missing_bow = []

    # loop through the strat data if we are looking at it
    if sample_set == 'strat_data' or sample_set == 'all':
        for strat in outcrop_json['strat_data']:
            for samp in strat['samples']:
                if 'images' not in samp.keys() or len(samp['images']) == 0:
                    missing_images.append(samp['sample_name'])
                if 'geochem_measurements' not in samp.keys() or len(samp['geochem_measurements']) == 0:
                    missing_geochem.append(samp['sample_name'])
                if 'point_count_fracs' not in samp.keys() or len(samp['point_count_fracs']) == 0:
                    missing_point_count_fracs.append(samp['sample_name'])
                if 'point_counts' not in samp.keys() or len(samp['point_counts']) == 0:
                    missing_point_counts.append(samp['sample_name'])
                if 'bag_of_words' not in samp.keys() or len(samp['bag_of_words']) == 0:
                    missing_bow.append(samp['sample_name'])
    
    # loop through the grid data if we are looking at it
    if sample_set == 'grid_data' or sample_set == 'all':
        for grid in outcrop_json['grid_data']:
            for samp in grid['samples']:
                if 'images' not in samp.keys() or len(samp['images']) == 0:
                    missing_images.append(samp['sample_name'])
                if 'geochem_measurements' not in samp.keys() or len(samp['geochem_measurements']) == 0:
                    missing_geochem.append(samp['sample_name'])
                if 'point_count_fracs' not in samp.keys() or len(samp['point_count_fracs']) == 0:
                    missing_point_count_fracs.append(samp['sample_name'])
                if 'point_counts' not in samp.keys() or len(samp['point_counts']) == 0:
                    missing_point_counts.append(samp['sample_name'])
                if 'bag_of_words' not in samp.keys() or len(samp['bag_of_words']) == 0:
                    missing_bow.append(samp['sample_name'])

    # before printing results, refine point count output.
    # the set difference between missing_point_counts and missing_point_count_fracs will give us 
    # the samples that have point counts but no point count fractions and just need to be updated
    just_update = list(set(missing_point_counts) - set(missing_point_count_fracs))
        

    # print out the results, in a nice format
    print('The following samples are missing images (n = ' + str(len(missing_images)) + '):')
    for name in missing_images:
        print(name)
    print('The following samples are missing geochem data (n = ' + str(len(missing_geochem)) + '):')
    for name in missing_geochem:
        print(name)
    print('The following samples are missing point counts altogethe (n = ' + str(len(missing_point_counts)) + '):')
    for name in missing_point_counts:
        print(name)
    print('The following samples are missing point count fractions, but have point counts (n = ' + str(len(just_update)) + '):')
    for name in just_update:
        print(name)
    print('The following samples are missing bag of words data: (n = ' + str(len(missing_bow)) + '):')
    for name in missing_bow:
        print(name)