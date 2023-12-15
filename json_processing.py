# set of functions for working with json files for analysis
# written by R. A. Manzuk 12/15/2023
# last updated 12/15/2023

##########################################################################################
# package imports
##########################################################################################
import numpy as np # for array handling
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