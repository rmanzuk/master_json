# Starting the part of this project where I collect functions that I will use to process and keep json files
# This script holds package imports and functions that I will use in the jupyter notebooks for now
# written 10/17/2023 by Ryan A. Manzuk
# last edited 10/17/2023 by Ryan A. Manzuk

####################################################################################################
# Imports
import json
import glob
import pandas as pd
import pdb
import os
import matplotlib.pyplot as plt
import numpy as np
from skimage import io
from skimage import exposure
from skimage.transform import rescale, resize
from skimage.morphology import disk, erosion, dilation, opening, closing, white_tophat
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


# take strat data and look up the lithology for a given height from the strat column with the same name likeness code
def get_lithology(strat_data, strat_cols):
    # strat_data is a dict that contains stratigraphic samples. strat_cols is a list of dicts that contains stratigraphic columns
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


# get all of the unique likeness codes in a json file
def unique_likenesses(input_json):
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


# get the indices of where an item is in a list of strings
def string_find(solo_string, string_list):
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


# set up a directory of all files in a directory
def get_file_list(path_to_directory, wild_card='*'):
    file_list = glob.glob(path_to_directory + wild_card)
    return file_list

# read in a csv with point count coordinates from J micro vision and return a list of dictionaries 
# that can be placed into a sample dictionary.
# note the positions of the x and y are returned in terms of fraction of the image width and height. 
# That facilitates plotting on any resampled image.
def jmicro_to_dict(csv_path, max_x, max_y, delimiter=';'):
    # read in the csv to a dataframe
    coord_data = pd.read_csv(csv_path,sep=delimiter)
    
    # make a counts array to hold the count dicts
    all_counts = []
    # loop through each row in the coord df
    for row in coord_data.iterrows():   
        # set up the dict for this row
        this_count = {}
        # add the class
        # if the class is Nan, we need to make it null for the json
        if pd.isnull(row[1]['Class']):
            this_count['class'] = None
        else:
            this_count['class'] = row[1]['Class']
        # add the x and y positions
        this_count['x_position'] = row[1]['Pos X']/max_x
        this_count['y_position'] = row[1]['Pos Y']/max_y
        # place in the counts array
        all_counts.append(this_count)

    # return the counts array
    return all_counts


coord_path = '/Users/ryan/Dropbox (Princeton)/reef_survey_project/nevada/data_spreadsheets/point_counts/smg_excel_exports/smg_1_Point Counting_random-350_coordinates.csv'
coords_for_dict = jmicro_to_dict(coord_path, 5000, 3800, delimiter=';')