# Starting the part of this project where I collect functions that I will use to process and keep json files
# This script holds package imports and functions that I will use in the jupyter notebooks for now
# written 10/17/2023 by Ryan A. Manzuk
# last edited 11/27/2023 by Ryan A. Manzuk

####################################################################################################
# Imports
import json
import glob
import pandas as pd
import pdb
import os
import math
import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage as ndi
from skimage import io
from skimage import exposure
from skimage.filters import threshold_otsu, threshold_local, try_all_threshold
from skimage.transform import rescale, resize, downscale_local_mean
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
from skimage.morphology import disk, ellipse, erosion, dilation, opening, closing, white_tophat
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
####################################################################################################

# a function to just take a sample dict/json and return a 3 channel image
def sample_3channel(sample_dict, light_source='reflectance', wavelengths=[625,530,470], to_float=True):
    # get the indices of images that match the light source
    ls_indices = [i for i, s in enumerate(sample_dict['images']) if light_source == s['light_source']]
    # get the indices of images that match the wavelengths
    wl_indices = [i for i, s in enumerate(sample_dict['images']) if s['wavelength'] in wavelengths]
    # get the intersection of the two lists
    indices = list(set(ls_indices) & set(wl_indices))
    # sort the indices to match the order of the input wavelengths
    indices.sort(key=lambda x: wavelengths.index(sample_dict['images'][x]['wavelength']))
    # get the image names
    image_names = [sample_dict['images'][i]['file_name'] for i in indices]
    # combine the image names into full paths given the path_to_ims 
    image_paths = [sample_dict['path_to_ims']+i for i in image_names]
    # read in the images and stack them
    images = [io.imread(i) for i in image_paths]
    # if to_float, convert to float, normalize to 0-1
    if to_float:
        images = [normalize_band(i, '0to1') for i in images]

    # stack into 3 channel image
    return np.dstack(images)

# function to display the point counts on the image
def display_point_counts(sample_dict, downscale_factor=5):
    # first need to read in the rgb image
    rgb = sample_3channel(sample_dict, light_source='reflectance', wavelengths=[625,530,470], to_float=True)
    # downsize the image to make it easier to display
    downsized_rgb = downscale_local_mean(rgb, (downscale_factor,downscale_factor,1))

    # take out the list of pc_dicts to make things easier
    pc_list = sample_dict['point_counts']

    # and now we need to extract the point counts from the pc_dict, making 3 litsts; x, y, and class
    x = [i['x_position'] for i in pc_list]
    y = [i['y_position'] for i in pc_list]
    pc_class = [i['class'] for i in pc_list]

    # need to know the unique classes
    unique_classes = list(set(pc_class))

    # should be ready to show the image and overlay each class
    fig, ax = plt.subplots(figsize=(10,10))
    ax.imshow(downsized_rgb)
    for i in unique_classes:
        # get the indices of the points that match the class
        indices = [j for j, s in enumerate(pc_class) if i == s]
        # to handle the label, if the class is none, we'll just call it 'none'
        if i == None:
            i = 'none'
        # plot the points, multiplying by the size of the image to get the right coordinates
        ax.scatter(np.array(x)[indices]*downsized_rgb.shape[1], np.array(y)[indices]*downsized_rgb.shape[0], s=5, label=i)
    # turn off the axes
    ax.axis('off')
    # add the legend in the upper right, outside the plot
    ax.legend(loc='upper right', bbox_to_anchor=(1.2,1))

# function to normalize a band with different methods
def normalize_band(band_im, method='0to1'):
    # 0to1 method just normalizes between 0 and 1
    if method == '0to1':
        # get the min and max
        band_min = np.min(band_im)
        band_max = np.max(band_im)
        # normalize
        normed_band = (band_im - band_min) / (band_max - band_min)

    #  standard stretch method defines transfer functions for each band such that
    # 0 maps to 1/255
    # mean - 2*std maps to 2/255
    # mean + 2*std maps to 254/255
    # 255 (or other max value) maps to 255/255
    elif method == 'std_stretch':
        # get the mean and std
        band_mean = np.mean(band_im)
        band_std = np.std(band_im)
        band_min = np.min(band_im)
        band_max = np.max(band_im)
        # define the original band values, but first need to get dtype to know what the possible max is
        if band_im.dtype == 'uint8':
            possible_max = 255
        elif band_im.dtype == 'uint16':
            possible_max = 65536
        elif band_im.dtype == 'float32':
            possible_max = 1
        elif band_im.dtype == 'float64':
            possible_max = 1
        else:
            print('Error: dtype not recognized')
            return
        band_vals = np.array([0, np.max([band_min,1,band_mean - 2*band_std]), 
                              np.min([band_mean + 2*band_std, band_max]), possible_max])
        transfer_vals = np.array([0, 1/255, 254/255, 255/255])
        transfer_func = interp1d(band_vals, transfer_vals)
        # apply the transfer function
        normed_band = transfer_func(band_im)
    return normed_band

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
