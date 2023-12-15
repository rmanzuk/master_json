# set of functions for getting data from images
# written by R. A. Manzuk 12/15/2023
# last updated 12/15/2023

##########################################################################################
# package imports
##########################################################################################
import numpy as np # for array handling
from scipy.interpolate import interp1d # for making transfer functions
import skimage.io as io # for image reading
##########################################################################################
# function definitions
##########################################################################################

# ----------------------------------------------------------------------------------------
def normalize_band(band_im, method='0to1'):
    """
    A function to normalize a single band image via a variety of methods.

    Keyword arguments:
    band_im -- the image to be normalized
    method -- the method to use for normalization. Options are:
        '0to1' -- normalize between 0 and 1
        'std_stretch' -- normalize between 0 and 1, but using a transfer function that maps
        mean - 2*std to 2/255 and mean + 2*std to 254/255

    Returns:
    normed_band -- the normalized band
    """
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

# ----------------------------------------------------------------------------------------
def sample_3channel(sample_dict, light_source='reflectance', wavelengths=[625,530,470], to_float=True):
    """
    A function to sample a 3 channel image from a sample dict.

    Keyword arguments:
    sample_dict -- the sample dict to sample from
    light_source -- the light source to sample from (default 'reflectance')
    wavelengths -- the wavelengths to sample from (default [625,530,470])
    to_float -- whether to convert the image to float (default True)

    Returns:
    image -- the 3 channel image
    """
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