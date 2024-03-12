# set of functions for getting data from images
# written by R. A. Manzuk 12/15/2023
# last updated 12/29/2023

##########################################################################################
# package imports
##########################################################################################
import numpy as np # for array handling
from scipy import ndimage as ndi # for distance transforms, filtering
from scipy.interpolate import interp1d # for making transfer functions
import skimage.io as io # for image reading
from sklearn.decomposition import PCA # for PCA
from skimage.transform import rescale # for rescaling images
from skimage import filters # for image processing
import skimage.measure # for entropy
from skimage.morphology import ellipse # for making structuring elements
import bottleneck as bn # for fast nan functions
from datetime import date # for getting today's date
import pdb # for debugging
##########################################################################################
# function definitions
##########################################################################################

# ----------------------------------------------------------------------------------------
def calc_fill_im_metric(sample_json, light_source, wavelengths, scales, metrics, to_mask=False, to_normalize=False):
    """
    function to calculate an image metric for the sample and fill it in to the sample json

    Keyword arguments:
    sample_json -- the sample json to fill in
    light_source -- the light source for the image desired
    wavelengths -- list of wavelengths for the image desired, as integers
    scales -- list of scales for the image desired, as decimal fractions
    metric -- list of metrics to calculate, as strings, options are:
        'rayleigh_anisotropy' -- the anisotropy of image, according to the Rayleigh criterion
        'entropy' -- the entropy of the image
        'percentile_n' -- the percentile of the image the n after the underscore should be 
        replaced with the desired percentile. multiple percentiles can be calculated, just
        list them all separated by underscores

    Returns:
    sample_json -- the sample json with the metric filled in
    """

    # this function will require looping through a few things, we want to avoid reading 
    # in the image multiple times, so we'll make wavelengths the outer loop
    for w in wavelengths:
        print('Calculating metrics for wavelength ' + str(w))

        # first thing is to figure out which image corresponds to the light source and wavelength
        # get the indices of images that match the light source
        ls_indices = [i for i, s in enumerate(sample_json['images']) if light_source == s['light_source']]

        # get the indices of images that match this wavelength
        wl_indices = [i for i, s in enumerate(sample_json['images']) if s['wavelength'] == w]

        # get the intersection of the two lists
        indices = list(set(ls_indices) & set(wl_indices))

        # if there is more than one image that matches, that's weird, throw an error
        if len(indices) > 1:
            print('Error: more than one image matches the light source and wavelength')
            return sample_json
        
        # if there are no images that match, that's also a problem, throw an error
        if len(indices) == 0:
            print('Error: no images match the light source and wavelength' + str(w))
            return sample_json
        
        # otherwise, read in the image
        raw_im = io.imread(sample_json['path_to_ims'] + '/' + sample_json['images'][indices[0]]['file_name'])

        # first make the image a float spread between 0 and 1
        if raw_im.dtype == 'uint8':
            raw_im = raw_im / 255
        elif raw_im.dtype == 'uint16':
            raw_im = raw_im / 65536
        elif raw_im.dtype == 'float32':
            raw_im = raw_im
        elif raw_im.dtype == 'float64':
            raw_im = raw_im
        else:
            print('Error: image dtype not recognized')
            return sample_json

        # if we're masking, we need to do that here
        if to_mask:
            # find the image index where the tag is 'boundary_mask'
            mask_index = [i for i, s in enumerate(sample_json['images']) if 'boundary_mask' in s['tag']]
            # if there is more than one image that matches, that's weird, throw an error
            if len(mask_index) > 1:
                print('Error: more than one image matches the mask tag')
                return sample_json
            # if there are no images that match, that's also a problem, throw an error
            if len(mask_index) == 0:
                print('Error: no images match the mask tag')
                return sample_json
            # otherwise, read in the mask
            mask = io.imread(sample_json['path_to_ims'] + '/' + sample_json['images'][mask_index[0]]['file_name'], plugin='pil')
            # use the mask to get the indices of the pixels to set to nan
            nan_indices = np.where(mask == 0)
            # set the pixels to nan
            raw_im[nan_indices] = np.nan

        # if we're normalizing, we need to do that here
        if to_normalize:
            # normalize the image
            normed_im = normalize_band(raw_im, '0to1')
        else:
            # if we're not normalizing, just set the normalized image to the raw image
            normed_im = raw_im
            
        # also, before looping, if this image entry doesn't have a metrics key, make it
        if 'metrics' not in sample_json['images'][indices[0]]:
            sample_json['images'][indices[0]]['metrics'] = []

        # now we can loop through the scales
        for s in scales:
            print('Calculating metrics for scale ' + str(s))    
        
            # rescale the image, only if the scale is less than 1
            if s < 1:
                scaled_im = rescale(normed_im, s, anti_aliasing=False)
            else:
                scaled_im = normed_im

            # now we can loop through the metrics
            for m in metrics:
                print('Calculating metric ' + m)

                # if the sample json already has this metric at this scale , we don't want to recalculate it
                # frist check if any metrics match
                metric_match = [d for d in sample_json['images'][indices[0]]['metrics'] if d['metric'] == m]
                # and then check if any of those matches have the same scale
                scale_match = [d for d in metric_match if d['scale'] == s]
                # and check check if any of those matches have the same normalization status
                norm_match = [d for d in scale_match if d['normalized'] == to_normalize]
                # if there is a match, we don't want to recalculate
                if len(norm_match) > 0:
                    print('Metric ' + m + ' already calculated at scale ' + str(s))
                    continue

                # check which metric we're calculating
                if m == 'rayleigh_anisotropy':
                    
                    # calculate the anisotropy
                    r, theta_r = grad_r(im_grad(scaled_im, 'sobel')[1], mag_im=im_grad(scaled_im, 'sobel')[0])

                    # now make a dictionary for this measurement that we can add to the sample json
                    metric_dict = {'metric': m, 'value': r, 'scale': s, 'angle': theta_r, 'normalized': to_normalize}

                    # also add today's date to the dictionary, because that might be good metatdata to compare to code version
                    metric_dict['date_calculated'] = date.today().strftime("%m/%d/%Y")

                    # add the metric to the image entry
                    sample_json['images'][indices[0]]['metrics'].append(metric_dict)

                elif m == 'entropy':

                    # calculate the entropy
                    ent = skimage.measure.shannon_entropy(scaled_im)

                    # now make a dictionary for this measurement that we can add to the sample json
                    metric_dict = {'metric': m, 'value': ent, 'scale': s, 'normalized': to_normalize}

                    # also add today's date to the dictionary, because that might be good metatdata to compare to code version
                    metric_dict['date_calculated'] = date.today().strftime("%m/%d/%Y")

                    # add the metric to the image entry
                    sample_json['images'][indices[0]]['metrics'].append(metric_dict)

                elif 'percentile' in m:

                    # get the list of percentiles asked for
                    percentiles = m.split('_')[1:]

                    # calculate the percentiles
                    perc_vals = np.nanpercentile(scaled_im, [int(i) for i in percentiles])

                    # make a dictionary for each percentile measurement that we can add to the sample json
                    for p, pv in zip(percentiles, perc_vals):
                        metric_dict = {'metric': 'percentile', 'value': pv, 'scale': s, 'percentile': p, 'normalized': to_normalize}

                        # also add today's date to the dictionary, because that might be good metatdata to compare to code version
                        metric_dict['date_calculated'] = date.today().strftime("%m/%d/%Y")

                        # add the metric to the image entry
                        sample_json['images'][indices[0]]['metrics'].append(metric_dict)


                # if the metric isn't recognized, throw an error
                else:
                    print('Error: metric ' + m + ' not recognized') 
                    return sample_json

    # all done, return the sample json
    return sample_json

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
        band_min = np.nanmin(band_im)
        band_max = np.nanmax(band_im)
        # normalize
        normed_band = (band_im - band_min) / (band_max - band_min)

    #  standard stretch method defines transfer functions for each band such that
    # 0 maps to 1/255
    # mean - 2*std maps to 2/255
    # mean + 2*std maps to 254/255
    # 255 (or other max value) maps to 255/255
    elif method == 'std_stretch':
        # get the mean and std
        band_mean = np.nanmean(band_im)
        band_std = np.nanstd(band_im)
        band_min = np.nanmin(band_im)
        band_max = np.nanmax(band_im)
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
        band_vals = np.array([0, np.nanmax([band_min,1,band_mean - 2*band_std]), 
                              np.nanmin([band_mean + 2*band_std, band_max]), possible_max])
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

# ----------------------------------------------------------------------------------------
def im_pca(image):
    """
    A function to perform PCA on an image, returning the multichannel PCA image

    Keyword arguments:
    image -- the image to perform PCA on, any format is fine, we'll normalize within the function

    Returns:
    pca_image -- the image after PCA has been performed
    """
    # get the shape of the image
    shape = image.shape

    # reshape the image to be 2D
    reshaped_image = np.reshape(image, (shape[0]*shape[1], shape[2]))

    # normalize with 0 center and std 1
    reshaped_image = (reshaped_image - np.mean(reshaped_image, axis=0)) / np.std(reshaped_image, axis=0)

    # the SVD of the PCA can be pretty inefficient, so if we have greater than 10000 pixels, randomly sample 10000
    too_big = False
    if reshaped_image.shape[0] > 10000:
        reshaped_image = reshaped_image[np.random.choice(reshaped_image.shape[0], 10000, replace=False),:]
        too_big = True

    # perform PCA
    pca = PCA(svd_solver='randomized')
    pca_image = pca.fit_transform(reshaped_image)

    # if we randomly subsampleed, we need to get the loadings to project the image back to the original space
    loadings = pca.components_
    explained_variance = pca.explained_variance_ratio_
    if too_big:
        original_cols = np.reshape(image, (shape[0]*shape[1], shape[2]))
        original_cols = (original_cols - np.mean(original_cols, axis=0)) / np.std(original_cols, axis=0)
        pca_image = np.dot(original_cols, loadings.T)

    # rescale to 0-1
    pca_image = (pca_image - np.min(pca_image, axis=0)) / (np.max(pca_image, axis=0) - np.min(pca_image, axis=0))

    # reshape back to 3D
    pca_image = np.reshape(pca_image, (shape[0], shape[1], shape[2]))

    return pca_image, loadings, explained_variance

# ----------------------------------------------------------------------------------------
def im_grad(input_im, filter_type):
    """
    return the gradient magnitude and direction of an image using a given filter

    Keyword arguments:
    input_im -- the image to extract data from
    filter_type -- the type of filter to use, either 'sobel', 'scharr', 'prewitt', 
    'roberts', or 'farid'

    Returns:
    grad_mag -- the gradient magnitude
    grad_dir -- the gradient direction
    """
    # user can decide which filter to use
    if filter_type == 'sobel':
        # use the sobel filter
        grad_x = filters.sobel_h(input_im)
        grad_y = filters.sobel_v(input_im)
    elif filter_type == 'scharr':
        # use the scharr filter
        grad_x = filters.scharr_h(input_im)
        grad_y = filters.scharr_v(input_im)
    elif filter_type == 'prewitt':
        # use the prewitt filter
        grad_x = filters.prewitt_h(input_im)
        grad_y = filters.prewitt_v(input_im)
    elif filter_type == 'roberts':
        # use the roberts filter
        grad_x = filters.roberts_pos_diag(input_im)
        grad_y = filters.roberts_neg_diag(input_im)
    elif filter_type == 'farid':
        # use the farid filter
        grad_x = filters.farid_h(input_im)
        grad_y = filters.farid_v(input_im)
    else:
        # return an error that the filter type is not recognized
        print('Error: filter type not recognized')

    # calculate the magnitude and direction of the gradient
    grad_mag = np.sqrt(grad_x**2 + grad_y**2)
    grad_dir = np.arctan2(grad_y, grad_x)

    return grad_mag, grad_dir

# ----------------------------------------------------------------------------------------
def grad_r(dir_im, mag_weights=None, mag_thresh=None, neigh_rad=None, mag_im=None):
    """
    return the angular dispersion of the gradient direction, possibly just withing 
    neighborhoods or weighted by magnitude

    Keyword arguments:
    dir_im -- the image of gradient directions
    mag_weights -- the magnitude image to use to weight the gradient directions
    mag_thresh -- the percentile threshold to use to mask the image
    neigh_rad -- the radius of the neighborhood to use to calculate the angular dispersion
    mag_im -- the image of gradient magnitudes

    Returns:
    r -- the angular dispersion
    theta_r -- the mean angle
    """
    # no matter what options are input, double the gradient angles and take the sine and cosine
    sin_im = np.sin(2*dir_im)
    cos_im = np.cos(2*dir_im)

    # if no neighborhood is input, we can do this simply and globally
    if neigh_rad is None:
        # if there is a threshold, we can use it to mask the image
        if mag_thresh is not None:
            # make mask of gradients less than the percentile threshold
            nan_mask = np.isnan(mag_im)
            thresh_mask = np.greater(dir_im, np.percentile(mag_im, mag_thresh), where=~nan_mask)
            # use the mask to make low gradient pixels nan in sine and cosine images
            sin_im[~thresh_mask] = np.nan
            cos_im[~thresh_mask] = np.nan

            sin_mean = bn.nanmean(sin_im)
            cos_mean = bn.nanmean(cos_im)
        # now we can move on to take a mean of the sine and cosine images
        if mag_weights is not None:
            # if there are weights, we can use them to weight the mean
            # first, we need to normalize the weights
            norm_weights = mag_weights/np.sum(mag_weights)
            # now we can take the weighted mean
            sin_mean = bn.nansum(sin_im*norm_weights)
            cos_mean = bn.nansum(cos_im*norm_weights)
        else:
            # otherwise we just take the regular mean
            sin_mean = bn.nanmean(sin_im)
            cos_mean = bn.nanmean(cos_im)

        # now we can calculate the angular dispersion
        r = np.hypot(sin_mean, cos_mean) 

        # and continue to to get the mean angle
        theta_r = np.arctan((sin_mean/r)/(cos_mean/r))
    else: 
        # in this case we need to think about neighborhoods
        # first we'll need a structuring element
        selem = ellipse(neigh_rad, neigh_rad)
        # if there is a threshold, we can use it to mask the image
        if mag_thresh is not None:
            # in this case, it's not really clear what to do. Return an error for now
            print('Error: thresholding not implemented for neighborhood option')
        if mag_weights is not None:
            # in this case, we need to to know the mag_weights for each pixel in the neighborhood sum to 1
            # so we need to make a sum filter of the image. We'll apply it later
            sum_filtered = ndi.correlate(mag_weights, selem.astype(float))
            # multiply by the raw weights
            sin_im = sin_im * mag_weights
            cos_im = cos_im * mag_weights
            # now we can sum the raw weighted sine and cosine images
            # and normalize by the weight sum after
            sin_mean = ndi.correlate(sin_im, selem.astype(float))/sum_filtered
            cos_mean = ndi.correlate(cos_im, selem.astype(float))/sum_filtered
        else:
            # if we didn't have weights, we can just use the mean filter over the sine and cosine images
            sin_mean = ndi.correlate(sin_im, selem.astype(float)/np.sum(selem))
            cos_mean = ndi.correlate(cos_im, selem.astype(float)/np.sum(selem))

        # now we can calculate the angular dispersion
        r = np.hypot(sin_mean, cos_mean)
        # and continue to to get the mean angle
        theta_r = np.arctan((sin_mean/r)/(cos_mean/r))

    return r, theta_r