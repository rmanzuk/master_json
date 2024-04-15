# set of functions for making plots particular to the project
# written by R. A. Manzuk 12/15/2023
# last updated 12/15/2023

##########################################################################################
# package imports
##########################################################################################
from skimage.transform import downscale_local_mean
import matplotlib.pyplot as plt # for plotting
import numpy as np # for array handling and math
##########################################################################################
# function imports from other files
##########################################################################################
from im_processing import sample_3channel
##########################################################################################
# function definitions
##########################################################################################

# ----------------------------------------------------------------------------------------
def display_point_counts(sample_dict, downscale_factor=5, star_points=None):
    """
    A function to display the point counts on the image, at their x and y coordinates.

    Keyword arguments:
    sample_dict -- a dictionary containing the sample information
    downscale_factor -- the factor by which to downscale the image for display
    star_points -- a list of point indices to mark with a star. They should be input as 1-indexed

    Returns:
    None
    """

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

    # if we have star points, we'll plot them as well
    if star_points:
        for star in star_points:
            # need to subtract 1 to make it 0-indexed
            star = star - 1
            ax.scatter(x[star]*downsized_rgb.shape[1], y[star]*downsized_rgb.shape[0], s=100, marker='*', color='black')
            # and label the star with the point number
            ax.text(x[star]*downsized_rgb.shape[1], y[star]*downsized_rgb.shape[0], str(star+1), fontsize=12, color='red')
    # turn off the axes
    ax.axis('off')
    # add the legend in the upper right, outside the plot
    ax.legend(loc='upper right', bbox_to_anchor=(1.2,1))

    plt.show()