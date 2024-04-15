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

# ----------------------------------------------------------------------------------------
def pc_crossplot_vectors(pc_scores, pc_loadings, pc_nums, variable_labels=None, n_vectors=2, n_recon=5, phase_tags=None):
    """
    A function to make a crossplot of the scores of two principal components, ask the user to input 
    points that represent the ends of vectors, and then make plots that show how  

    Keyword arguments:
    pc_scores -- the scores of the principal components. just input the whole array, and the 
    function will take the columns corresponding to the pc_nums
    pc_loadings -- the loadings of the principal components. just input the whole array, and the
    function will take the rows corresponding to the pc_nums
    pc_nums -- a list of the principal components to compare as a 2-element list. Should be 1-indexed
    variable_labels -- a list of string labels for the variables that correspond to the loadings
    n_vectors -- the number of vectors to investigate within the crossplot
    n_recon -- the number of points to reconstruct along each vector
    phase_tags -- a list of phase tags of same length as the number of sample to color code the points

    Returns:
    None
    """

    # subtract 1 to make the pc_nums 0-indexed
    pc_nums = [i-1 for i in pc_nums]

    # start by making a plot of the scores of the two principal components for the user to select vectors
    fig, ax = plt.subplots(1,1, figsize=(8,5))
    ax.scatter(pc_scores[:,pc_nums[0]], pc_scores[:,pc_nums[1]], c='gray')
    ax.set_xlabel('PC'+str(pc_nums[0]+1))
    ax.set_ylabel('PC'+str(pc_nums[1]+1))

    # now ginput n_vectors * 2 times, and store the vectors
    vectors = []
    print('Click on the plot to input vectors')
    for i in range(n_vectors):
        vectors.append(plt.ginput(2))       

    # now get the x and y coordinates of the recon points from the vectors
    # get the n_recon points in the cross plot along the vectors 
    recon_points_x = np.zeros((n_vectors, n_recon))
    recon_points_y = np.zeros((n_vectors, n_recon))
    for i in range(n_vectors):
        # just need to linspace between end points of the vectors
        recon_points_x[i] = np.linspace(vectors[i][0][0], vectors[i][1][0], n_recon)
        recon_points_y[i] = np.linspace(vectors[i][0][1], vectors[i][1][1], n_recon)

    # now that we have the recon points, use them in pc space to make spectra
    recon_spectra = np.zeros((pc_scores.shape[1], n_recon, n_vectors))
    for i in range(n_vectors):
        for j in range(n_recon):
            reconstructed_x = pc_loadings[pc_nums[0],:]*recon_points_x[i,j]
            reconstructed_y = pc_loadings[pc_nums[1],:]*recon_points_y[i,j]
            recon_spectra[:,j,i] = reconstructed_x + reconstructed_y

    # and now make a pretty multipanel plot. The furthest left panel will be the color, size coded cross plot, and the rest will be the reconstructed spectra
    fig, ax = plt.subplots(1,n_vectors+1, figsize=(5*(n_vectors+1),5))

    # if we don't have phase tags, we can just scatter the points in gray on the first plot
    if phase_tags == None:
        ax[0].scatter(pc_scores[:,pc_nums[0]], pc_scores[:,pc_nums[1]], c='gray')

    # if we do have phase tags, we'll scatter the points in different colors
    else:
        # we may run out of colors before getting through phases, so list a few symbols to switch to once we run out of colors
        symbols = ['o', 's', 'D', 'v', '^', '<', '>', 'p', 'P', '*', 'X']
        n_colors = len(plt.rcParams['axes.prop_cycle'].by_key()['color']) 
        # make the color coded cross plot
        unique_phases = np.unique(phase_tags)
        count = 1
        for phase in unique_phases:
            if count >= n_colors:
                inds_to_plot = np.where(np.array(phase_tags) == phase)[0]
                ax[0].scatter(pc_scores[inds_to_plot,pc_nums[0]], pc_scores[inds_to_plot,pc_nums[1]], label=phase, marker=symbols[count-n_colors])
            else:
                inds_to_plot = np.where(np.array(phase_tags) == phase)[0]
                ax[0].scatter(pc_scores[inds_to_plot,pc_nums[0]], pc_scores[inds_to_plot,pc_nums[1]], label=phase)
            count += 1

    # label the axes
    ax[0].set_xlabel('PC'+str(pc_nums[0]+1))
    ax[0].set_ylabel('PC'+str(pc_nums[1]+1))
    ax[0].legend(loc='center left', bbox_to_anchor=(1, 0.5))

    # also add the vectors to the plot as arrows
    for i in range(n_vectors):
        ax[0].arrow(vectors[i][0][0], vectors[i][0][1], vectors[i][1][0]-vectors[i][0][0], vectors[i][1][1]-vectors[i][0][1], head_width=0.1, head_length=0.1, fc='black', ec='black')

    # now we'll plot the reconstructed spectra
    cmap = plt.get_cmap('PuBuGn')
    cmap_vals = np.linspace(0,1,n_recon)
    for i in range(n_vectors):
        for j in range(n_recon):
            ax[i+1].plot(recon_spectra[:,j,i], color=cmap(cmap_vals[j]))
        ax[i+1].set_xlabel('Variable')
        ax[i+1].set_ylabel('Value')

    # if we have variable labels, we'll use them
    if variable_labels:
        for i in range(n_vectors+1):
            ax[i].set_xticks(np.linspace(0,len(variable_labels),len(variable_labels)))
            ax[i].set_xticklabels(variable_labels, rotation=90)

    plt.tight_layout()
    plt.show()



