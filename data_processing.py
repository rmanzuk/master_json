# set of functions for doing data processing tasks, mostly in arrays after sampling from json
# written by R. A. Manzuk 03/23/2024
# last updated 03/23/2024

##########################################################################################
# package imports
##########################################################################################
import numpy as np # for array handling and math
import pdb # for debugging

##########################################################################################
# function definitions
##########################################################################################

# ----------------------------------------------------------------------------------------
def random_sample_strat(input_data, strat_heights, height_interval, n_samplings):
    """
    A function to take a set of data listed stratigraphically and randomly sample from 
    evenly spaced height intervals. Can make as many random sample vectors as desired.
    
    keyword arguments:
    input_data -- the data to sample from
    strat_heights -- the corresponding stratigraphic heights of the data
    height_interval -- the height interval to sample from
    n_samplings -- the number of samplings to make
    """

    # first group the samples by stratigraphic interval
    min_strat = np.nanmin(strat_heights)
    max_strat = np.nanmax(strat_heights)
    strat_bin_edges = np.arange(min_strat,max_strat,height_interval)
    strat_bins = np.digitize(strat_heights,strat_bin_edges)


    # now we can loop through the samplings
    random_samples = np.zeros((len(strat_bin_edges)-1, n_samplings))

    for i in range(n_samplings):
        for j in range(len(strat_bin_edges)-1):
            # get the indices of the data in this bin
            bin_indices = np.where(strat_bins == j)[0]
            # if there are no indices, just skip, put an nan in the output
            if len(bin_indices) == 0:
                random_samples[j,i] = np.nan
                continue
            # sample from these indices
            random_samples[j,i] = np.random.choice(input_data[bin_indices])
        
    # make bin centers to return
    bin_centers = strat_bin_edges[:-1] + height_interval/2

    return random_samples, bin_centers
