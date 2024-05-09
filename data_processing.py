# set of functions for doing data processing tasks, mostly in arrays after sampling from json
# written by R. A. Manzuk 03/23/2024
# last updated 03/23/2024

##########################################################################################
# package imports
##########################################################################################
import numpy as np # for array handling and math
from scipy.cluster.hierarchy import dendrogram, linkage # for making dendrograms
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

    Returns:
    random_samples -- a 2D array with the random samples in each row
    bin_centers -- the centers of the stratigraphic bins
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

# ----------------------------------------------------------------------------------------
def dendrogram_with_inds(input_data, linkage_method='single', dendrogram_p=None):
    """
    A function to make a dendrogram with sample indices within each leaf. Returns the dendrogram
    and the indices of the leaves.
    
    keyword arguments:
    input_data -- the data to make the dendrogram from, should be a 2D array with samples as rows
    linkage_method -- the linkage method to use in making the dendrogram, see scipy linkage
    dendrogram_p -- the p parameter to pass to the dendrogram function for truncate_mode

    Returns:
    d -- the dendrogram object
    cluster_list -- a list of lists, each list contains the indices of the samples in a leaf
    """

    # make the linkage
    Z = linkage(input_data, linkage_method)

    # make the dendrogram
    if dendrogram_p is not None:
        d = dendrogram(Z, truncate_mode='level', p=dendrogram_p)
    else:
        d = dendrogram(Z)

    # time to get the indices of the samples within each leaf 
    leaves = d['leaves']

    # initiate the cluster list with the cluster number for each leaf
    cluster_list = [[leaf] for leaf in leaves]

    # loop through each leaf to get the cluster list 
    for i in range(len(leaves)):    
        still_going = True

        # quick check if the leaf is already just a single sample
        if len(cluster_list[i]) == 1 and cluster_list[i][0] <= len(Z):
            continue
        # otherwise we gotta keep going
        while still_going:
            # get the cluster numbers available for this leaf
            these_clusters = cluster_list[i].copy()

            # if it's a single sample, we know we should just split it and move on
            if len(these_clusters) == 1:
                linkage_ind = these_clusters[0] - len(input_data)
                cluster_list[i] = [Z[linkage_ind,0], Z[linkage_ind,1]]
                # and go back through the while loop
                continue

            # otherwise, we need to scan for clusters to split
            to_split = []
            for j in range(len(these_clusters)):
                if these_clusters[j] > len(Z):
                    to_split.append(j)

            # if there are no clusters to split, we are done
            if len(to_split) == 0:
                still_going = False
                continue

            # otherwise, we need to split each of the to_split clusters
            for j in to_split:
                # get the cluster number
                cluster_num = these_clusters[j]

                # get the linkage index
                linkage_ind = cluster_num - len(input_data)

                # get the two clusters that result from the split
                cluster1 = Z[int(linkage_ind),0]
                cluster2 = Z[int(linkage_ind),1]

                # remove the old cluster and add the new ones
                cluster_list[i].remove(cluster_num)
                cluster_list[i].append(cluster1)
                cluster_list[i].append(cluster2)

    return d, cluster_list