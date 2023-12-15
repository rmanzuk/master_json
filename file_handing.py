# set of functions for managing and updating files as well as transferring data between
# written by R. A. Manzuk 12/06/2023
# last updated 12/06/2023

##########################################################################################
# package imports
##########################################################################################

import pandas as pd # for data handling
import glob # for getting file lists

##########################################################################################
# function definitions
##########################################################################################

# ----------------------------------------------------------------------------------------
def jmicro_to_dict(csv_path, max_x, max_y, delimiter=';'):
    """
    A function to take a csv of jmicro point counts and convert it to a list of dicts, 
    which can be easily placed in a sample json file. Note that the coordinates are
    stored as fraction of each dimension (x/max_x, y/max_y) to make it easier to 
    plot, etc. later and not have to worry about the resolution of the image. The upper left
    is the origin of the image.

    Keyword arguments:
    csv_path -- the path to the csv file containing the jmicro data
    max_x -- the maximum x value of the point counts. As in, what is the size 
    of the x dimension when the point counts were taken
    max_y -- the maximum y value of the point counts. As in, what is the size
    of the y dimension when the point counts were taken
    delimiter -- the delimiter used in the csv file (default ';')

    Returns:
    all_counts -- a list of dicts, each of which contains the class and x and y positions
    """
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

# ----------------------------------------------------------------------------------------
def get_file_list(path_to_directory, wild_card='*'):
    """
    A function to get a list of all files in a directory. This is useful for
    getting a list of all the jsons in a directory, for example.

    Keyword arguments:
    path_to_directory -- the path to the directory you want to get the file list from  
    wild_card -- a wildcard to filter the file list (default '*')

    Returns:
    file_list -- a list of all the files in the directory
    """
    file_list = glob.glob(path_to_directory + wild_card)
    return file_list