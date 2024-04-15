# set of functions for doing geospatial operations
# written by R. A. Manzuk 03/21/2024
# last updated 03/21/2024

##########################################################################################
# package imports
##########################################################################################

from pyproj import Proj # for converting utm to lat long
import numpy as np # for array operations
import pdb # for debugging

##########################################################################################
# function definitions
##########################################################################################

# ----------------------------------------------------------------------------------------
def latlong_to_utm(lat, long, zone, hemisphere):
    """ 
    convert lat long coordinates to utm coordinates using the pyproj package

    Keyword arguments:
    lat -- the latitude coordinate list
    long -- the longitude coordinate list
    zone -- the utm zone number, as an integer
    hemisphere -- the hemisphere, as a string, either 'north' or 'south'

    Returns:
    x -- the x coordinate list
    y -- the y coordinate list    
    """
    p = Proj("+proj=utm +zone="+str(zone)+" +"+hemisphere+" +ellps=WGS84 +datum=WGS84 +units=m +no_defs")
    x, y = p(long, lat)
    return x, y 


# ----------------------------------------------------------------------------------------
def dip_correct_elev(x, y, z, dip, dip_dir):
    """
    A function to correct elevation values for a given dip and dip direction. This function
    takes in a set of x, y, and z coordinates and a dip and dip direction and returns a set
    of x, y, and z coordinates that are corrected for the dip and dip direction. 

    Keyword arguments:
    x -- the x coordinate list
    y -- the y coordinate list
    z -- the z coordinate list
    dip -- the dip of the surface, in degrees
    dip_dir -- the dip direction of the surface, in degrees

    Returns:
    x -- the x coordinate list
    y -- the y coordinate list
    z -- the z coordinate list
    """
    # start by identifying the southern most point as the origin
    origin_ind = np.argmin(y)

    # correct points such that orrigin is 0
    y_corrected = y - y[origin_ind]
    x_corrected = x - x[origin_ind]

    # assemple into an array
    for_rotation = np.array([x_corrected, y_corrected, z])

    # convert dip direction and dip to radians
    dip_rad = np.radians(dip)
    dip_dir_rad = np.radians(dip_dir)

    # rotate the points to the z axis so that the dip points exactly north
    z_rotation_mat = np.array([[np.cos(dip_dir_rad), -np.sin(dip_dir_rad), 0],
                                 [np.sin(dip_dir_rad), np.cos(dip_dir_rad), 0],
                                 [0, 0, 1]])

    points_prime = np.dot(z_rotation_mat, for_rotation)

    # rotate the points to the x axis so that the dip points exactly horizontal
    x_rotation_mat = np.array([[1, 0, 0],
                                 [0, np.cos(dip_rad), -np.sin(dip_rad)],
                                 [0, np.sin(dip_rad), np.cos(dip_rad)]])
    points_double_prime = np.dot(x_rotation_mat, points_prime)

    # subtract the minimum new z value from all z values to make the lowest point 0
    z_corrected = points_double_prime[2] - np.min(points_double_prime[2])
    x_corrected = points_double_prime[0]
    y_corrected = points_double_prime[1]

    return x_corrected, y_corrected, z_corrected  

# ----------------------------------------------------------------------------------------
def im_grid(im_bounds, im_dims):
    """ 
    return a grid of x and y coordinates for each pixel in an image given the bounds and 
    dimensions 
    
    Keyword arguments:
    im_bounds -- a list of the utm boundary coordinates of the image in the form 
    [im_left, im_right, im_bottom, im_top]
    im_dims -- a list of dimensions of the image in the form [im_width, im_height]

    Returns:
    x_grid -- a grid of x coordinates for each pixel in the image
    y_grid -- a grid of y coordinates for each pixel in the image
    """

    x_vec = np.linspace(im_bounds[0], im_bounds[1], im_dims[0]+1)
    y_vec = np.linspace(im_bounds[3], im_bounds[2], im_dims[1]+1)
    x_grid, y_grid = np.meshgrid(x_vec, y_vec)
    return x_grid, y_grid