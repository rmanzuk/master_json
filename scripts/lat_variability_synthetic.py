# script with sections to make a synthetic experiment to explore the impact of lateral variability
# on the preserved signal in a stratigraphic section
# written by R. A. Manzuk 06/07/2024
# last updated 06/07/2024

##########################################################################################
# package imports
##########################################################################################
# %%
import matplotlib.pyplot as plt # for plotting
import numpy as np # for numerical operations
import matplotlib # for custom color maps
import cv2 # for image processing
import os # for file operations
from scipy import ndimage # for image processing
#%%
##########################################################################################
# local function imports
##########################################################################################
# %% 


# %%
##########################################################################################
# script lines
##########################################################################################
# %% set up some plotting stuff

# define a default color order for plotting, from Paul Tol's "Colour Schemes"
# https://personal.sron.nl/~pault/
# and we'll use the same colors for the same things throughout the paper
indigo = '#332288'
cyan = '#88CCEE'
teal = '#44AA99'
green = '#117733'
olive = '#999933'
sand = '#DDCC77'
rose = '#CC6677'
wine = '#882255'
purple = '#AA4499'

muted_colors = [rose, indigo, sand, green, cyan, wine, teal, olive, purple]

# set the muted colors as the default color cycle
muted_cmap = matplotlib.colors.ListedColormap(muted_colors)
plt.rcParams['axes.prop_cycle'] = plt.cycler(color=muted_cmap.colors)

# and turn the grid on by default, with thin dotted lines
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.linestyle'] = ':'
plt.rcParams['grid.linewidth'] = 0.5

# make fonts work out for Adobe Illustrator
plt.rcParams['pdf.fonttype'] = 42

# %% first thing we need is a random walk signal

# set the number of time steps
n_steps = 500

# set the number of signal realizations
n_realizations = 1

# set the min max range of the random walk
range_rw = [-10, 10]

# make a time vector
time = np.arange(n_steps)

# initialize the random walks at time step 0 with a random values
random_walk = np.random.uniform(low=range_rw[0], high=range_rw[1], size=n_realizations)

# complete the random walk
for i in range(1, n_steps):
    random_walk = np.vstack((random_walk, random_walk[-1] + np.random.uniform(-1, 1, n_realizations)))

# scale each random walk to the desired range
random_walk = random_walk - np.min(random_walk, axis=0)
random_walk = random_walk / np.max(random_walk, axis=0) * (range_rw[1] - range_rw[0]) + range_rw[0]

# plot the random walk, just to see what it looks like
plt.figure()
plt.plot(time, random_walk)
plt.xlabel('Time')
plt.ylabel('Random Walk Value')
plt.title('Random Walk Signal')
plt.show()

# %% now, based on the random walk, make a set of tiles that represent a platform uniformly recording the signal

tile_width = 250
tile_height = 250

# and we can just grab the random walk signal and expand it to the size of the tiles
rw_oned = np.squeeze(random_walk[:,0])
tiles = np.tile(rw_oned, (tile_height, tile_width, 1))

# %% going to define a couple of functions for modeling the lateral variability, just so we can easily make a few model runs

def get_object_centers(n_objects, object_max_verts, object_max_scale, object_max_move, variability_range, n_time_steps):
    """
    This function uses some input parameters about the desired objects to place on the tiles to generate
    random polygons and returns their transfer function, polygons, and center locations throughout the run

    Keyword arguments:
    n_objects -- the number of objects to place on the tiles
    object_max_verts -- the maximum number of vertices for the objects, as polygons
    object_max_scale -- the maximum scale of the objects, as a radius
    object_max_move -- the maximum distance the objects can move in a time step
    variability_range -- the range of values the objects can apply to the signal
    n_time_steps -- the number of time steps the model will be run for

    Returns:
    object_transfer_functions -- the transfer functions of the objects
    object_polygons -- the polygons of the objects
    object_centers_x -- the x coordinates of the object centers over time
    object_centers_y -- the y coordinates of the object centers over time
    """

    # generate the object transfer functions as random pulls from a uniform distribution
    object_transfer_functions = np.random.uniform(low=variability_range[0], high=variability_range[1], size=(n_objects,1))

    # generate random polygons to represent the objects
    object_polygons = []
    for i in range(n_objects):
        # set the number of vertices
        n_verts = np.random.randint(3, object_max_verts)

        # we'll get the vertices as random, seqential polar coordinates
        object_thetas = np.random.uniform(0, 2*np.pi, n_verts)
        object_thetas = np.sort(object_thetas)
        object_rs = np.random.uniform(0, object_max_scale, n_verts)

        # and convert to cartesian coordinates
        object_xs = np.cos(object_thetas) * object_rs
        object_ys = np.sin(object_thetas) * object_rs

        # and we'll make the polygon
        object_polygons.append(np.column_stack((object_xs, object_ys)))

    # generate some random starting coordinates for the objects
    object_starts_x = np.random.randint(0, tile_width + object_max_scale/2, n_objects)
    object_starts_y = np.random.randint(0, tile_height + object_max_scale/2, n_objects)

    # now, for each time step from 1 to the end, we want some random movement vectors, always headed towards the upper left
    object_movement_x = np.random.uniform(-object_max_move, 0, (n_objects, n_time_steps-1))
    object_movement_y = np.random.uniform(-object_max_move, 0, (n_objects, n_time_steps-1))

    # need to round the movement vectors to integers
    object_movement_x = np.round(object_movement_x)
    object_movement_y = np.round(object_movement_y)

    # based upon the movement vectors, we can generate a set of object ceter coordinates over time
    object_centers_x = [object_starts_x]
    object_centers_y = [object_starts_y]
    for i in range(0, n_time_steps-1):
            
            # extract the previous center coordinates
            prev_x = object_centers_x[-1]
            prev_y = object_centers_y[-1]
    
            # make the new center coordinates
            new_x = prev_x + object_movement_x[:,i]
            new_y = prev_y + object_movement_y[:,i]
    
            # make sure the new coordinates are within the max scale of the upper left corner. If not, respawn the object opposite where it went out
            new_x[new_x + object_max_scale/2 < 0] = tile_width + object_max_scale/2
            new_y[new_y + object_max_scale/2 < 0] = tile_height + object_max_scale/2
    
            # append the new coordinates
            object_centers_x = np.vstack((object_centers_x, new_x))
            object_centers_y = np.vstack((object_centers_y, new_y))

    return object_transfer_functions, object_polygons, object_centers_x, object_centers_y


def apply_objects_to_tiles(base_tiles, object_transfer_functions, object_polygons, object_centers_x, object_centers_y):
    """
    This function takes a set of tiles and applies objects to them based on the input parameters

    Keyword arguments:
    base_tiles -- the set of uniform tiles to apply the objects to
    object_transfer_functions -- the transfer functions of the objects
    object_polygons -- the polygons of the objects
    object_centers_x -- the x coordinates of the object centers over time
    object_centers_y -- the y coordinates of the object centers over time

    Returns:
    tiles_varied -- the set of tiles with the objects applied
    """

    # make a copy of the base tiles
    tiles_varied = np.copy(base_tiles)

    # get the number of objects and time steps
    n_objects = object_transfer_functions.shape[0]
    n_time_steps = object_centers_x.shape[0]

    # roll through each time step, place the objects
    for i in range(n_time_steps):
        for j in range(n_objects):
            # get the object polygon, its center, and transfer function
            object_center = np.array([object_centers_x[i,j], object_centers_y[i,j]])
            object_tf = object_transfer_functions[j]
            object_polygon = object_polygons[j] + object_center

            # make a mask of the object
            object_mask = np.zeros((tile_height, tile_width))
            object_mask = cv2.fillPoly(object_mask, [object_polygon.astype(np.int32)], 1)

            # apply the transfer function to the object
            object_mask = object_mask * object_tf

            # apply the object to the tile
            tiles_varied[:,:,i] = tiles_varied[:,:,i] + object_mask

        # print that we finished a time step
        print('Finished time step ' + str(i))

    return tiles_varied

# %% now we'll run the model a few times with different parameters

# I want to keep variable names simple, so I'll just make a key for them right here
# exp1 - relatively low variability introduced by the objects
# exp2 - relatively high variability introduced by the objects, but otherwise the same as exp1
# exp3 - variability remaining high, but make the objects move more per step
# exp4 - variability remaining high, but make the objects smaller (but increase their number so they have the same coverage)

###### parameters for exp1 ######
exp1_n_objects = 80
exp1_object_max_verts = 6
exp1_object_max_scale = 0.7 * tile_width
exp1_object_max_move = 0.01 * tile_width
exp1_variability_scale = 0.2
exp1_variability_range = [exp1_variability_scale * range_rw[0], exp1_variability_scale * range_rw[1]]

# and run dat model
print('Running exp1')
exp1_object_transfer_functions, exp1_object_polygons, exp1_object_centers_x, exp1_object_centers_y = get_object_centers(exp1_n_objects, exp1_object_max_verts, exp1_object_max_scale, exp1_object_max_move, exp1_variability_range, n_steps)
tiles_varied_exp1 = apply_objects_to_tiles(tiles.copy(), exp1_object_transfer_functions, exp1_object_polygons, exp1_object_centers_x, exp1_object_centers_y)

###### parameters for exp2 ######
exp2_n_objects = 80
exp2_object_max_verts = 6
exp2_object_max_scale = 0.7 * tile_width
exp2_object_max_move = 0.01 * tile_width
exp2_variability_scale = 0.8
exp2_variability_range = [exp2_variability_scale * range_rw[0], exp2_variability_scale * range_rw[1]]

# and run dat model
print('Running exp2')
exp2_object_transfer_functions, exp2_object_polygons, exp2_object_centers_x, exp2_object_centers_y = get_object_centers(exp2_n_objects, exp2_object_max_verts, exp2_object_max_scale, exp2_object_max_move, exp2_variability_range, n_steps)
tiles_varied_exp2 = apply_objects_to_tiles(tiles.copy(), exp2_object_transfer_functions, exp2_object_polygons, exp2_object_centers_x, exp2_object_centers_y)

###### parameters for exp3 ######
exp3_n_objects = 80
exp3_object_max_verts = 6
exp3_object_max_scale = 0.7 * tile_width
exp3_object_max_move = 0.05 * tile_width
exp3_variability_scale = 0.8
exp3_variability_range = [exp3_variability_scale * range_rw[0], exp3_variability_scale * range_rw[1]]

# and run dat model
print('Running exp3')
exp3_object_transfer_functions, exp3_object_polygons, exp3_object_centers_x, exp3_object_centers_y = get_object_centers(exp3_n_objects, exp3_object_max_verts, exp3_object_max_scale, exp3_object_max_move, exp3_variability_range, n_steps)
tiles_varied_exp3 = apply_objects_to_tiles(tiles.copy(), exp3_object_transfer_functions, exp3_object_polygons, exp3_object_centers_x, exp3_object_centers_y)

###### parameters for exp4 ######
exp4_n_objects = 560
exp4_object_max_verts = 6
exp4_object_max_scale = 0.1 * tile_width
exp4_object_max_move = 0.01 * tile_width   
exp4_variability_scale = 0.8
exp4_variability_range = [exp4_variability_scale * range_rw[0], exp4_variability_scale * range_rw[1]]

# and run dat model
print('Running exp4')
exp4_object_transfer_functions, exp4_object_polygons, exp4_object_centers_x, exp4_object_centers_y = get_object_centers(exp4_n_objects, exp4_object_max_verts, exp4_object_max_scale, exp4_object_max_move, exp4_variability_range, n_steps)
tiles_varied_exp4 = apply_objects_to_tiles(tiles.copy(), exp4_object_transfer_functions, exp4_object_polygons, exp4_object_centers_x, exp4_object_centers_y)

# %% time for plotting

# set a color map for the tiles
cmap = 'Spectral'

# frist let's export 5, evenly spaced tiles from the each experiment, as well as the base tiles
# set some paths
general_path = '/Users/ryan/Dropbox (Princeton)/figures/reef_survey/spatial_var_synthetic/'
base_sub = 'base_tiles'
exp1_sub = 'exp1_tiles'
exp2_sub = 'exp2_tiles'
exp3_sub = 'exp3_tiles'
exp4_sub = 'exp4_tiles'

# set up the time steps we will sample
sampled_times = np.linspace(0, 50, 5, dtype=int)

# get the max and min min values of all tile sets so we can normalize the color scales
all_tile_max = np.max([np.max(tiles_varied_exp1), np.max(tiles_varied_exp2), np.max(tiles_varied_exp3), np.max(tiles_varied_exp4)])
all_tile_min = np.min([np.min(tiles_varied_exp1), np.min(tiles_varied_exp2), np.min(tiles_varied_exp3), np.min(tiles_varied_exp4)])

# roll through and save the tiles as pngs
for i in range(len(sampled_times)):

    # make the file names
    base_name = general_path + base_sub + '/base_tile_' + str(i) + '.png'
    exp1_name = general_path + exp1_sub + '/exp1_tile_' + str(i) + '.png'
    exp2_name = general_path + exp2_sub + '/exp2_tile_' + str(i) + '.png'
    exp3_name = general_path + exp3_sub + '/exp3_tile_' + str(i) + '.png'
    exp4_name = general_path + exp4_sub + '/exp4_tile_' + str(i) + '.png'

    # make and save all the tiles
    plt.imsave(base_name, tiles[:,:,sampled_times[i]], cmap=cmap, vmin=all_tile_min, vmax=all_tile_max)
    plt.imsave(exp1_name, tiles_varied_exp1[:,:,sampled_times[i]], cmap=cmap, vmin=all_tile_min, vmax=all_tile_max)
    plt.imsave(exp2_name, tiles_varied_exp2[:,:,sampled_times[i]], cmap=cmap, vmin=all_tile_min, vmax=all_tile_max)
    plt.imsave(exp3_name, tiles_varied_exp3[:,:,sampled_times[i]], cmap=cmap, vmin=all_tile_min, vmax=all_tile_max)
    plt.imsave(exp4_name, tiles_varied_exp4[:,:,sampled_times[i]], cmap=cmap, vmin=all_tile_min, vmax=all_tile_max)

# %% now we'll make a plot of the signal variability over time for the base signal and the four experiments
    
# set up where we will sample the stacks
sampled_row_ind = 25
sampled_col_ind = 125

# get the max and min of all signals so we can set the y limits
all_signal_max = np.max([np.max(tiles_varied_exp1[sampled_row_ind, sampled_col_ind, :]), np.max(tiles_varied_exp2[sampled_row_ind, sampled_col_ind, :]), np.max(tiles_varied_exp3[sampled_row_ind, sampled_col_ind, :]), np.max(tiles_varied_exp4[sampled_row_ind, sampled_col_ind, :])])
all_signal_min = np.min([np.min(tiles_varied_exp1[sampled_row_ind, sampled_col_ind, :]), np.min(tiles_varied_exp2[sampled_row_ind, sampled_col_ind, :]), np.min(tiles_varied_exp3[sampled_row_ind, sampled_col_ind, :]), np.min(tiles_varied_exp4[sampled_row_ind, sampled_col_ind, :])])


# set up the figure, we'll do 4 long skinny subplots
fig, axs = plt.subplots(4, 1, figsize=(8, 4))

# we'll plot each experimental signal, with the base signal in the background
axs[0].plot(time, tiles_varied_exp1[sampled_row_ind, sampled_col_ind, :], label='Exp 1', linewidth=2)
axs[0].plot(time, tiles[sampled_row_ind, sampled_col_ind, :], label='Base', linewidth=2)

axs[1].plot(time, tiles_varied_exp2[sampled_row_ind, sampled_col_ind, :], label='Exp 2', linewidth=2)
axs[1].plot(time, tiles[sampled_row_ind, sampled_col_ind, :], label='Base', linewidth=2)

axs[2].plot(time, tiles_varied_exp3[sampled_row_ind, sampled_col_ind, :], label='Exp 3', linewidth=2)
axs[2].plot(time, tiles[sampled_row_ind, sampled_col_ind, :], label='Base', linewidth=2)

axs[3].plot(time, tiles_varied_exp4[sampled_row_ind, sampled_col_ind, :], label='Exp 4', linewidth=2)
axs[3].plot(time, tiles[sampled_row_ind, sampled_col_ind, :], label='Base', linewidth=2)

# set the y limits
axs[0].set_ylim(all_signal_min, all_signal_max)
axs[1].set_ylim(all_signal_min, all_signal_max)
axs[2].set_ylim(all_signal_min, all_signal_max)
axs[3].set_ylim(all_signal_min, all_signal_max)

# set the x limits
axs[0].set_xlim(0, n_steps)
axs[1].set_xlim(0, n_steps)
axs[2].set_xlim(0, n_steps)
axs[3].set_xlim(0, n_steps)

# and we can remove the x ticks on all
axs[0].set_xticks([])
axs[1].set_xticks([])
axs[2].set_xticks([])
axs[3].set_xticks([])

# set the labels and titles
axs[3].set_xlabel('Time')
axs[0].set_ylabel('Signal Value')
axs[1].set_ylabel('Signal Value')
axs[2].set_ylabel('Signal Value')
axs[3].set_ylabel('Signal Value')

plt.tight_layout()


# save the figure
fig.savefig(general_path + 'signal_variability_over_time.pdf')
