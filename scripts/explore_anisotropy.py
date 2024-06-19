# script with sections to explore the anisotropy metric, starting with ichnofabric images
# written by R. A. Manzuk 04/22/2024
# last updated 05/07/2024

##########################################################################################
# package imports
##########################################################################################
# %%
import matplotlib.pyplot as plt # for plotting
import numpy as np # for numerical operations
import skimage.io as io # for image io
from skimage.filters import gaussian # for smoothing the image
import matplotlib # for custom color maps
import os # for file operations
#%%
##########################################################################################
# local function imports
##########################################################################################
# %% 
from im_processing import im_grad, grad_r
from custom_plotting import rayleigh_r_visual

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

# %% set up the file and load the data

# set the file path
ichnofab_dir = '/Volumes/ryan_ims/sms_images/icnofabric_inds'

# make a list of the files in the directory, remove any hidden files
ichnofab_files = os.listdir(ichnofab_dir)
ichnofab_files = [file for file in ichnofab_files if file[0] != '.']

# load images into a dictionary, so we can note the ichnofabric number
ichnofab_dict = {}
for file in ichnofab_files:
    # get the ichnofabric number
    ichnofab_num = file.split('.')[0]
    # load the image
    ichnofab_dict[ichnofab_num] = io.imread(os.path.join(ichnofab_dir, file))

# set up an output directory for images
output_dir = '/Users/ryan/Dropbox (Princeton)/figures/reef_survey/anisotropy_intuition'

# %% calculate the gradient of the images, and show the gradient magnitude, direction, and a polar plot of the gradient direction

grad_mag_dict = {}
grad_dir_dict = {}
for ichnofab_num, im in ichnofab_dict.items():
    grad_mag_dict[ichnofab_num], grad_dir_dict[ichnofab_num] = im_grad(im, 'sobel')

# calculate the rayleigh r for each image, only taking in the top 90% of gradient magnitudes
rayleigh_r_dict = {}
for ichnofab_num, grad_mag in grad_mag_dict.items():
    rayleigh_r_dict[ichnofab_num], _ = grad_r(grad_dir_dict[ichnofab_num], mag_thresh=90, mag_im=grad_mag)

# force the gradient directions to be between 0 and pi, and then double them so they are between 0 and 2pi
for ichnofab_num, grad_dir in grad_dir_dict.items():
    grad_dir[grad_dir < 0] = grad_dir[grad_dir < 0] + np.pi
    grad_dir_dict[ichnofab_num] = grad_dir * 2

# roll through each image, plot its visual, and save it as a pdf
for ichnofab_num, grad_dir in grad_mag_dict.items():
    # and we'll only pass through the directions associated with the highest 90% of gradient magnitudes
    highest_90_inds = grad_mag_dict[ichnofab_num] > np.percentile(grad_mag_dict[ichnofab_num], 90)

    rayleigh_r_visual(grad_dir_dict[ichnofab_num][highest_90_inds])
    plt.savefig(os.path.join(output_dir, ichnofab_num + '_rayleigh_r.pdf'))
    plt.close()

# %% make a contour plot of ichnofabric 1, smoothed

ichno1_smooth = gaussian(ichnofab_dict['1'], sigma=20)

plt.contour(ichno1_smooth, cmap='Spectral', linewidths=3, levels=6)
plt.axis('off')
plt.colorbar()
plt.show()

# and save the figure
#plt.savefig(os.path.join(output_dir, 'ichnofabric_1_contour.pdf'))

# %% do the same thing but for ichnofabric 5

ichno5_smooth = gaussian(ichnofab_dict['5'], sigma=15)

plt.contour(ichno5_smooth, cmap='Spectral', linewidths=3, levels=6)
plt.axis('off')

# and save the figure
plt.savefig(os.path.join(output_dir, 'ichnofabric_5_contour.pdf'))

