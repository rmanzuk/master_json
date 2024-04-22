# script with sections to explore the anisotropy metric for a given sample image
# written by R. A. Manzuk 04/22/2024
# last updated 04/22/2024

##########################################################################################
# package imports
##########################################################################################
# %%
import matplotlib.pyplot as plt # for plotting
import numpy as np # for numerical operations
import skimage.io as io # for image io
from scipy.ndimage import rotate # for rotating the image
from skimage.filters import gaussian # for smoothing the image
from skimage.transform import rescale # for rescaling images
import matplotlib # for custom color maps
#%%
##########################################################################################
# local function imports
##########################################################################################
# %% 
from im_processing import im_grad, grad_r

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
input_file = '/Users/ryan/Downloads/grotzinger1995.fig6c.tiff'

# load the data
im = io.imread(input_file)

# in this case, I know we just want the first channel
im = im[:,:,0]

# and make the image a float between 0 and 1 based on its data type
if im.dtype == 'uint8':
    im = im / 255
elif im.dtype == 'uint16':
    im = im / 65535
else:
    print('Image data type not recognized. Please provide an image with data type uint8 or uint16.')

# %% the image has some weird artifacts, so we can just smooth it a bit
gaus_sigma = 2
im = gaussian(im, gaus_sigma)
# %% calculate the gradient of the image, and show the gradient magnitude, direction, and a polar plot of the gradient direction
grad_mag, grad_dir = im_grad(im, 'sobel')

# our gradient currently ranges from -pi to pi, but we only want it to range from 0 to pi. 
# for negative values, they should be equal to their inverse, so -p/4 should be equal to 3pi/4
grad_dir[grad_dir < 0] = grad_dir[grad_dir < 0] + np.pi

# make 3 subplots
fig = plt.figure(figsize = (15, 5))
ax1 = plt.subplot(131)
ax2 = plt.subplot(132)
ax3 = plt.subplot(133, projection = 'polar', xlim = (0, 180))
ax3.set_theta_direction(-1)
ax3.set_theta_zero_location('N')
ax3.set_thetamin(0)
ax3.set_thetamax(180)

# plot the gradient magnitude
ax1.imshow(grad_mag, cmap = 'gray')
ax1.set_title('Gradient Magnitude')
plt.colorbar(ax1.imshow(grad_mag, cmap = 'gray'), ax = ax1)

# plot the gradient direction
ax2.imshow(grad_dir, cmap = 'hsv')
ax2.set_title('Gradient Direction')
plt.colorbar(ax2.imshow(grad_dir, cmap = 'hsv'), ax = ax2)

# make the bin edges and bind the gradient directions to the bins
bin_edges = np.linspace(0, np.pi, 100)
hist, _ = np.histogram(grad_dir, bins = bin_edges)
# plot the histogram
ax3.bar(bin_edges[:-1], hist, width = np.diff(bin_edges)*2, bottom = 0)
ax3.set_title('Gradient Direction Histogram')

# show the plot
plt.show()

# %% now we can calculate the anisotropy metric

# recalculate the gradient magnitude and direction so there are no adjustments
grad_mag, grad_dir = im_grad(im, 'sobel')

# we can calculate a few globally with with different parameters
global_unweighted_r, global_unweighted_theta = grad_r(grad_dir)
global_weighted_r, global_weighted_theta = grad_r(grad_dir, mag_weights=grad_mag)
global_90pct_r, global_90pct_theta = grad_r(grad_dir, mag_thresh=90, mag_im=grad_mag)

# make 2 subplots, the first will have few polar histograms of the gradient directions, and the second will have the anisotropy metric
# and first redistribute the gradient directions to be between 0 and pi
grad_dir[grad_dir < 0] = grad_dir[grad_dir < 0] + np.pi
fig = plt.figure(figsize = (15, 5))
ax1 = plt.subplot(131, projection = 'polar', xlim = (0, 180))
ax1.set_theta_direction(-1)
ax1.set_theta_zero_location('N')
ax1.set_thetamin(0)
ax1.set_thetamax(180)
ax2 = plt.subplot(132)

# make the bin edges, and bind the gradient directions to the bins
bin_edges = np.linspace(0, np.pi, 100)
hist, _ = np.histogram(grad_dir, bins = bin_edges)

# make a weighted verstion of the histogram, first maek the average gradient magnitude = 1
grad_mag_weight = grad_mag / np.mean(grad_mag)
weighted_hist, _ = np.histogram(grad_dir, bins = bin_edges, weights = grad_mag_weight)

# and make one that only includes the top 90% of gradient magnitudes
top_90 = np.percentile(grad_mag, 90)
top_90_hist, _ = np.histogram(grad_dir[grad_mag > top_90], bins = bin_edges)

# plot the histograms
ax1.bar(bin_edges[:-1], hist, width = np.diff(bin_edges)*2, bottom = 0, label = 'Unweighted')
ax1.bar(bin_edges[:-1], weighted_hist, width = np.diff(bin_edges)*2, bottom = 0, label = 'Weighted')
ax1.bar(bin_edges[:-1], top_90_hist, width = np.diff(bin_edges)*2, bottom = 0, label = 'Top 90%')
ax1.set_title('Gradient Direction Histograms')
ax1.legend()

# plot the anisotropy metric
ax2.bar(['Unweighted', 'Weighted', 'Top 90%'], [global_unweighted_r, global_weighted_r, global_90pct_r])

# %% make a few neighborhood-wise and showthem

# make a few neighborhood radii
radii = [5, 10, 20]

# set up the array to hold the anisotropy metrics and angles
neighborhood_r = np.zeros((im.shape[0], im.shape[1], len(radii)))
neighborhood_theta = np.zeros((im.shape[0], im.shape[1], len(radii)))

# loop through the radii and calculate the anisotropy metrics
for i, radius in enumerate(radii):
    neighborhood_r[:,:,i], neighborhood_theta[:,:,i] = grad_r(grad_dir, neigh_rad=radius)

# make a few subplots
fig = plt.figure(figsize = (15, 5))
ax1 = plt.subplot(131)
ax2 = plt.subplot(132)
ax3 = plt.subplot(133)

# plot the anisotropy metrics
ax1.imshow(neighborhood_r[:,:,0], cmap = 'viridis')
ax1.set_title('Neighborhood Radius = 5')
plt.colorbar(ax1.imshow(neighborhood_r[:,:,0], cmap = 'viridis'), ax = ax1)

ax2.imshow(neighborhood_r[:,:,1], cmap = 'viridis')
ax2.set_title('Neighborhood Radius = 10')
plt.colorbar(ax2.imshow(neighborhood_r[:,:,1], cmap = 'viridis'), ax = ax2)

ax3.imshow(neighborhood_r[:,:,2], cmap = 'viridis')
ax3.set_title('Neighborhood Radius = 20')
plt.colorbar(ax3.imshow(neighborhood_r[:,:,2], cmap = 'viridis'), ax = ax3)

# show the plot
plt.tight_layout()
plt.show()

# %% last thing is to calculate it while resizing the image

# make a few resizing factors
im_scales = [1,0.5,0.25,0.125,0.0625,0.03125,0.015625,0.0078125]
# set up the array to hold the anisotropy metrics and angles
resize_r = np.zeros((len(im_scales)))
resize_theta = np.zeros((len(im_scales)))

# loop through the resizing factors and calculate the anisotropy metrics
for i, scale in enumerate(im_scales):
    # resize the image
    resized_im = rescale(im, scale)
    # calculate the gradient of the resized image
    resized_grad_mag, resized_grad_dir = im_grad(resized_im, 'sobel')
    # calculate the anisotropy metric
    resize_r[i], resize_theta[i] = grad_r(resized_grad_dir)

# make a plot of the anisotropy metric vs scale, with the x-axis on a log scale
plt.plot(im_scales, resize_r)
plt.xscale('log')
plt.xlabel('Image scale')
plt.ylabel('Anisotropy Metric')
plt.title('Anisotropy Metric vs Scale Factor')
# but still maek the x-axis ticks the actual scale factors
plt.xticks(im_scales, im_scales)

plt.show()
