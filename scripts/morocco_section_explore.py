# script with sections look through a section from the Cambrian of morocco and see if there
# are any associations between carbon isotopes and basic facies types
# written by R. A. Manzuk 06/14/2024
# last updated 06/14/2024

##########################################################################################
# package imports
##########################################################################################
# %%
import matplotlib.pyplot as plt # for plotting
import numpy as np # for numerical operations
import matplotlib # for custom color maps
import pandas as pd # for handling dataframes
from sklearn.decomposition import PCA
import cv2

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

# %% set up paths and read the data

# define the path to the folder with csvs, can subselect a specific section from there
section_path = '/Users/ryan/Dropbox (Princeton)/reef_survey_project/morocco_sections/'

# comment out the sections you don't want to look at
section_name = 'Msal_all.csv'
# section_name = 'Sdas_all.csv'
# section_name = 'Yissi_all.csv'
# section_name = 'Zawyat_all.csv'

# read in the data, with no column names
section_data = pd.read_csv(section_path + section_name, header=None)

# it doesn't come with column names, so let's add those
col_headers = ['delta13C', 'height', 'mineralogy', 'facies']
section_data.columns = col_headers

# also merge the facies and mineralogy columns
section_data['mineralogy'] = section_data['mineralogy'].astype(str)
section_data['facies'] = section_data['mineralogy'] + '_' + section_data['facies']

# %% we're going to make some facies groupings manually

microbialites = ['dl_mi', 'ls_mi']
grainstones = ['dl_gr', 'ls_gr', 'dl_fn', 'gl_gr', 'dl_br','dl_R']
sromatolites = ['dl_st', 'ls_st']
thrombolites = ['dl_th', 'ls_th']
siliciclastics = ['si_sts', 'ml_sts', 'qa_fss', 'qa_vfss', 'ar_vfss']
ribbonites = ['dl_ri', 'mi_ri', 'ml_ri', 'ls_ri']
others = ['cover_cover', 'ash_ash', 'ash_sts', 'ch_ch']

# add a new column that designates the facies group
facies_group = []
for i in range(len(section_data)):
    this_facies = section_data['facies'].iloc[i]
    if this_facies in microbialites:
        facies_group.append('microbialites')
    elif this_facies in grainstones:
        facies_group.append('grainstones')
    elif this_facies in sromatolites:
        facies_group.append('stromatolites')
    elif this_facies in thrombolites:
        facies_group.append('thrombolites')
    elif this_facies in siliciclastics:
        facies_group.append('siliciclastics')
    elif this_facies in ribbonites:
        facies_group.append('ribbonites')
    else:
        facies_group.append('others')

section_data['facies_group'] = facies_group
# %% look in moving windows and judge the proportions of facies groups

# set the window size
window_size = 40 # that's in meters

# set up a one hot matrix for the facies types that's n_facies x len(section_data)
facies_one_hot = pd.get_dummies(section_data['facies_group']).T

# to do the moving window sum, we need bed thicknesses, so let's get those
bed_thicknesses = section_data['height'].diff().fillna(0)

# because stratigraphic height isn't evenly sampled, we'll have to do some tricks to get the moving window sum
# first, we'll multiply the one hot matrix by the bed thicknesses
facies_one_hot = facies_one_hot.mul(bed_thicknesses, axis=1)

# we'll reorder the row index how we want it
row_index = ['microbialites', 'stromatolites', 'thrombolites', 'grainstones',  'ribbonites', 'siliciclastics', 'others']
facies_one_hot = facies_one_hot.loc[row_index]

# and then for every bed, we need a binary row that's 1 for all the beds in its window
# we'll do this by making a boolean matrix that's n_beds x n_beds, that's 1 if column is in the row's window
beds_in_window = np.zeros((len(section_data), len(section_data)))

# we can fill this in a loop, won't be too bad
for i in range(len(section_data)):
    # start by getting the difference between this height and all the other heights
    height_diffs = section_data['height'] - section_data['height'].iloc[i]
    # and then we'll make a boolean array that's true if the height difference is less than half the window size
    in_window = np.abs(height_diffs) < window_size/2
    # and then we'll put that into the matrix
    beds_in_window[i, :] = in_window

# now we just have to sum up the one hot matrix, informed by the beds in window matrix
facies_sums = np.dot(facies_one_hot, beds_in_window)

# make all rows sum to 1 to have proportions
facies_props = facies_sums / facies_sums.sum(axis=0)

# do a moving window correlation between all facies groups and the carbon isotopes
corr_window_size = 50

# make a new dataframe that's just the carbon isotopes and the facies proportions
facies_props_df = pd.DataFrame(facies_props.T, columns=facies_one_hot.index)
for_correlation = section_data[['delta13C']]
for_correlation = pd.concat([for_correlation, facies_props_df], axis=1)

corr_matrix = np.zeros((len(section_data), len(facies_props)))
# need to keep in mind the variable sampling of height
for i in range(len(section_data)):
    # get the carbon isotopes and facies proportions for this window
    this_window = for_correlation.iloc[i-corr_window_size//2:i+corr_window_size//2]
    # and then calculate the correlation
    this_corr = this_window.corr()
    # and store the correlation in a matrix
    corr_matrix[i, :] = this_corr['delta13C'].values[1:]

# %% one last thing before plotting, we want to give each facies group a color from the 'spectral' colormap

# get the raw color map values
spectral_cmap = plt.get_cmap('Spectral')
spectral_colors = spectral_cmap(np.linspace(0, 1, len(facies_one_hot.index)))
# %% plt the carbon isotopes and the first few pcs, and the correlation between them

# set up the figure
fig, ax = plt.subplots(3, 1, figsize=(10,6), sharex=True)

# get the min and max heights for x limits
min_height = section_data['height'].min()
max_height = section_data['height'].max()

# plot the carbon isotopes, as open circles
ax[0].scatter(section_data['height'], section_data['delta13C'], s=10, facecolors='none', edgecolors='k')

# plot the facies proportions as a stacked area plot, with the colors from the spectral colormap
ax[1].stackplot(section_data['height'], facies_props, colors=spectral_colors, labels=facies_one_hot.index)

# and then make a line plot where the line for each facies is solid, horizontal line when the absolute value of its correlation with the carbon isotopes is above 0.5
# and dashed when it's below 0.5
high_correlation_bool = np.abs(corr_matrix) > 0.5

# we need the intervals of true for the boolean
# make the boolean not a df
high_correlation_bool = high_correlation_bool.astype(int)
truth_spikes = np.diff(high_correlation_bool, axis=0)
truth_starts = np.argwhere(truth_spikes == 1)
truth_ends = np.argwhere(truth_spikes == -1)

# based upon the second argument of each start and end, sort them into dictionaries where the starts and ends are separated by facies
start_dict = {}
end_dict = {}
for i in range(len(row_index)):
    start_dict[row_index[i]] = []
    end_dict[row_index[i]] = []

for i in range(len(truth_starts)):
    start_dict[row_index[truth_starts[i][1]]].append(truth_starts[i][0])
    end_dict[row_index[truth_ends[i][1]]].append(truth_ends[i][0])

# if the number of starts and ends don't line up for a key, either add a dummy start of 0 or a dummy end of the last index
for i in range(len(row_index)):
    if len(start_dict[row_index[i]]) > len(end_dict[row_index[i]]):
        end_dict[row_index[i]].append(len(row_index)-1)
    elif len(start_dict[row_index[i]]) < len(end_dict[row_index[i]]):
        start_dict[row_index[i]].insert(0, 0)

# now plot the lines as rectangles on different y levels for each facies
for i in range(len(row_index)):
    for j in range(len(start_dict[row_index[i]])):
        this_start_height = section_data['height'].iloc[start_dict[row_index[i]][j]]
        this_end_height = section_data['height'].iloc[end_dict[row_index[i]][j]]
        # and just plot a line from the start to the end, at the y height of this index
        ax[2].plot([this_start_height, this_end_height], [i, i], linestyle='-', linewidth=4, color=spectral_colors[i])


# add some labels
ax[0].set_ylabel('Carbon Isotopes')
ax[1].set_ylabel('Facies Proportions')
ax[2].set_ylabel('Correlation with Carbon Isotopes')
ax[2].set_xlabel('Height (m)')
ax[1].legend()
ax[2].legend()

# and set the x limits
ax[0].set_xlim(min_height, max_height)
ax[1].set_xlim(min_height, max_height)
ax[2].set_xlim(min_height, max_height)

# and y limits on the stacked area plot
ax[1].set_ylim(0, 1)

plt.tight_layout()

# set the path and save the figure
fig_path = '/Users/ryan/Dropbox (Princeton)/figures/reef_survey/morocco_section_motivation'
fig_name = section_name.split('.')[0] + '_plots.pdf'
#plt.savefig(fig_path + '/' + fig_name)
plt.show()
# %% make a histogram of the delta13C values for each facies group, when it is the dominant facies

# get the dominant facies for each bed
dominant_facies = np.argmax(facies_props, axis=0)

# set up the figure
fig, ax = plt.subplots(len(row_index), 1, figsize=(10, 10), sharex=True)

# make a histogram for each facies group
for i in range(len(row_index)):
    this_facies = row_index[i]
    this_deltas = section_data[section_data['facies_group'] == this_facies]['delta13C']
    ax[i].hist(this_deltas, bins=20, alpha=0.5, label=this_facies)
    ax[i].set_ylabel(this_facies)

# add some labels
ax[0].set_title('Carbon Isotope Distributions by Facies Group')
ax[-1].set_xlabel('Carbon Isotopes')
plt.tight_layout()

plt.show()

# %% make a cross plot of range in the carbon isotopes in a window and the evenness of the facies proportions

window_size = 10 # in meters

# get the range of the carbon isotopes in the moving window
beds_in_window = np.zeros((len(section_data), len(section_data)))

# we can fill this in a loop, won't be too bad
isotope_ranges = np.zeros(len(section_data))
for i in range(len(section_data)):
    # start by getting the difference between this height and all the other heights
    height_diffs = section_data['height'] - section_data['height'].iloc[i]
    # and then we'll make a boolean array that's true if the height difference is less than half the window size
    in_window = np.abs(height_diffs) < window_size/2
    # and then we'll put that into the matrix
    this_window = section_data[in_window]
    isotope_ranges[i] = this_window['delta13C'].max() - this_window['delta13C'].min()

# and then we'll get the evenness of the facies proportions at all points
# a perfectly even distribution will have even proportions of all faces, so we'll just calculate an r squared value compared to that
even_facies_props = np.ones(facies_props.shape) / len(facies_props)
sum_of_squares = np.sum((facies_props - even_facies_props)**2, axis=0)
r_squared = 1 - sum_of_squares  

# set up the figure
fig, ax = plt.subplots(1, 1, figsize=(6, 6))

# plot the range of the carbon isotopes against the r squared value
ax.scatter(isotope_ranges, r_squared)

# add some labels
ax.set_xlabel('Range in Carbon Isotopes')
ax.set_ylabel('R Squared of Facies Proportions')

plt.show()
# %% last thing is to make a 'map' of the possible facies on the ancient platform, given their proportions

# set up the 'map' as a 2d array of zeros
map_height = 1000
map_width = 1000
facies_map = np.ones((map_height, map_width))

# we're gonna have ribbonite as the background, which is index 5
facies_map = facies_map * 5

# set up a distribution of radii for the facies groups with mean and std
# remember facies order is 'microbialites', 'stromatolites', 'thrombolites', 'grainstones','ribbonites', 'siliciclastics', 'others'
# sizes are listed as a ratio of the map height
mean_radii_ratios = [0.01, 0.01, 0.005, 0.01, 0.01, 0.15]
std_radii_ratios = [0.01, 0.01, 0.005, 0.03, 0.005, 0.05]

# and when we're making the shapes, we'll let the radius wobble with a bit of variance
radius_wobble_ratio = 1

# we also need to know the overall proportions of all facies in the section, make sure they sum to 1
overall_facies_props = facies_props.mean(axis=1)
# we're removing the 'others' facies from the map, so we'll remove it from the proportions
overall_facies_props = overall_facies_props[:-1]
overall_facies_props = overall_facies_props / overall_facies_props.sum()

# now we should be ready to populate the map, we'll do that in a loop that ends once the map is full, and the proportions are met
map_full = False
map_facies_prop_diffs = overall_facies_props.copy()
map_facies_prop_diffs[4] = map_facies_prop_diffs[4] - 1
stop_prop = 0.01

while np.any(np.abs(map_facies_prop_diffs) > stop_prop):
    
    # we'll decide which facies to place based on the one that is furthest under its proportion
    facies_to_place = np.argmax(map_facies_prop_diffs) + 1
    print(facies_to_place)

    # generate a random radius for this facies
    this_rad_mean = mean_radii_ratios[facies_to_place-1] * map_height

    # make the shape, which will start as a circle, but we'll randomly add 10 points where the radius is wobbled, and we'll interpolate between them
    all_thetas = np.linspace(0, 2*np.pi, 1000)
    perturb_thetas = np.random.choice(all_thetas, 4) 
    
    # if the start and end thetas are not in the perturb thetas, add them 
    if 0 not in perturb_thetas:
        perturb_thetas = np.append(perturb_thetas, 0)
    if 2*np.pi not in perturb_thetas:
        perturb_thetas = np.append(perturb_thetas, 2*np.pi)
    perturb_thetas = np.sort(perturb_thetas)
    perturb_radii = np.random.normal(this_rad_mean, this_rad_mean * radius_wobble_ratio, len(perturb_thetas))
    
    # make the last radius the same as the first
    perturb_radii[-1] = perturb_radii[0]
    all_radii = np.interp(all_thetas, perturb_thetas, perturb_radii)

    # and smooth the radii, padding the ends with the values from the other side
    smoothing_window = 100
    all_radii = np.pad(all_radii, (smoothing_window//2, smoothing_window//2), mode='wrap')
    all_radii = np.convolve(all_radii, np.ones(100)/100, mode='valid')
    # gotta chop off one from the end to make the lengths match
    all_radii = all_radii[:-1]

    # make the shape in cartesian coordinates
    shape_x = np.cos(all_thetas) * all_radii
    shape_y = np.sin(all_thetas) * all_radii

    # give the shape a random aspect ratio, and rotate it to a random angle
    aspect_ratio = np.random.uniform(0.5, 2)
    rotation_angle = np.random.uniform(0, 2*np.pi)
    rotation_matrix = np.array([[np.cos(rotation_angle), -np.sin(rotation_angle)], [np.sin(rotation_angle), np.cos(rotation_angle)]])
    shape = np.array([shape_x, shape_y]).T
    shape[:, 0] = shape[:, 0] * aspect_ratio
    shape = np.dot(shape, rotation_matrix)

    # we'll move the shape to have a center randomly selected from an empty part of the map
    # find the empty parts of the map
    empty_map = facies_map == 0
    empty_indices = np.argwhere(empty_map)
    if len(empty_indices) > 0:
        # randomly select one of the empty indices
        this_center = empty_indices[np.random.choice(len(empty_indices))]
        # and move the shape to that center
        shape_x = shape_x + this_center[1]
        shape_y = shape_y + this_center[0]
    else:
        # in this case we'll place it on a class that is over its proportion
        place_on = np.argmin(map_facies_prop_diffs)
        place_on_indices = np.argwhere(facies_map == place_on+1)
        this_center = place_on_indices[np.random.choice(len(place_on_indices))]

    # and move the shape to that center
    shape_x = shape_x + this_center[1]
    shape_y = shape_y + this_center[0]

    # now we have a shape, make it a polygon, and then a mask
    shape = np.array([shape_x, shape_y]).T
    mask = np.zeros((map_height, map_width))
    mask = cv2.fillPoly(mask, [shape.astype(int)], 1)

    # return the facies map values to 0 for the mask
    facies_map = facies_map * (1 - mask)

    # and then we'll add the mask to the map, and update the facies proportions
    facies_map = facies_map + mask * facies_to_place
    # need to account for the fact that not all facies may be present
    facies_sums = np.zeros(len(overall_facies_props))
    for i in range(len(overall_facies_props)):
        facies_sums[i] = np.sum(facies_map == i+1)
    map_facies_prop_diffs = overall_facies_props - facies_sums / (map_height * map_width)

    # and check if the map is full
    if np.all(facies_map > 0):
        map_full = True

    print(map_facies_prop_diffs)


# make the facies map into an rgb image with the colors associated with the facies
facies_map_rgb = np.zeros((map_height, map_width, 3))
for i in range(len(row_index)):
    facies_map_rgb[facies_map == i+1] = spectral_colors[i][:3]

# and save the facies map as a tiff
facies_map_path = '/Users/ryan/Dropbox (Princeton)/figures/reef_survey/morocco_section_motivation'
facies_map_name = section_name.split('.')[0] + '_facies_map.tiff'
plt.imsave(facies_map_path + '/' + facies_map_name, facies_map_rgb)
# %%
