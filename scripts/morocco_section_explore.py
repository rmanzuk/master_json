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

# %% look in moving windows and judge the proportions of facies types

# set the window size
window_size = 50 # that's in meters

# set up a one hot matrix for the facies types that's n_facies x len(section_data)
facies_one_hot = pd.get_dummies(section_data['facies'])

# do the moving window sum
facies_sums = facies_one_hot.rolling(window_size, center=True, min_periods=1).sum()

# divide by the window size to get proportions
facies_props = facies_sums.div(window_size, axis=0)

# get the top 5 facies types, make them a dataframe, and put all other facies types into an 'other' category
top_facies = facies_props.sum().sort_values(ascending=False).head(5).index
facies_props['other'] = 1 - facies_props[top_facies].sum(axis=1)
top_facies_and_other = top_facies.append(pd.Index(['other']))
facies_props_top_and_other = facies_props[top_facies_and_other]

# do a moving window correlation between the top 5 plus 'other' facies and the carbon isotopes
corr_window_size = 50
facies_correlations = facies_props_top_and_other.rolling(corr_window_size, center=True, min_periods=1).corr(section_data['delta13C'])

# %% plt the carbon isotopes and the first few pcs, and the correlation between them

# set up the figure
fig, ax = plt.subplots(3, 1, figsize=(10,6), sharex=True)

# get the min and max heights for x limits
min_height = section_data['height'].min()
max_height = section_data['height'].max()

# plot the carbon isotopes, as open circles
ax[0].scatter(section_data['height'], section_data['delta13C'], s=10, facecolors='none', edgecolors='k')

# plot the facies proportions as a stacked area plot
ax[1].stackplot(section_data['height'], facies_props_top_and_other.T, labels=top_facies_and_other)

# and then make a line plot where the line for each facies is solid, horizontal line when the absolute value of its correlation with the carbon isotopes is above 0.5
# and dashed when it's below 0.5
high_correlation_bool = np.abs(facies_correlations) > 0.5

# we need the intervals of true for the boolean
# make the boolean not a df
high_correlation_bool = high_correlation_bool.values.astype(int)
truth_spikes = np.diff(high_correlation_bool, axis=0)
truth_starts = np.argwhere(truth_spikes == 1)
truth_ends = np.argwhere(truth_spikes == -1)

# based upon the second argument of each start and end, sort them into dictionaries where the starts and ends are separated by facies
start_dict = {}
end_dict = {}
for i in range(len(top_facies_and_other)):
    start_dict[top_facies_and_other[i]] = []
    end_dict[top_facies_and_other[i]] = []

for i in range(len(truth_starts)):
    start_dict[top_facies_and_other[truth_starts[i][1]]].append(truth_starts[i][0])
    end_dict[top_facies_and_other[truth_ends[i][1]]].append(truth_ends[i][0])

# if the number of starts and ends don't line up for a key, either add a dummy start of 0 or a dummy end of the last index
for i in range(len(top_facies_and_other)):
    if len(start_dict[top_facies_and_other[i]]) > len(end_dict[top_facies_and_other[i]]):
        end_dict[top_facies_and_other[i]].append(len(facies_correlations)-1)
    elif len(start_dict[top_facies_and_other[i]]) < len(end_dict[top_facies_and_other[i]]):
        start_dict[top_facies_and_other[i]].insert(0, 0)

# make a color order for the facies types that follows the stacked area plot
facies_colors = [muted_colors[i] for i in range(len(top_facies_and_other))]

# now plot the lines as rectangles on different y levels for each facies
for i in range(len(top_facies_and_other)):
    for j in range(len(start_dict[top_facies_and_other[i]])):
        this_start_height = section_data['height'].iloc[start_dict[top_facies_and_other[i]][j]]
        this_end_height = section_data['height'].iloc[end_dict[top_facies_and_other[i]][j]]
        # and just plot a line from the start to the end, at the y height of this index
        ax[2].plot([this_start_height, this_end_height], [i, i], linestyle='-', linewidth=4, color=facies_colors[i])


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
plt.savefig(fig_path + '/' + fig_name)
