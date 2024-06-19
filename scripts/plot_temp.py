# script with sections to plot historical temperature data
# not part of the json project, but needed for the paper
# written by R. A. Manzuk 05/28/2024
# last updated 05/28/2024
##########################################################################################
# package imports
##########################################################################################
# %%
import matplotlib.pyplot as plt # for plotting
import matplotlib # for custom color maps
import numpy as np # for numerical operations
import pandas as pd # for data manipulation
import datetime # for date operations
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
temp_data_path = '/Users/ryan/Dropbox (Princeton)/reef_survey_project/temp_data/heightstown_weather.csv'

# load the data
temp_data = pd.read_csv(temp_data_path)

# %% extract date and temp vectors

# get the date vec as a datetime object
date_vec = pd.to_datetime(temp_data['DATE'])

# get the temperature vector, convert to celsius
temp_vec = (temp_data['TMAX'] - 32) * (5/9)

# %% get the yearly average temperature, and a regression line

# get the year of each date
year_vec = date_vec.dt.year

# get the unique years, and cut off the last year because it's incomplete
unique_years = np.unique(year_vec)
unique_years = unique_years[:-1]

# initialize a vector to hold the yearly average temperature
yearly_avg_temp = np.zeros(len(unique_years))

# loop through the years and get the average temperature
for i, year in enumerate(unique_years):
    # get the indices for the year
    year_inds = year_vec == year
    # get the average temperature
    yearly_avg_temp[i] = np.mean(temp_vec[year_inds])

# do the regression
regression = np.polyfit(unique_years, yearly_avg_temp, 1)
regression_line = np.polyval(regression, unique_years)

# %% make a vector that randomly samples once per decade

# initialize a vector to hold the indices
decadal_sample_inds = np.zeros(len(date_vec), dtype=bool)

# assign each year to a decade
decade_vec = np.floor(year_vec / 10) * 10

# get the unique decades
unique_decades = np.unique(decade_vec)

# loop through the decades and randomly sample one data point
for decade in unique_decades:
    # get the indices for the decade
    decade_inds = decade_vec == decade
    # randomly sample one index
    random_ind = np.random.choice(np.where(decade_inds)[0], 1)
    # assign the index to the decadal sample indices
    decadal_sample_inds[random_ind] = True


# %% plot the data

# make the plot
fig, ax = plt.subplots(1, 1, figsize=(10, 2))

# first plot all the data, with a low alpha
ax.plot(date_vec, temp_vec, color=indigo, alpha=0.3)

# then plot the yearly average temperature
ax.plot([datetime.datetime(year, 1, 1) for year in unique_years], regression_line, color=indigo, linewidth=2)

# and then plot the randomly sampled data
ax.plot(date_vec[decadal_sample_inds], temp_vec[decadal_sample_inds], color=green, linewidth=2)

# set the labels
ax.set_xlabel('Date')
ax.set_ylabel('Temperature (C)')

# make the axis limits tight
ax.set_xlim([datetime.datetime(1900, 1, 1), datetime.datetime(2024, 1, 1)])
ax.set_ylim([-20, 40])

plt.tight_layout()
plt.show()

