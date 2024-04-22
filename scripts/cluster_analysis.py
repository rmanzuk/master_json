# script with sections to summarily look at clustering whole rock data
# written by R. A. Manzuk 04/18/2024
# last updated 04/18/2024

##########################################################################################
# package imports
##########################################################################################
# %%
import matplotlib.pyplot as plt # for plotting
import matplotlib # for color handling
import json # for json handling
import numpy as np # for numerical operations
import pandas as pd
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.decomposition import PCA

#%%
##########################################################################################
# local function imports
##########################################################################################
# %% 
from json_processing import select_gridded_bow, assemble_samples, select_gridded_im_metrics, select_gridded_point_counts, select_gridded_pa
from data_processing import dendrogram_with_inds

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

# %% define paths, and read in the outcrop json, and assemble samples

outcrop_json_file = '/Users/ryan/Dropbox (Princeton)/code/master_json/stewarts_mill.json'
sample_json_dir = '/Users/ryan/Dropbox (Princeton)/code/master_json/stewarts_mill_grid_samples/'

with open(outcrop_json_file, 'r') as f:
    outcrop_data = json.load(f)

outcrop_data = assemble_samples(outcrop_data, sample_json_dir, data_type=['grid_data'], data_name=["Stewart's Mill Grid"])

# %% select the gridded bag of words, im metrics, and point counts

bow_df = select_gridded_bow(outcrop_data)

im_metric_df =  select_gridded_im_metrics(outcrop_data, desired_metrics=['percentile', 'rayleigh_anisotropy', 'entropy'], desired_scales=[1,0.5, 0.25, 0.125, 0.0625, 0.03125, 0.015625, 0.0078125, 0.00390625, 0.001953125])

point_count_df = select_gridded_point_counts(outcrop_data)

presence_absence_df = select_gridded_pa(outcrop_data)

# %% do PCA to get anisotropy components
# first get all the unique scales that entropy was calculated at
# get the index of all rows where the metric name is entropy
all_metrics_anisotropy = im_metric_df.metric_name
anisotropy_inds = [x for x in range(len(all_metrics_anisotropy)) if 'rayleigh_anisotropy' in all_metrics_anisotropy[x]]

unique_scales_anisotropy = np.unique([im_metric_df.scale[x] for x in anisotropy_inds])

# and the unique bands it was calculated at
unique_bands_anisotropy = np.unique([im_metric_df.wavelength[x] for x in anisotropy_inds])

unique_samples = im_metric_df.sample_name.unique()

# make an array to hold the anisotropy spectra
n_samples = len(unique_samples)
n_scales = len(unique_scales_anisotropy)
n_bands = len(unique_bands_anisotropy)

anisotropy_spectra = np.zeros((n_samples, n_scales, n_bands))

# and to hold the names of the samples
anisotropy_names = []

# iterate through all rows of the im_metric_df
for index, row in im_metric_df.iterrows():
    # first check if this metric is entropy, otherwise skip
    if 'rayleigh_anisotropy' in row['metric_name']:
        # get the sample index
        sample_index = np.where(unique_samples == row['sample_name'])[0][0]
        # get the scale index
        scale_index = np.where(unique_scales_anisotropy == row['scale'])[0][0]
        # get the band index
        band_index = np.where(unique_bands_anisotropy == row['wavelength'])[0][0]
        # put the value in the array
        anisotropy_spectra[sample_index, scale_index, band_index] = row['value']

        # if this is the first time we've seen this sample, get the name
        if row['sample_name'] not in anisotropy_names:
            anisotropy_names.append(row['sample_name'])


# normalize the spectra matrix to have a mean of 0 and a standard deviation of 1
original_anisotropy_spectra = anisotropy_spectra.copy()
anisotropy_spectra = (anisotropy_spectra - np.mean(anisotropy_spectra, axis=0))/np.std(anisotropy_spectra, axis=0)

# now we can do a PCA on each band
n_components = anisotropy_spectra.shape[1]
pca_list = []
for band_index in range(n_bands):
    pca = PCA(n_components=n_components)
    pca.fit(anisotropy_spectra[:,:,band_index])
    pca_list.append(pca)

# and make a new 3d array to hold the PCA reprjections
anisotropy_scores = np.zeros((n_samples, n_components, n_bands))
for band_index in range(n_bands):
    anisotropy_scores[:,:,band_index] = pca_list[band_index].transform(anisotropy_spectra[:,:,band_index])

# make an array to hold the explained variance
anisotropy_exp_var = np.zeros((n_bands, n_components))
for band_index in range(n_bands):
    anisotropy_exp_var[band_index,:] = pca_list[band_index].explained_variance_ratio_

# make an array to hold the loadings
anisotropy_loadings = np.zeros((n_bands, n_components, n_scales))
for band_index in range(n_bands):
    anisotropy_loadings[band_index,:,:] = pca_list[band_index].components_

# %% do PCA to get color components
    
# first need to extract the color percentile spectra from the im_metric_df
# we'll make a 3D array that is n_samples x n_percentiles x n_bands, which we can do a PCA on each band

# get a list of the unique samples
unique_samples = im_metric_df.sample_name.unique()

# look in the metrics to find the unique percentiles, which are listed as 'percentile_XX'
all_metrics_percentile = im_metric_df.metric_name
percentile_metrics = all_metrics_percentile[all_metrics_percentile.str.contains('percentile')]
unique_percentiles = percentile_metrics.unique()

# and extract just the number and sort them
unique_percentiles = np.sort([int(x.split('_')[1]) for x in unique_percentiles])

# get the unique bands
unique_bands_percentile = im_metric_df.wavelength.unique()

# and flag if we want the normalized or unnormalized spectra
normalized = False

# make a 3D array to hold the dataf
n_samples = len(unique_samples)
n_percentiles = len(unique_percentiles)
n_bands = len(unique_bands_percentile)

percentile_spectra = np.zeros((n_samples, n_percentiles, n_bands))

# and separate ones to hold names 
percentile_names = []

# iterate through all rows of the im_metric_df
for index, row in im_metric_df.iterrows():
    # first check if this metric is a percentile, otherwise skip
    if 'percentile' in row['metric_name'] and row['normalized'] == normalized:
        # get the sample index
        sample_index = np.where(unique_samples == row['sample_name'])[0][0]
        # get the percentile index
        percentile_index = np.where(unique_percentiles == int(row['metric_name'].split('_')[1]))[0][0]
        # get the band index
        band_index = np.where(unique_bands_percentile == row['wavelength'])[0][0]
        # put the value in the array
        percentile_spectra[sample_index, percentile_index, band_index] = row['value']

        # if this is the first time we've seen this sample, get the name
        if row['sample_name'] not in percentile_names:
            percentile_names.append(row['sample_name'])

# normalize the spectra matrix to have a mean of 0 and a standard deviation of 1
#percentile_spectra = (percentile_spectra - np.nanmean(percentile_spectra, axis=0))/np.nanstd(percentile_spectra, axis=0)

# there should be no variance in the 100th percentile, so we can remove it
percentile_spectra = percentile_spectra[:,:-1,:]

# now we can do a PCA on each band
n_components = percentile_spectra.shape[1]
pca_list = []
for band_index in range(n_bands):
    pca = PCA(n_components=n_components)
    pca.fit(percentile_spectra[:,:,band_index])
    pca_list.append(pca)

# and make a new 3d array to hold the PCA reprjections
percentile_scores = np.zeros((n_samples, n_components, n_bands))
for band_index in range(n_bands):
    percentile_scores[:,:,band_index] = pca_list[band_index].transform(percentile_spectra[:,:,band_index])

# make an array to hold the explained variance
percentile_exp_var = np.zeros((n_bands, n_components))
for band_index in range(n_bands):
    percentile_exp_var[band_index,:] = pca_list[band_index].explained_variance_ratio_

# make an array to hold the loadings
percentile_loadings = np.zeros((n_bands, n_components, n_components))
for band_index in range(n_bands):
    percentile_loadings[band_index,:,:] = pca_list[band_index].components_

# %% Look at PCAs for point count fractions

# this df is mostly ready to go, just need to extract data, make some adjustments, and do a PCA
#pc_classes = point_count_df.columns[4:] 
pc_names = point_count_df.sample_name.copy()
pc_names = pc_names.to_numpy()

# going to manually select classes for now
pc_classes = ['Microb', 'Spar', 'Dol', 'Arch', 'Mi', 'ooid']

# extract the data into an array
pc_data = point_count_df[pc_classes].to_numpy()

# before redoing fractions, replace nans with zeros
pc_data = np.nan_to_num(pc_data)

# make the rows sum to 1 again
pc_data = pc_data/np.sum(pc_data, axis=1)[:,None]

# now normalize and do a PCA
pc_data_original = pc_data.copy()
pc_data = (pc_data - np.mean(pc_data, axis=0))/np.std(pc_data, axis=0)

pca = PCA(n_components=len(pc_classes))
pca.fit(pc_data)
pc_scores = pca.transform(pc_data)
pc_loadings = pca.components_
pc_explained_variance = pca.explained_variance_ratio_

# %% process the bag of words to get LDA topic scores
    
bow_array = bow_df['bow'].values

# unravel the bow array into one single list, and for each bow, keep track of the sample name
bow_list = []
bow_sample_names = []
for bow, i in zip(bow_array, range(len(bow_array))):
    bow_list.extend(bow)
    # make a list repeating the sample name for each entry in the bow
    samp_name_repeated = [bow_df['sample_name'].values[i]] * len(bow)
    bow_sample_names.extend(samp_name_repeated)

# split each entry in the bow list at the spaces so we have a list of lists
bow_list = [bow.split() for bow in bow_list]


# go through the bow list and replace any words that are plurals with the singular form
# at this point all plurals are just the word + 's', except a few
plural_exceptions = ['debris', 'corteces']
for i, bow in enumerate(bow_list):
    for j, word in enumerate(bow):
        if word[-1] == 's' and word not in plural_exceptions:
            bow_list[i][j] = word[:-1]

# go through the bow list and replace any words that are synonyms with the same word
# we'll list out the synonym pairs
synonym_pairs = [['heavily', 'highly'], ['skeletal', 'shell',], ['rare','sparse'], ['large','giant']]

for i, bow in enumerate(bow_list):
    for j, word in enumerate(bow):
        for pair in synonym_pairs:
            if word in pair:
                bow_list[i][j] = pair[0]

               
# put together a list of all the unique words
unique_words = []
for bow in bow_list:
    # first check if the words in this list are already in the unique words, if not, add them, 
    # agnostic of case
    for word in bow:
        if word.lower() not in [u.lower() for u in unique_words]:
            unique_words.append(word)

# from the unique words, put together a list of all possible pairs of words, agnostic of order
word_pairs = []
for i in range(len(unique_words)):
    for j in range(i+1, len(unique_words)):
        word_pairs.append([unique_words[i], unique_words[j]])

# and all possible triplets
word_triplets = []
for i in range(len(unique_words)):
    for j in range(i+1, len(unique_words)):
        for k in range(j+1, len(unique_words)):
            word_triplets.append([unique_words[i], unique_words[j], unique_words[k]])

# reduce the word pairs and triplets to only those that are actually in the bow lists
pair_exists = np.zeros(len(word_pairs)).astype(bool)
for i, pair in enumerate(word_pairs):
    for bow in bow_list:
        if pair[0] in bow and pair[1] in bow:
            pair_exists[i] = True

word_pairs = [word_pairs[i] for i in range(len(word_pairs)) if pair_exists[i]]

triplet_exists = np.zeros(len(word_triplets)).astype(bool)
for i, triplet in enumerate(word_triplets):
    for bow in bow_list:
        if triplet[0] in bow and triplet[1] in bow and triplet[2] in bow:
            triplet_exists[i] = True

word_triplets = [word_triplets[i] for i in range(len(word_triplets)) if triplet_exists[i]]

# make a 1 hot that is n_samples x n_unique_words + n_word_pairs + n_word_triplets
has_grams = np.zeros((len(bow_array), len(unique_words) + len(word_pairs) + len(word_triplets))).astype(bool)

# fill in the 1 hot
for i, bow in enumerate(bow_list):
    # first get its row index based upon sample name
    sample_index = np.where(bow_df['sample_name'].values == bow_sample_names[i])[0][0]
    # fill in the 1 hot for the unique words
    for j, word in enumerate(unique_words):
        if word in bow:
            has_grams[sample_index, j] = True
    # fill in the 1 hot for the word pairs
    for j, pair in enumerate(word_pairs):
        if pair[0] in bow and pair[1] in bow:
            has_grams[sample_index, len(unique_words) + j] = True
    # fill in the 1 hot for the word triplets
    for j, triplet in enumerate(word_triplets):
        if triplet[0] in bow and triplet[1] in bow and triplet[2] in bow:
            has_grams[sample_index, len(unique_words) + len(word_pairs) + j] = True

# first we'll remove any grams that are in less than 5% of the samples
min_support = 0.05
gram_support = np.sum(has_grams, axis=0) / len(has_grams)
has_grams = has_grams[:, gram_support > min_support]

# need to make has_grams a float so we can multiply it
has_grams = has_grams.astype(float)

# we'll add a little weight to bigrams and more weight to trigrams
has_grams[:, len(unique_words):len(unique_words) + len(word_pairs)] *= 2
has_grams[:, len(unique_words) + len(word_pairs):] *= 3

n_topics = 6
lda = LatentDirichletAllocation(n_components=n_topics)
lda.fit(has_grams)

# get the topic-word matrix
topic_word = lda.components_

# get the sample-topic matrix
sample_topic = lda.transform(has_grams)

# get the most likely topic for each sample
most_likely_topic = np.argmax(sample_topic, axis=1)

# need a list of all the words that are in the grams, and unpack the word pairs and triplets to be single strings
all_grams = unique_words + word_pairs + word_triplets
all_grams = [' '.join(gram) if type(gram) == list else gram for gram in all_grams]

# %% make the facies vector for 'whole rock' metrics

whole_rock_fields = ['sample_name', 'field_lithology', 'lat', 'lon', 'msl', 'anisotropy_pc1', 'anisotropy_pc2', 'anisotropy_pc3', 'pc_pc1', 'pc_pc2', 'pc_pc3', 'pc_pc4', 'pc_pc5']

# and add in the original point count classes
for pc_class in pc_classes:
    whole_rock_fields.append(pc_class)

# add in the presence or absence classes
for pa_class in presence_absence_df.columns[4:]:
    whole_rock_fields.append(pa_class)

# add in the number of topics as fields
for i in range(n_topics):
    whole_rock_fields.append('LDA_topic' + str(i+1))

# and systematically add in the first 3 pcs for each of the percentile spectra
band_strs = unique_bands_percentile.astype(int).astype(str)
for i in range(3):
    for band in band_strs:
        whole_rock_fields.append(band + '_percentile_pc' + str(i+1))

# we'll use anisotropy as the standard, and give the other pcs inds to match
pc_inds = []
pa_inds = []
percentile_inds = []
lda_inds = []
for i in range(len(anisotropy_names)):
    sample_name = anisotropy_names[i]
    if sample_name in pc_names:
        pc_inds.append(np.where(pc_names == sample_name)[0][0])
    else:
        pc_inds.append(np.nan)
    if sample_name in presence_absence_df.sample_name.to_numpy():
        pa_inds.append(np.where(presence_absence_df.sample_name == sample_name)[0][0])
    else:
        pa_inds.append(np.nan)
    if sample_name in percentile_names:
        percentile_inds.append(np.where(np.array(percentile_names) == sample_name)[0][0])  
    else:
        percentile_inds.append(np.nan)
    if sample_name in bow_df.sample_name.to_numpy():
        lda_inds.append(np.where(bow_df.sample_name.to_numpy() == sample_name)[0][0])
    else:
        lda_inds.append(np.nan)

# check the other samples to see if they have samples that the anisotropy samples don't, and add them to the list (but not for the presence absence)
for i in range(len(pc_names)):
    sample_name = pc_names[i]
    if sample_name not in anisotropy_names:
        pc_inds.append(i)
        anisotropy_names = np.append(anisotropy_names, [sample_name])
        if sample_name in percentile_names:
            percentile_inds.append(np.where(np.array(percentile_names) == sample_name)[0][0])
        else:
            percentile_inds.append(np.nan)
        if sample_name in presence_absence_df.sample_name:
            pa_inds.append(np.where(presence_absence_df.sample_name == sample_name)[0][0])
        else:
            pa_inds.append(np.nan)
        if sample_name in bow_df.sample_name.to_numpy():
            lda_inds.append(np.where(bow_df.sample_name.to_numpy() == sample_name)[0][0])
        else:
            lda_inds.append(np.nan)

for i in range(len(presence_absence_df.sample_name)):
    sample_name = presence_absence_df.sample_name[i]
    if sample_name not in anisotropy_names:
        pa_inds.append(i)
        anisotropy_names = np.append(anisotropy_names, [sample_name])
        if sample_name in pc_names:
            pc_inds.append(np.where(pc_names == sample_name)[0][0])
        else:
            pc_inds.append(np.nan)
        if sample_name in percentile_names:
            percentile_inds.append(np.where(np.array(percentile_names) == sample_name)[0][0])
        else:
            percentile_inds.append(np.nan)
        if sample_name in bow_df.sample_name.to_numpy():
            lda_inds.append(np.where(bow_df.sample_name.to_numpy() == sample_name)[0][0])
        else:
            lda_inds.append(np.nan)

for i in range(len(percentile_names)):
    sample_name = percentile_names[i]
    if sample_name not in anisotropy_names:
        percentile_inds.append(i)
        anisotropy_names = np.append(anisotropy_names, [sample_name])
        if sample_name in pc_names:
            pc_inds.append(np.where(pc_names == sample_name)[0][0])
        else:
            pc_inds.append(np.nan)
        if sample_name in presence_absence_df.sample_name:
            pa_inds.append(np.where(presence_absence_df.sample_name == sample_name)[0][0])
        else:
            pa_inds.append(np.nan)
        if sample_name in bow_df.sample_name.to_numpy():
            lda_inds.append(np.where(bow_df.sample_name.to_numpy() == sample_name)[0][0])
        else:
            lda_inds.append(np.nan)

for i in range(len(bow_df.sample_name.to_numpy())):
    sample_name = bow_df.sample_name.to_numpy()[i]
    if sample_name not in anisotropy_names:
        lda_inds.append(i)
        anisotropy_names = np.append(anisotropy_names, [sample_name])
        if sample_name in pc_names:
            pc_inds.append(np.where(pc_names == sample_name)[0][0])
        else:
            pc_inds.append(np.nan)
        if sample_name in presence_absence_df.sample_name:
            pa_inds.append(np.where(presence_absence_df.sample_name == sample_name)[0][0])
        else:
            pa_inds.append(np.nan)
        if sample_name in percentile_names:
            percentile_inds.append(np.where(np.array(percentile_names) == sample_name)[0][0])
        else:
            percentile_inds.append(np.nan)
               

# bring in field lithologies and locations for the vector
field_liths = []
field_locs = np.zeros((len(anisotropy_names), 3))

# to get some of the field data, will be easiest to look at the outcrop data, but we need a list of the sample names in there
outcrop_sample_names = []
for sample in outcrop_data['grid_data'][0]['samples']:
    outcrop_sample_names.append(sample['sample_name'])  

# then add the field lithologies, lat, lon, and msl
for i in range(len(anisotropy_names)):
    outcrop_index = outcrop_sample_names.index(anisotropy_names[i])
    field_liths.append(outcrop_data['grid_data'][0]['samples'][outcrop_index]['field_lithology'])
    field_locs[i,0] = outcrop_data['grid_data'][0]['samples'][outcrop_index]['latitude']
    field_locs[i,1] = outcrop_data['grid_data'][0]['samples'][outcrop_index]['longitude']
    field_locs[i,2] = outcrop_data['grid_data'][0]['samples'][outcrop_index]['msl']

# assemble the anisotropy array from the first 3 pcs, which is just the anisotropy scores
anisotropy_scores_full = anisotropy_scores[:,:3,:]
# and squeeze it to get rid of the extra dimension
anisotropy_scores_full = np.squeeze(anisotropy_scores_full)
# check if it is too short because we added samples from other metrics
if anisotropy_scores_full.shape[0] < len(anisotropy_names):
    missing_rows = len(anisotropy_names) - anisotropy_scores_full.shape[0]
    anisotropy_scores_full = np.append(anisotropy_scores_full, np.full((missing_rows, 3, n_bands), np.nan), axis=0)

# assemble the pc array from the first 5 pcs
pc_scores_full = np.zeros((len(anisotropy_names), 5 + len(pc_classes)))
for i in range(len(anisotropy_names)):
    if not np.isnan(pc_inds[i]):
        pc_scores_full[i,:5] = np.squeeze(pc_scores[int(pc_inds[i]),0:5])
        pc_scores_full[i,5:] = pc_data_original[int(pc_inds[i]),:]
    else:
        pc_scores_full[i,:] = np.nan

# assemble the lda array
lda_scores_full = np.zeros((len(anisotropy_names), n_topics))
for i in range(len(anisotropy_names)):
    if not np.isnan(lda_inds[i]):
        lda_scores_full[i,:] = np.squeeze(sample_topic[int(lda_inds[i]),0:n_topics])

# assemble the presence absence array
pa_scores_full = np.zeros((len(anisotropy_names), presence_absence_df.shape[1] - 4))
for i in range(len(anisotropy_names)):
    if not np.isnan(pa_inds[i]):
        pa_scores_full[i,:] = np.squeeze(presence_absence_df.iloc[int(pa_inds[i]),4:])

# assemble the percentile array
percentile_scores_full = np.zeros((len(anisotropy_names), 3*len(unique_bands_percentile)))
for i in range(len(anisotropy_names)):
    if not np.isnan(percentile_inds[i]):
        percentile_scores_full[i,:] = np.reshape(percentile_scores[int(percentile_inds[i]),0:3,:], (1,3*len(unique_bands_percentile)))

# should be good to go to assemble the dataframe
whole_rock_vector = np.column_stack((anisotropy_names, field_liths, field_locs, anisotropy_scores_full, pc_scores_full, pa_scores_full, lda_scores_full, percentile_scores_full))
whole_rock_vector = pd.DataFrame(whole_rock_vector, columns=whole_rock_fields)

# combining the dataframe turned the floats back into strings, so convert them back
data_cols = whole_rock_vector.columns[2:]
whole_rock_vector[data_cols] = whole_rock_vector[data_cols].astype(float)


# %% extract the data and make a pairwise correlation matrix

whole_rock_data = whole_rock_vector.drop(columns=['sample_name', 'field_lithology', 'lat', 'lon', 'msl'])

# get the correlation matrix
corr_matrix = whole_rock_data.corr()

# and show the matrix as an image with a color bar and cells labeled
plt.figure()
plt.imshow(corr_matrix, cmap='viridis')
plt.colorbar()
plt.xticks(np.arange(len(whole_rock_data.columns)), whole_rock_data.columns, rotation=90)
plt.yticks(np.arange(len(whole_rock_data.columns)), whole_rock_data.columns)
plt.show()
