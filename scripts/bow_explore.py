# script with sections to look at PCA of geochemical data
# written by R. A. Manzuk 04/15/2024
# last updated 04/18/2024

##########################################################################################
# package imports
##########################################################################################
# %%
import matplotlib.pyplot as plt # for plotting
import matplotlib # for color handling
import json # for json handling
import numpy as np # for numerical operations
from mlxtend.frequent_patterns import apriori
import pandas as pd
from sklearn.decomposition import LatentDirichletAllocation

#%%
##########################################################################################
# local function imports
##########################################################################################
# %% 
from json_processing import select_gridded_bow, assemble_samples
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

# %% define paths, and read in the outcrop json, assemble samples, and get the bow

outcrop_json_file = '/Users/ryan/Dropbox (Princeton)/code/master_json/stewarts_mill.json'
sample_json_dir = '/Users/ryan/Dropbox (Princeton)/code/master_json/stewarts_mill_grid_samples/'

with open(outcrop_json_file, 'r') as f:
    outcrop_data = json.load(f)

outcrop_data = assemble_samples(outcrop_data, sample_json_dir, data_type=['grid_data'], data_name=["Stewart's Mill Grid"])

bow_df = select_gridded_bow(outcrop_data)

# %% extract all the bow lists into a single numpy array, keeping track of the sample names

bow_array = bow_df['bow'].values

# unravel the bow array into one single list, and for each bow, keep track of the sample name
bow_list = []
sample_names = []
for bow, i in zip(bow_array, range(len(bow_array))):
    bow_list.extend(bow)
    # make a list repeating the sample name for each entry in the bow
    samp_name_repeated = [bow_df['sample_name'].values[i]] * len(bow)
    sample_names.extend(samp_name_repeated)

# split each entry in the bow list at the spaces so we have a list of lists
bow_list = [bow.split() for bow in bow_list]

# %% deal with plurals and synonyms

# go through the bow list and replace any words that are plurals with the singular form
# at this point all plurals are just the word + 's', except a few
plural_exceptions = ['debris', 'corteces', 'cross']
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

# %% we'll treat every entry phrase in the bow_list as a transaction, and we'll make a 1 hot encoding of all the unique words for each transaction
                
# put together a list of all the unique words
unique_words = []
for bow in bow_list:
    # first check if the words in this list are already in the unique words, if not, add them, 
    # agnostic of case
    for word in bow:
        if word.lower() not in [u.lower() for u in unique_words]:
            unique_words.append(word)

# make a 1 hot that is n_descriptions x n_unique_words
has_words = np.zeros((len(bow_list), len(unique_words))).astype(bool)

# fill in the 1 hot
for i, bow in enumerate(bow_list):
    for j, word in enumerate(unique_words):
        if word in bow:
            has_words[i, j] = True

# %% use apriori to find frequent word combinations

# we'll use the apriori algorithm to find frequent word combinations
# first we need to make a dataframe from the 1 hot matrix
has_words_df = pd.DataFrame(has_words, columns=unique_words)

# we'll a flexible minimum support threshold
min_support = 0.01
frequent_itemsets = apriori(has_words_df, min_support=min_support, use_colnames=True)
frequent_itemsets['length'] = frequent_itemsets['itemsets'].apply(lambda x: len(x))

# we'll inverse sort by length
frequent_itemsets = frequent_itemsets.sort_values(by=['length'], ascending=False)

# right now support is for the number of entries in the bow list, but we wnat it to be for the number of samples
bow_per_sample = len(has_words_df) / len(np.unique(sample_names))
frequent_itemsets['support'] = frequent_itemsets['support'] * bow_per_sample

# %% that' fine, but we want to find associations of words and phrases within samples that are frequent among samples, so we'll add bigrams and trigrams to the mix

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
    sample_index = np.where(bow_df['sample_name'].values == sample_names[i])[0][0]
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

# %% preprocess the has_grams matrix to remove infewquent grams and add weight to bigrams and trigrams
            
# first we'll remove any grams that are in less than 5% of the samples
min_support = 0.05
gram_support = np.sum(has_grams, axis=0) / len(has_grams)
has_grams = has_grams[:, gram_support > min_support]

# need to make has_grams a float so we can multiply it
has_grams = has_grams.astype(float)

# we'll add a little weight to bigrams and more weight to trigrams
has_grams[:, len(unique_words):len(unique_words) + len(word_pairs)] *= 2
has_grams[:, len(unique_words) + len(word_pairs):] *= 3

            
# %% do a latent dirichlet allocation on the grams
            
# first optimize the number of topics
n_topics = np.arange(2, 20)
perplexities = []
for n in n_topics:
    lda = LatentDirichletAllocation(n_components=n)
    lda.fit(has_grams)
    perplexities.append(lda.perplexity(has_grams))

# change perplexity to log likelihood
perplexities = np.array(perplexities)
log_likelihoods = -perplexities

plt.scatter(n_topics, log_likelihoods)
plt.xlabel('Number of Topics')
plt.ylabel('Perplexity')
plt.show()

# %% based upont the peak in the log likelihood, we'll choose 5 topics

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

# for each topic, print the top 10 words, and the top 3 samples that are most likely to be in that topic
for i in range(n_topics):
    print(f'Topic {i}')
    topic_words = np.argsort(topic_word[i])[::-1]
    print([all_grams[j] for j in topic_words[:10]])
    top_samples = np.argsort(sample_topic[:, i])[::-1]
    print([bow_df['sample_name'].values[j] for j in top_samples[:3]])

# %% put the lda result into a dendrogram
    
dendro, cluster_list = dendrogram_with_inds(sample_topic, linkage_method='single', dendrogram_p=1)

# turn the cluster list into a list of sample names
cluster_sample_names = []
for cluster in cluster_list:
    cluster_sample_names.append([bow_df['sample_name'].values[int(i)] for i in cluster])

# show the dendrogram
plt.show()

# %% print out the sampe that contains a particular word

word = 'wackestone:'
word_index = all_grams.index(word)
word_samples = np.where(has_grams[:, word_index])[0]
print([bow_df['sample_name'].values[i] for i in word_samples])
