from numpy.random import shuffle

from helpers import *

# load dataset
raw_dataset = load_clean_sentences('eng-' + lang2 + '.pkl')

# reduce dataset size
n_sentences = 16000
n_test = 1000
idx_cutoff = n_sentences - n_test

dataset = raw_dataset[:n_sentences, :]
# random shuffle
shuffle(dataset)
# split into train/test
train, test = dataset[:idx_cutoff], dataset[idx_cutoff:]
# save
save_clean_data(dataset, 'eng-' + lang2 + '-both.pkl')
save_clean_data(train, 'eng-' + lang2 + '-train.pkl')
save_clean_data(test, 'eng-' + lang2 + '-test.pkl')
