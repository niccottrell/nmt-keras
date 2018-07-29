from numpy.random import shuffle

from helpers import *

# load dataset
raw_dataset = load_clean_sentences()

# reduce dataset size
n_sentences = 32000
n_test = 2000
idx_cutoff = n_sentences - n_test

dataset = raw_dataset[:n_sentences, :]
# random shuffle
shuffle(dataset)
# split into train/test
train, test = dataset[:idx_cutoff], dataset[idx_cutoff:]
# save
save_clean_data(dataset, 'both')
save_clean_data(train, 'train')
save_clean_data(test, 'test')
