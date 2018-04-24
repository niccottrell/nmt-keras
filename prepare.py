from pickle import load
from pickle import dump
from numpy.random import shuffle

lang2 = 'sve'


# load a clean dataset
def load_clean_sentences(filename):
    return load(open(filename, 'rb'))


# save a list of clean sentences to file
def save_clean_data(sentences, filename):
    dump(sentences, open(filename, 'wb'))
    print('Saved: %s' % filename)


# load dataset
raw_dataset = load_clean_sentences('eng-' + lang2 + '.pkl')

# reduce dataset size
n_sentences = 10000
dataset = raw_dataset[:n_sentences, :]
# random shuffle
shuffle(dataset)
# split into train/test
train, test = dataset[:9000], dataset[9000:]
# save
save_clean_data(dataset, 'eng-' + lang2 + '-both.pkl')
save_clean_data(train, 'eng-' + lang2 + '-train.pkl')
save_clean_data(test, 'eng-' + lang2 + '-test.pkl')
