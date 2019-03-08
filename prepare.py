from numpy.random import shuffle

import config

n_test = 200  # higher values will make evaluation slow

if config.data.__class__.__name__ == 'Files':

    # load dataset
    raw_dataset = config.data.load_clean_sentences()

    # reduce dataset size
    n_sentences = raw_dataset.shape[0]
    idx_cutoff = n_sentences - n_test

    dataset = raw_dataset[:n_sentences, :]
    # random shuffle
    shuffle(dataset)
    # split into train/test
    train, test = dataset[:idx_cutoff], dataset[idx_cutoff:]
    # save
    config.data.save_clean_data(dataset, 'both')
    config.data.save_clean_data(train, 'train')
    config.data.save_clean_data(test, 'test')

else:  # mongodb

    config.data.reset_test_flag(n_test)
