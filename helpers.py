from pickle import load
from pickle import dump

from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical
from numpy import array
from punct_tokenizer import PunctTokenizer

import unicodedata
import re

lang2 = 'sve'


# Remove control charactesr in a unicode-aware way
def remove_control_characters(s):
    return "".join(ch for ch in s if unicodedata.category(ch)[0] != "C")


# load a clean dataset
def load_clean_sentences(filename):
    return load(open(filename, 'rb'))


# save a list of clean sentences to file
def save_clean_data(sentences, filename):
    dump(sentences, open(filename, 'wb'))
    print('Saved {0} to {1}'.format(sentences.shape, filename))


# fit a tokenizer
def create_tokenizer(lines) -> Tokenizer:
    return create_tokenizer_simple(lines)


def create_tokenizer_simple(lines) -> Tokenizer:
    tokenizer = Tokenizer(
        filters='"#$%&()*+-/:;<=>@[\\]^_`{|}~\t\n',
        lower=False)  # Since in German case has significance
    tokenizer.fit_on_texts(lines)
    return tokenizer


def prepare_lines(lines) -> list:
    """ Inelegant way to preserve punctuation """
    regex = re.compile(r"([.?,!])")
    res = list()
    for line in lines:
        # add spaces before punctuation
        line = regex.sub(" \g<1>", line)
        res.append(line)
    return res


# max sentence length
def max_length(lines):
    return max(len(line.split()) for line in lines)


# encode and pad sequences
def encode_sequences(tokenizer, length, lines):
    # integer encode sequences
    X = tokenizer.texts_to_sequences(lines)
    # pad sequences with 0 values
    X = pad_sequences(X, maxlen=length, padding='post')
    return X


# one hot encode target sequence
def encode_output(sequences, vocab_size):
    ylist = list()
    for sequence in sequences:
        encoded = to_categorical(sequence, num_classes=vocab_size)
        ylist.append(encoded)
    y = array(ylist)
    y = y.reshape(sequences.shape[0], sequences.shape[1], vocab_size)
    return y
