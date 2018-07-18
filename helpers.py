from pickle import load
from pickle import dump

from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical
from numpy import array

from nltk.stem import WordNetLemmatizer
from nltk.tag.hunpos import HunposTagger

from os.path import expanduser

import hunspell
import unicodedata
import re
import nltk
import string
import pyphen

nltk.download('averaged_perceptron_tagger')

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


# create a tokenizer
def create_tokenizer(lines) -> Tokenizer:
    return create_tokenizer_simple(lines)


# Tokenize lines on spaces (not preserved) - don't lowercase, but filter out most punctuation, tabs and newlines
def create_tokenizer_simple(lines) -> Tokenizer:
    tokenizer = Tokenizer(
        filters='"#$%&()*+-/:;<=>@[\\]^_`{|}~\t\n',
        lower=False)  # Since in German case has significance
    tokenizer.fit_on_texts(lines)
    return tokenizer


def pos_tag(line, lang='en'):
    """

    :type line: list(str)
    """
    tuples = do_pos_tag(lang, line)
    result = []
    for tuple in tuples:
        if tuple[0] in set(string.punctuation):  # It's just punctuation
            result.append(tuple[0])
        else:
            pos = tuple[1].decode('utf-8')
            result.append(
                tuple[0] + "." + pos[:2])  # Only take the first 2 letters of the POS, e.g. 'NN_UTR_SIN_DEF_NOM' -> 'NN'
    return result


def do_pos_tag(lang, line):
    """
    Append part-of-speech tags to each word
    :type line: list(str)
    """
    iso3 = ('sve' if lang[:2] == 'sv' else 'eng')
    if (iso3 == 'eng'):
        # tuples = nltk.tag.pos_tag(line, lang=iso3)
        model = 'en_wsj.model'
    else:  # Swedish
        model = 'suc-suctags.model'
    ht = HunposTagger(model, path_to_bin='./hunpos-tag')
    tuples = ht.tag(line)
    return tuples


def prepare_lines(lines, lang='en', lc_first=None) -> list:
    """ Inelegant way to preserve punctuation 
    :type lang: str
    """
    regex = re.compile(r"([.?,!])")
    res = list()
    for line in lines:
        # add spaces before punctuation
        line = regex.sub(" \g<1>", line)
        # language-specific fixes
        if (lang is 'en'):
            line = re.sub(r'\'m\s+', ' am ', line)
            line = re.sub(r'\b(s?he|it)\'s\s+', r'\1 is ', line, flags=re.IGNORECASE)
            line = re.sub(r'(\w+)\'s\s+', r" \1 's ", line)
            line = re.sub(r'\'re\s+', ' are ', line)
        line = re.sub(r'\s+', ' ', line)
        # tokenize on space
        words = line.split(' ')
        # lowercase if found in dictionary
        if lc_first == 'lookup' and is_in_dict(words[0], lang): words[0] = words[0].lower()
        res.append(' '.join(words).strip())
    return res


def is_proper(word, lang):
    return is_capitalized(word) and not is_in_dict(word, lang)  # or is_noun(word, lang))


def is_capitalized(word):
    return True if word[0] == word[0].upper() else False


def is_noun(word, lang):
    tuples = do_pos_tag(lang, [word])
    pos = tuples[0][1].decode('utf-8')
    return True if pos[0] == 'N' else False


def is_in_dict(word, lang):
    home = expanduser("~")
    path = home + '/Library/Spelling/'
    if (lang == 'en'):
        hobj = hunspell.HunSpell(path + 'en_US.dic', path + 'en_US.aff')
    elif (lang == 'sv'):
        hobj = hunspell.HunSpell(path + 'sv_SE.dic', path + 'sv_SE.aff')
    else:
        raise Exception("Do not support language: " + lang)
    return hobj.spell(word)


def hyphenate(word, lang):
    if (lang == 'sv'):
        dic = pyphen.Pyphen(lang='sv_SE')
    else:
        dic = pyphen.Pyphen(lang='en_US')
    sep = '$'
    return dic.inserted(word, hyphen=sep).split(sep)


def lemmatize_verbs(words):
    """Lemmatize verbs in list of tokenized words"""
    lemmatizer = WordNetLemmatizer()
    lemmas = []
    for word in words:
        lemma = lemmatizer.lemmatize(word, pos='v')
        lemmas.append(lemma)
    return lemmas


# max sentence length
def max_length(lines):
    return max(len(line.split()) for line in lines)


def word_for_id(integer, tokenizer):
    """
    map an integer to a word
    :param integer: the index assigned to the word during initial discovery
    :param tokenizer: the tokenizer used
    :return: the original word with this index
    """
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
    return None

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
