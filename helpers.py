from pickle import load
from pickle import dump

from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical

from numpy import array

from nltk.stem import WordNetLemmatizer

from os.path import expanduser
import traceback

# import hunspell
import unicodedata
import re  # standard regex system
import regex  # better regex system
import nltk
import string
import pyphen

# nltk.download('averaged_perceptron_tagger', download_dir=config.nltk_data)

lang2 = 'swe'

version = '201808c'


def remove_control_characters(s):
    """
    Remove control characters in a unicode-aware way
    """
    return "".join(ch for ch in s if unicodedata.category(ch)[0] != "C")


def get_filename(subset=None):
    filename = 'eng-' + lang2
    if subset is not None:
        filename += '-' + subset
    filename += '.pkl'
    return filename


def load_clean_sentences(subset=None):
    """
    load a clean dataset
    :param subset: str
    :return: ndarray
    """
    filename = get_filename(subset)
    return load(open(filename, 'rb'))


# save a list of clean sentences to file
def save_clean_data(sentences, subset=None):
    filename = get_filename(subset)
    dump(sentences, open(filename, 'wb'))
    print('Saved {0} to {1}'.format(sentences.shape, filename))


def create_tokenizer(lines) -> Tokenizer:
    """
    create a tokenizer
    :param lines: list(list(str)) already tokenized lines
    :rtype: Tokenizer
    """
    return create_tokenizer_simple(lines)


# Tokenize lines on spaces (not preserved) - don't lowercase, but filter out most punctuation
def create_tokenizer_simple(lines) -> Tokenizer:
    """
    :rtype: Tokenizer
    """
    tokenizer = Tokenizer(
        filters='"#$%&()*+-/:;<=>@[\\]^_`{|}~',  # Don't filter \t and \n since we use them as sequence markers
        lower=False)  # Since in German (at least) case has significance; In English, it tends to indicate Proper nouns
    tokenizer.fit_on_texts(lines)
    return tokenizer



re_print = re.compile('[^%s]' % re.escape(string.printable))

regex_endpunct = re.compile(r"([.?,!])")


def prepare_lines(lines, lang='en', lc_first=None) -> list:
    """ Inelegant way to preserve punctuation 
    :type lang: str
    """
    res = list()
    for line in lines:
        rejoined = prepare_line(line, lang, lc_first)
        res.append(rejoined)
    return res


def prepare_line(line, lang='en', lc_first=None):
    """
    :param line: str
    :param lang: str
    :param lc_first: bool
    """
    # add spaces before punctuation
    line = regex_endpunct.sub(" \g<1>", line)
    # language-specific fixes
    if lang is 'en':
        line = re.sub(r'\'m\s+', ' am ', line)
        line = re.sub(r'\b(s?he|it)\'s\s+', r'\1 is ', line, flags=re.IGNORECASE)
        line = re.sub(r'(\w+)\'s\s+', r" \1 's ", line)
        line = re.sub(r'\'re\s+', ' are ', line)
    line = re.sub(r'\s+', ' ', line)
    # tokenize on space
    words = line.split(' ')
    # lowercase if found in dictionary
    if lc_first == 'lookup':
        idx = 1 if is_punct(words[0]) else 0
        if is_in_dict(words[idx], lang): words[idx] = words[idx].lower()
    rejoined = ' '.join(words).strip()
    return rejoined


def is_proper(word, lang):
    return is_capitalized(word) and not is_in_dict(word, lang)  # or is_noun(word, lang))


def is_capitalized(word):
    return True if word[0] == word[0].upper() else False


def built_dict(path):
    with open(path) as f:
        return dict([line.split() for line in f])


dict_en = built_dict('dicts/en.txt')
dict_sv = built_dict('dicts/sv.txt')

def is_in_dict(word, lang):
    """
    :param str word: the word unit
    :param str lang: language code
    :return:
    """
    # path = './hunspell/'
    if lang == 'en':
      #  hobj = hunspell.HunSpell(path + 'en_US.dic', path + 'en_US.aff')
      return word.lower() in dict_en
    elif lang == 'sv':
      #  hobj = hunspell.HunSpell(path + 'sv_SE.dic', path + 'sv_SE.aff')
      return word.lower() in dict_sv
    else:
      raise Exception("Do not support language: " + lang)
    # return hobj.spell(word)


def is_punct(str):
    return regex.match(r"\p{P}+", str)

def lemmatize_verbs(words):
    """Lemmatize verbs in list of tokenized words"""
    lemmatizer = WordNetLemmatizer()
    lemmas = []
    for word in words:
        lemma = lemmatizer.lemmatize(word, pos='v')
        lemmas.append(lemma)
    return lemmas


def max_length(lines):
    """
    max sentence length
    :param lines: list(list(str)
    :return: int
    """
    assert len(lines) > 0, "lines is empty"
    return max(len(line) for line in lines)


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


def encode_sequences(tokenizer, lines, pad_length=None):
    """
    encode words to integers and pad sequences (if desired)
    :param tokenizer: Tokenizer to map from word to integer and back
    :param pad_length: int Pad the sequence up to this length
    :param lines: list(list(str): Already tokenized lines (normally words, but also phrases or sub-words)
    :return: Numpy array with values as word indexes
    """
    # check for None
    for line in lines:
        if None in line: raise ValueError('Found None value in line', line)
    # integer encode sequences
    X = tokenizer.texts_to_sequences(lines)
    # pad sequences with 0 values (only if specified)
    if pad_length != None:
        X = pad_sequences(X, maxlen=pad_length, padding='post')
    return X


def encode_1hot(sequences, vocab_size):
    """
    one hot encode integer sequence
    Converts a class vector (integers) to binary class matrix.
    :param sequences: Numpy array of integers (first level is a sentence, second is each word index) shape=(num_sentences, max_length)
    :param vocab_size: int: Create one class per vocab in this target language?
    :return: reshaped Numpy array now with 3 dimensions but one-hot encoded; shape=(num_sentences, max_length, vocab_size)
    """
    y_list = list()
    for sequence in sequences:
        encoded = to_categorical(sequence, num_classes=vocab_size)
        y_list.append(encoded)
    y_array = array(y_list)
    y = y_array.reshape(sequences.shape[0], sequences.shape[1], vocab_size)
    return y


def mark_ends(lines_tokenized):
    """
    Marks the ends of each lines with special tokens.
    Used for target sentences, to help track the beginning on end of sentences when doing inference with attention
    :param lines_tokenized: list(list(str)) tokenized lines
    :return: list(list(str))
    """
    output = []
    for line in lines_tokenized:
        output.append(['\t'] + line + ['\n'])
    return output
