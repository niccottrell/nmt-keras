from pickle import load
from pickle import dump

from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical
from keras import optimizers

from numpy import array

from nltk.stem import WordNetLemmatizer
from nltk.tag.hunpos import HunposTagger

from os.path import expanduser
import traceback

import hunspell
import unicodedata
import re  # standard regex system
import regex  # better regex system
import nltk
import string
import pyphen

from models import *

nltk.download('averaged_perceptron_tagger')

lang2 = 'sve'

version = '201808a'


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
    :return: Tokenizer
    """
    return create_tokenizer_simple(lines)


# Tokenize lines on spaces (not preserved) - don't lowercase, but filter out most punctuation, tabs and newlines
def create_tokenizer_simple(lines) -> Tokenizer:
    tokenizer = Tokenizer(
        filters='"#$%&()*+-/:;<=>@[\\]^_`{|}~\t\n',
        lower=False)  # Since in German (at least) case has significance; In English, it tends to indicate Proper nouns
    tokenizer.fit_on_texts(lines)
    return tokenizer


def simple_lines(lines, lang):
    """
    Tokenize lines by whitespace but discard whitespace
    :param lines: list(str)
    :return: list(list(str)
    """
    from nltk.tokenize import WordPunctTokenizer
    wpt = WordPunctTokenizer()  # See http://text-processing.com/demo/tokenize/ for examples
    tokenized_lines = []
    for line in lines:
        tokenized_lines.append(wpt.tokenize(line))
    return tokenized_lines


def pos_tag(line, lang='en'):
    """
    Append part-of-speech tags to each word, keeping the units as a list
    :type line: list(str)
    :return list(str)
    """
    pattern = re.compile("[\d€{}]+$".format(re.escape(string.punctuation)))
    try:
        tuples = pos_tag_tokens(line, lang)
        result = []
        for tuple in tuples:
            if pattern.match(tuple[0]):  # It's just punctuation/digits/symbols
                result.append(tuple[0])
            else:
                pos = tuple[1].decode('utf-8')
                result.append(
                    tuple[0] + "." + pos[
                                     :2])  # Only take the first 2 letters of the POS, e.g. 'NN_UTR_SIN_DEF_NOM' -> 'NN'
        return result
    except:
        print('Error tagging line `%s` in %s' % (line, lang))
        traceback.print_exc()


def pos_tag_lines(lines, lang):
    """Post tag lines in bulk
    :param lines: list(str)
    :param lang: The 2-letter language code
    :return: list(list(str))
    """
    tokenized_lines = simple_lines(lines, lang)
    tagged_lines = []
    for line in tokenized_lines:
        tagged_lines.append(pos_tag(line, lang))
    return tagged_lines

# keep HunposTaggers loaded
ht_cache = {}

def pos_tag_tokens(line, lang):
    """
    Do POS-tagging but return tuples for each input word
    :type line: list(str) An already tokenized line
    """
    iso3 = ('sve' if lang[:2] == 'sv' else 'eng')
    if iso3 in ht_cache:
        ht = ht_cache[iso3]
    else:
        if (iso3 == 'eng'):
            model = 'en_wsj.model'
            enc = 'utf-8'
        else:  # Swedish
            model = 'suc-suctags.model'
            enc = 'ISO-8859-1'
        # build the tagger
        ht = HunposTagger(model, path_to_bin='./hunpos-tag', encoding=enc)
        # cache it
        ht_cache[iso3] = ht
    tuples = ht.tag(line)
    return tuples


re_print = re.compile('[^%s]' % re.escape(string.printable))

def replace_proper(line, lang):
    """
    Append part-of-speech tags to each word, keeping the units as a list
    :type line: list(str)
    :type lang: str 2-letter language code
    :return list(str)
    """
    inside_proper = False
    proper_idx = 1
    pattern = re.compile("[\d€{}]+$".format(re.escape(string.punctuation)))
    try:
        tuples = pos_tag_tokens(line, lang)
        result = []
        for tuple in tuples:
            if pattern.match(tuple[0]):  # It's just punctuation/digits/symbols
                result.append(tuple[0])
            else:
                pos = tuple[1].decode('utf-8')
                if pos == 'NNP' or pos == 'PM_NOM':  # Proper noun
                    if not inside_proper:  # Don't write NP twice
                        result.append('NP' + str(proper_idx))
                        inside_proper = True
                        proper_idx += 1
                else:
                    result.append(tuple[0])
                    inside_proper = False
        return result
    except:
        print('Error tagging line `%s` in %s' % (line, lang))
        traceback.print_exc()


def replace_proper_lines(lines, lang):
    """Post tag lines in bulk
    :param lines: list(str)
    :param lang: The 2-letter language code
    :return: list(list(str))
    """
    tokenized_lines = simple_lines(lines, lang)
    tagged_lines = []
    for line in tokenized_lines:
        tagged_lines.append(replace_proper(line, lang))
    return tagged_lines


def word2phrase_lines(lines, lang):
    """
    Convert all line inputs to tokenized chunked phrases
    :param lines: list(str) The lines to process this time
    :param lang: str The language code (but ignored for now)
    :return: list(list(str))
    """
    # Pre-split the lines
    tokenized_lines = simple_lines(lines, lang)

    # Import the required functions
    from thirdparty.word2phrase import learn_vocab_from_train_iter, filter_vocab, apply

    # settings
    min_count = 3
    threshold = 25

    # TODO: we should cache this and only 'train' once per session per language
    dataset_both = load_clean_sentences('both')
    # prepare english tokenizer
    lang_idx = 0 if lang == 'en' else 1
    dataset_thislang = dataset_both[:, lang_idx]
    dataset_tokenized = simple_lines(dataset_thislang, lang)

    # vocab_iter, train_iter = tee(tokenized_lines)
    vocab, train_words = learn_vocab_from_train_iter(dataset_tokenized)
    print("word2phrase.train_model: raw vocab=%d, dataset_thislang=%d" % (len(vocab), train_words))
    vocab = filter_vocab(vocab, min_count)
    print("word2phrase.train_model: filtered vocab=%d" % len(vocab))

    # Now apply it to these lines
    lines = apply(tokenized_lines, vocab, train_words, '_', min_count, threshold)

    result = []
    for row in lines:
        result.append(row)  # train_model
    return result


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


def is_noun(word, lang):
    tuples = pos_tag_tokens([word], lang)
    pos = tuples[0][1].decode('utf-8')
    return True if pos[0] == 'N' else False


def is_in_dict(word, lang):
    path = './hunspell/'
    if lang == 'en':
        hobj = hunspell.HunSpell(path + 'en_US.dic', path + 'en_US.aff')
    elif lang == 'sv':
        hobj = hunspell.HunSpell(path + 'sv_SE.dic', path + 'sv_SE.aff')
    else:
        raise Exception("Do not support language: " + lang)
    return hobj.spell(word)


def is_punct(str):
    return regex.match(r"\p{P}+", str)


def hyphenate(word, lang):
    """
    Hyphenates a single word
    :param word:
    :param lang:
    :return:
    """
    if (lang == 'sv'):
        dic = pyphen.Pyphen(lang='sv_SE')
    else:
        dic = pyphen.Pyphen(lang='en_US')
    sep = '$'
    return dic.inserted(word, hyphen=sep).split(sep)


def hyphenate_lines(lines, lang):
    """
    Hyphenate all lines and wrap in a tokenizer
    :param lines: list(str)
    :param lang: the 2-letter language code
    :return: list(list(str))
    """
    if (lang == 'sv' or lang == 'sve'):
        dic = pyphen.Pyphen(lang='sv_SE')
    else:
        dic = pyphen.Pyphen(lang='en_US')
    sep = '$'
    hyphenated_lines = []
    for line in lines:
        next_line = []
        words = line.split(' ')
        for idx, word in enumerate(words):
            word_hyphenated = dic.inserted(word, hyphen=sep)
            parts = word_hyphenated.split(sep)
            for part in parts:
                next_line.append(part)
            if (idx + 1) < len(words):
                next_line.append(' ')
        hyphenated_lines.append(next_line)
    return hyphenated_lines


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


def encode_sequences(tokenizer, max_length, lines):
    """
    encode and pad sequences
    :param tokenizer: Tokenizer
    :param max_length: int Pad the sequence up to this length
    :param lines: list(list(str)
    :return: Numpy array
    """
    # check for None
    for line in lines:
        if None in line: raise ValueError('Found None value in line', line)
    # integer encode sequences
    X = tokenizer.texts_to_sequences(lines)
    # pad sequences with 0 values
    X = pad_sequences(X, maxlen=max_length, padding='post')
    return X


def encode_output(sequences, vocab_size):
    """
    one hot encode target sequence
    :param sequences: Numpy array
    :param vocab_size: int
    :return: reshaped Numpy array?
    """
    ylist = list()
    for sequence in sequences:
        encoded = to_categorical(sequence, num_classes=vocab_size)
        ylist.append(encoded)
    y = array(ylist)
    y = y.reshape(sequences.shape[0], sequences.shape[1], vocab_size)
    return y


models = {
    'simple': simple_model,
    #     'dense': dense_model
}

tokenizers = {
    'a': simple_lines,
    'b': hyphenate_lines,
    'c': word2phrase_lines,
    'd': replace_proper_lines,
    'e': pos_tag_lines
}

# key becomes part of the model name, the value is passed in the optimizer= parameter
optimizers = {
    'sgd': 'sgd',  # default parameters (reported to be more 'stable' than adam)
    'rmsprop': 'sgd',  # default lr=0.001
    'rmsprop2': optimizers.RMSprop(lr=0.01),  # same as previous but with 10x higher learning rate
    'adam': 'adam'
}
