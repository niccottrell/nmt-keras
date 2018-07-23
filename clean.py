import string
import re
from pickle import dump
from unicodedata import normalize
from numpy import array
from nltk.tokenize import word_tokenize

import nltk

nltk.download('punkt')

from helpers import *


# load doc into memory
def load_doc(filename):
    # open the file as read only
    file = open(filename, mode='rt', encoding='utf-8')
    # read all text
    text = file.read()
    # close the file
    file.close()
    return text


# split a loaded document into sentences
def to_pairs(doc):
    lines = doc.strip().split('\n')
    pairs = [line.split('\t') for line in lines]
    return pairs


# clean a list of lines
def clean_pairs(lines, langs):
    """
    Cleans and normalizes pairs of inputs
    """
    cleaned = list()
    # prepare regex for char filtering
    re_print = re.compile('[^%s]' % re.escape(string.printable))
    # prepare translation table for removing punctuation
    # table = str.maketrans('', '', string.punctuation)
    for pair in lines:
        clean_pair = list()
        for idx, sent in enumerate(pair):
            lang = langs[idx]
            # normalize unicode characters
            # sent = normalize('NFD', sent)
            # line = line.encode('ascii', 'ignore')
            # line = line.decode('UTF-8')
            # remove control characters
            sent = remove_control_characters(sent)
            # remove URLs
            sent = re.sub(r'http[s]?://[^\s<>"]+|www\.[^\s<>"]+', ' URL ', sent)
            # remove Twitter handles
            sent = re.sub(r'(^|[^@\w])@(\w{1,15})\b', ' USER ', sent)
            # alter language-specific abbreviations etc. [sic]
            sent = prepare_line(sent, lang, 'lookup')
            # tokenize more intelligently (TODO should this just use WordPunctTokenizer too?)
            tokens = word_tokenize(sent, 'english' if lang == 'en' else 'swedish')
            # convert to lowercase
            # line = [word.lower() for word in line]
            # remove punctuation from each token
            # line = [word.translate(table) for word in line]
            # remove non-ascii chars form each token
            # sent = [re_print.sub('', w) for w in sent]
            # remove tokens with non-alphas in them (would remove exclamation, question marks etc.)
            # sent = [word for word in sent if word.isalpha()]
            # store as string
            clean_pair.append(' '.join(tokens))
        cleaned.append(clean_pair)
    return array(cleaned)


# load dataset
filename = lang2 + '.txt'
doc = load_doc(filename)
# split into english-german pairs
pairs = to_pairs(doc)
# clean sentences
clean_pairs = clean_pairs(pairs, ['en', 'sv'])
# save clean pairs to file
save_clean_data(clean_pairs, 'eng-' + lang2 + '.pkl')
# spot check
for i in range(100):
    print('[%s] => [%s]' % (clean_pairs[i, 0], clean_pairs[i, 1]))
