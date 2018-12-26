"""
This module helps us examine/debug the different tokenizers
"""
import abc
import string

import pyphen

from helpers import create_tokenizer, max_length, lang2, load_clean_sentences
from nltk.tokenize import WordPunctTokenizer

import re  # standard regex system
import traceback
from nltk.tag.hunpos import HunposTagger

re_punct_digits = re.compile("[\d€{}]+$".format(re.escape(string.punctuation)))

def is_noun(word, lang):
    tuples = pos_tag_tokens([word], lang)
    pos = tuples[0][1].decode('utf-8')
    return True if pos[0] == 'N' else False


class BaseTokenizer(object):

    def __init__(self, name):
        """
        :param name: the tokenizer name, e.g. "simple_lines"

        """
        self.name = name

    @abc.abstractmethod
    def tokenize_one(self, source, lang):
        """
        :param source: the input natural language
        :type source: string
        :return: string the most likely translation as a string
        :rtype: list(string)
        """
        return None

    @abc.abstractmethod
    def tokenize(self, lines, lang):
        """
        :param lines: the input natural language
        :type lines: list(string)
        :return: string the most likely translation as a string
        :rtype: list(string)
        """
        return None

    @abc.abstractmethod
    def join(self, tokens, lang):
        """
        Joins tokens back to natural language,
        :param tokens: the tokens from the decoder
        :type tokens: list(string)
        :param lang: The language of the tokens
        :return: string the recombined natural sentence
        :rtype: string)
        """
        return None


class SimpleLines(BaseTokenizer):
    wpt = WordPunctTokenizer()  # See http://text-processing.com/demo/tokenize/ for examples

    def __init__(self):
        BaseTokenizer.__init__(self, 'simple_lines')

    def tokenize_one(self, line, lang):
        """
        Tokenize lines by whitespace but discard whitespace
        :param line: string
        :return: list(str)
        """
        return self.wpt.tokenize(line)

    def tokenize(self, lines, lang):
        """
        Tokenize lines by whitespace but discard whitespace
        :param lines: list(str)
        :return: list(list(str)
        """
        tokenized_lines = []
        for line in lines:
            tokenized_lines.append(self.wpt.tokenize(line))
        return tokenized_lines

    def join(self, tokens, lang):
        return ' '.join(tokens).strip()

simple_lines = SimpleLines()

class LetterByLetter(BaseTokenizer):

    def __init__(self):
        BaseTokenizer.__init__(self, 'let2let')

    def tokenize_one(self, line, lang):
        """
        Tokenize lines by breaking into individual letters, keeping all spaces and punctuation
        :param line: str
        :return: list(str)
        """
        chars = []
        for t, char in enumerate(line):
            chars.append(char)
        return chars

    def tokenize(self, lines, lang):
        """
        Tokenize lines by breaking into individual letters, keeping all spaces and punctuation
        :param lines: list(str)
        :return: list(list(str)
        """
        tagged_lines = []
        for line in lines:
            chars = []
            for t, char in enumerate(line):
                chars.append(char)
            tagged_lines.append(chars)
        return tagged_lines

    def join(self, tokens, lang):
        return ''.join(tokens).strip()


# keep HunposTaggers loaded
ht_cache = {}


def pos_tag_tokens(line, lang):
    """
    Do POS-tagging but return tuples for each input word
    :type line: list(str) An already tokenized line
    :type lang: str The language (2-letter code)
    """
    iso3 = ('sve' if lang[:2] == 'sv' else 'eng')
    if iso3 in ht_cache:
        ht = ht_cache[iso3]
    else:
        if iso3 == 'eng':
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


class PosTag(BaseTokenizer):

    def __init__(self):
        BaseTokenizer.__init__(self, 'pos_tag')

    def tag(self, line, lang='en'):
        """
        Append part-of-speech tags to each word, keeping the units as a list
        :type line: list(str)
        :type lang: str
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
                    result.append(tuple[0] + "." + pos[:2])  # Only take the first 2 letters of the POS, e.g. 'NN_UTR_SIN_DEF_NOM' -> 'NN'
            return result
        except:
            print('Error tagging line `%s` in %s' % (line, lang))
            traceback.print_exc()

    def tokenize(self, lines, lang):
        """Post tag lines in bulk
        :param lines: list(str)
        :param lang: The 2-letter language code
        :return: list(list(str))
        """
        tokenized_lines = simple_lines.tokenize(lines, lang)
        tagged_lines = []
        for line in tokenized_lines:
            tagged_lines.append(self.tag(line, lang))
        return tagged_lines

    def join(self, tokens, lang):
        # todo: remove tag part before joining
        return ' '.join(tokens).strip()


class ReplaceProper(BaseTokenizer):


    def __init__(self):
        BaseTokenizer.__init__(self, 'replace_proper')

    def tokenize_one(self, line, lang):
        """
        Append part-of-speech tags to each word, keeping the units as a list
        :type line: list(str)
        :type lang: str 2-letter language code
        :return list(str)
        """
        inside_proper = False
        proper_idx = 1
        try:
            tuples = pos_tag_tokens(line, lang)
            result = []
            for tuple in tuples:
                if re_punct_digits.match(tuple[0]):  # It's just punctuation/digits/symbols
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


    def tokenize(self, lines, lang):
        """Post tag lines in bulk
        :param lines: list(str)
        :param lang: The 2-letter language code
        :return: list(list(str))
        """
        tokenized_lines = simple_lines.tokenize(lines, lang)
        tagged_lines = []
        for line in tokenized_lines:
            tagged_lines.append(self.tokenize_one(line, lang))
        return tagged_lines


class Word2Phrase(BaseTokenizer):

    def __init__(self):
        BaseTokenizer.__init__(self, 'word2phrase')

    def tokenize(self, lines, lang):
        """
        Convert all line inputs to tokenized chunked phrases
        :param lines: list(str) The lines to process this time
        :param lang: str The language code (but ignored for now)
        :return: list(list(str))
        """
        # Pre-split the lines
        tokenized_lines = simple_lines.tokenize(lines, lang)

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
        dataset_tokenized = simple_lines.tokenize(dataset_thislang, lang)

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

class Hyphenate(BaseTokenizer):

    dic_sv = pyphen.Pyphen(lang='sv_SE')
    dic_en = pyphen.Pyphen(lang='en_US')

    def __init__(self):
        BaseTokenizer.__init__(self, 'hyphenate')

    def get_dic(self, lang):
        if lang == 'sv' or lang == 'sve' or lang == 'swe':
            return self.dic_sv
        else:
            return self.dic_en

    def hyphenate(self, word, lang):
        """
        Hyphenates a single word
        :param word:
        :param str lang:
        :return:
        """
        sep = '$'
        dic = self.get_dic(lang)
        return dic.inserted(word, hyphen=sep).split(sep)

    def tokenize_one(self, line, lang):
        """
        Hyphenate all lines and wrap in a tokenizer
        :param line: str
        :param lang: the 2-letter language code
        :return: list(str)
        """
        sep = '$'
        dic = self.get_dic(lang)
        next_line = []
        words = line.split(' ')
        for idx, word in enumerate(words):
            word_hyphenated = dic.inserted(word, hyphen=sep)
            parts = word_hyphenated.split(sep)
            for part in parts:
                next_line.append(part)
            if (idx + 1) < len(words):
                next_line.append(' ')
        return next_line

    def tokenize(self, lines, lang):
        """
        Hyphenate all lines and wrap in a tokenizer
        :param lines: list(str)
        :param lang: the 2-letter language code
        :return: list(list(str))
        """
        sep = '$'
        dic = self.get_dic(lang)
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