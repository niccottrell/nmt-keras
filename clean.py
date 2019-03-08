import sys
import getopt
import nltk

import config

nltk.download('punkt', download_dir=config.nltk_data)

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
    print("Splitting data files into pairs")
    lines = doc.strip().split('\n')
    pairs = [line.split('\t') for line in lines]
    return pairs

# prepare regex for char filtering
re_print = re.compile('[^%s]' % re.escape(string.printable))

# clean a list of lines
def clean_pairs(lines, langs):
    """
    Cleans and normalizes pairs of inputs
    """
    print("Cleaning %d pairs" % len(lines))
    cleaned = list()
    # prepare translation table for removing punctuation
    # table = str.maketrans('', '', string.punctuation)
    for pair_idx, pair in enumerate(lines):
        clean_pair = list()
        try:
            for idx, sent in enumerate(pair):
                if idx > 1: raise Exception("Weird index: %s" % str(pair))
                lang = langs[idx]
                joined = clean_line(sent, lang)
                clean_pair.append(joined)
            cleaned.append(clean_pair)
        except:
            print("Error processing line: " + "~".join(pair))
        batch_size = 100
        if pair_idx % batch_size == 0:
            print('.',
                  end=('\n' if pair_idx % (80 * batch_size) == 0 else ''))  # print a dot for each batch of input lines
            sys.stdout.flush()
    return array(cleaned)


_intab = "\u201C\u201D\u2018\u2019"
_outtab = "\"\"''"
_trantab = str.maketrans(_intab, _outtab)


def clean_line(sent, lang):
    # normalize unicode characters
    sent = unicodedata.normalize('NFC', sent)
    # sent = unicodedata.normalize('NFD', sent)
    # sent = sent.encode('ascii', 'ignore') # Ignores any non-ascii characters like fancy quotes
    # sent = sent.decode('UTF-8')
    # remove control characters
    sent = remove_control_characters(sent)
    # remove URLs
    sent = re.sub(r'http[s]?://[^\s<>"]+|www\.[^\s<>"]+', ' URL ', sent)
    # remove Twitter handles
    sent = re.sub(r'(^|[^@\w])@(\w{1,15})\b', ' USER ', sent)
    # remove prices
    sent = re.sub(r'[$\u20AC\u00A3]\d+[.,]?\d{0,2}', 'PRICE', sent)
    # change any remaining euro symbols to dollar :(
    sent = re.sub(r'[\u20AC\u00A3]', '$', sent)
    # replace special quotes with ascii quotes
    sent = sent.translate(_trantab)
    # replace Unicode characters with ascii equivalents
    # if (lang == 'sv'): sent = unidecode(sent) # since the POS tagger for Swedish doesn't accept utf8 # unidecode removes ASCII Swedish characters too :(
    # alter language-specific abbreviations etc. [sic]
    sent = prepare_line(sent, lang, 'lookup')
    # # tokenize more intelligently (TODO should this just use WordPunctTokenizer too?)
    # tokens = word_tokenize(sent, 'english' if lang == 'en' else 'swedish')
    # # convert to lowercase
    # # line = [word.lower() for word in line]
    # # remove punctuation from each token
    # # line = [word.translate(table) for word in line]
    # remove non-ascii chars form each token
    # sent = [re_print.sub('', w) for w in sent]
    # # remove tokens with non-alphas in them (would remove exclamation, question marks etc.)
    # # sent = [word for word in sent if word.isalpha()]
    # # store as string
    # joined = ' '.join(tokens)
    # return joined
    return sent


if __name__ == '__main__':
    # Process arguments
    source = ''
    cmd_example = 'clean.py --source (anki|OpenSubtitles)'
    try:
        opts, args = getopt.getopt(sys.argv[1:], "h", ["source="])
    except getopt.GetoptError:
        print(cmd_example)
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print(cmd_example)
            sys.exit()
        elif opt in ("--source"):
            source = arg
    # load dataset
    if source == "anki":
        filename = lang2 + '.txt'
        doc = load_doc(filename)
        # split into language pairs
        pairs = to_pairs(doc)
    else:
        doc_lang1 = load_doc("data/os/OpenSubtitles.en-sv.en")
        doc_lang2 = load_doc("data/os/OpenSubtitles.en-sv.sv")
        # Merge into paris
        print("Merging data files into pairs")
        lines1 = doc_lang1.strip().split('\n')
        lines2 = doc_lang2.strip().split('\n')
        pairs = []
        for idx, line in enumerate(lines1):
            line2 = lines2[idx]
            diff = abs(len(line) - len(line2))
            score = diff / max(len(line), len(line2))
            if score < 0.2:
                pairs.append([line, line2])
    # clean sentences
    clean_pairs = clean_pairs(pairs, ['en', 'sv'])
    # save clean pairs
    config.data.save_clean_data(clean_pairs)
    # spot check
    for i in range(100):
        print('[%s] => [%s]' % (clean_pairs[i, 0], clean_pairs[i, 1]))
