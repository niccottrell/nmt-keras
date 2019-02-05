# README

The project is designed test various models for translation Swedish sentences to English. 
It uses Keras to define the models, and Tensorflow as the backend. Various models are tested, along with different tokenizing options and optimizers.
The results are scored using the BLEU unigram method which looks for the occurences of all keywords but doesn't consider word order. The highest results achieved so far is a score of 0.23 out of a maximum of 1.00. 
The low scores are probably based on the limited training set. 

# Environment

The development of this project assumes Python 3.6, and the easiest way to set up the correct packages is via Anaconda:

## Hardware

The models train much better with an Nvidia GPU. This is a standard Tensorflow requirement for performant neural net training, and needs the CUDA drivers (amongst others) to work.

## Install Conda

Download and install via https://www.anaconda.com/download/

```sh
conda create --name nmt-keras python=3.6
source activate nmt-keras
```

You may need to activate extra channels:

```sh
conda config --add channels conda-forge 
conda install <package-name>
```

## Windows considerations

Ensure that you are running the 64-bit version of Python.

You may need to increase the virtual memory available to the OS via the Advanced Settings > Performance Settings > Advanced > Virtual memory settings. 

## Install hunspell and dictionaries

### On Mac

```sh
brew install hunspell
cd ~/Library/Spelling
wget http://cgit.freedesktop.org/libreoffice/dictionaries/plain/en/en_US.aff
wget http://cgit.freedesktop.org/libreoffice/dictionaries/plain/en/en_US.dic
wget https://cgit.freedesktop.org/libreoffice/dictionaries/plain/sv_SE/sv_SE.aff
wget https://cgit.freedesktop.org/libreoffice/dictionaries/plain/sv_SE/sv_SE.dic
```

### On Windows

Download https://storage.googleapis.com/google-code-archive-downloads/v2/code.google.com/hunpos/hunpos-1.0-win.zip
to thirdparty\hunpos-win

### Install POS-Tagging models for Swedish

```
wget http://stp.lingfil.uu.se/~bea/resources/hunpos/suc-suctags.model.gz
gunzip suc-suctags.model.gz
```
```
wget https://storage.googleapis.com/google-code-archive-downloads/v2/code.google.com/hunpos/en_wsj.model.gz
gunzip en_wsj.model.gz
```

## Install Python packages

Use `pip` to read from `requirements.txt`

## Download training set 

Download content at http://www.manythings.org/anki/swe-eng.zip

## Automatic installer

Most of the steps above should work on Mac/Linux by running the `init.sh` script.

## Libraries used

### Pyphen

[Pyphen](http://pyphen.org/) is a pure Python module to hyphenate text using included or external Hunspell hyphenation dictionaries.

### Graphviz

On Mac, install with `brew install graphviz`

# python

```
import tensorflow as tf
hello = tf.constant('Hello, TensorFlow!')
sess = tf.Session()
print(sess.run(hello))
tf.VERSION
```

## Background information

Many of the assumptions of this project for Swedish are similiar to any German to English translation model, 
since Swedish also heavily uses noun compounding, although Swedish's morphology is much simpler than German. 

Schiller [2005] find that in a newspaper corpus in German, 5.5% of all tokens and 43% of all types were compounds.
Anne Schiller.  For example, the German word for eye drops is _Augentropfen_, consisting of Auge (eye), tropfen (drop), and an n in the middle.  A _Schulbuch_ (school book) consists of _Schule_ (school) and _Buch_ (book), but the final e of Schule is removed when using it as the first part of a compound.


In a few languages, such as German, Dutch, Hungarian, Greek, and the Scandinavian languages the resulting words, or compounds, are written as a single word without any special characters or whitespace in between.
The most frequent use of compounding by far [Baroni et al., 2002] is compounds consisting of two nouns, but adjectives and verbs form compounds as well.


[Koehn and Knight, 2003](http://www.aclweb.org/anthology/E03-1076) learn splitting rules from both monolingual as well as parallel corpora. They generate all possible splits of a given word, and take the one that maximizes the geometric mean of the word frequencies of its parts, although they find that this process often leads to both oversplitting words into more common parts (e.g. _Freitag_ (friday) into frei (free) and tag (day)), as well as not splitting some words that should be split, because the compound is more frequent than the geometric mean of the frequencies of its parts.

[Soricut and Och, 2015](http://jrmeyer.github.io/misc/2016/01/21/Soricut-and-Och-2015.html) use vector representations of words to uncover morpho- logical processes in an unsupervised manner. Their method is language-agnostic, and can be applied to rare words or out-of-vocabulary tokens (OOVs). Morphological transformations (e.g. rained = rain + ed) are learned from the word vectors themselves.
 

# Problems

Much of the example code online uses over-simplication to achieve any useful results. This often involves lower-casing all text, 
removing punctuation and accents etc. For this project I wanted to see what was possible with commodity GPUs while preserving punctuation and case.

I find this important since (for example) commas will change semantics and exclamation the emphasis. 
And obviously capitalization normally marks proper nouns. Therefore, part of the preparation phase is to normalize the input, and only lowercase the first word in a sentece if it is _not_ a proper noun. 
This reduces the word space since in principle any word would appear twice in the training set: once in the middle of the sentence, e.g. _eating_ in "I like eating", vs. _Eating_ in "Eating is a necessity". 
This could increase the model complexity and required training by a power of 2. 


## Tokenization

This project contains several tokenizer options in `tokenizers.py`:

|Tokenizer class | Description|
|-----------|---------------------|
|`SimpleLines`| Tokenize on spaces and punctuation, and keep _punctuation_ but _not_ spaces  | 
|`Hyphenate`| Use a [hypenate library](https://bitbucket.org/fhaxbox66/pyhyphen) to break do `SimpleLines` then break down longer words into sub-parts to reduce the dimensionality |
|`Word2Phrase`| Tokenize as above, but combine popular phrases into a single token, e.g. "good-bye" and "Eiffel Tower" will be single tokens.  |
|`ReplaceProper`| Tokenize, but replace proper nouns with a placeholder so we don't pollute the model with possibly unlimited [named entities](https://en.wikipedia.org/wiki/Named-entity_recognition).  |
|`PosTag`| Tokenize as above, but run a [part-of-speech tagger](https://en.wikipedia.org/wiki/Part-of-speech_tagging) on each sentence. This will result in token like "duck.VF" and "duck.NN" for verbs and nouns.    |
|`LetterByLetter`| Tokenize the sentence into individual words, preserving all spaces and punctuation. This gives a very small input space but should be possible for an  attention-based model.  |


# Acknowledgements

- Initial code was from https://machinelearningmastery.com/develop-neural-machine-translation-system-keras/

- [Parallel Corpora, Parallel Worlds: Selected Papers from a Symposium on Parallel and Comparable Corpora](https://books.google.de/books?id=-XCX7SRubY4C&dq=swedish+english+sentence+pairs&source=gbs_navlinks_s) at Uppsala University, Sweden, 22-23 April, 1999

- Anne Schiller. 2005. German compound analysis with wfsc. In International Workshop on Finite-State Methods and Natural Language Processing, pages 239–246. Springer.

# Future work

The results will probably be better with training taken from the larger Europarl, Wikipedia or GlobalVoices corpora:

- [Europarl](http://opus.nlpl.eu/download.php?f=Europarl/en-sv.tmx.gz)
- [Wikipedia](http://opus.nlpl.eu/download.php?f=Wikipedia/sv-en.txt.zip)
- [Global Voices](http://opus.nlpl.eu/download.php?f=GlobalVoices/en-sv.tmx.gz)
- [EMEA corpus](http://opus.nlpl.eu/EMEA.php) is a parallel corpus based on documents by the European Medicines Agency. 
 
Other avenues to explore:

- Add another tokenization option to denote context. Perhaps add a pre-token such as `H` for heading, `W` for written sentence, `S` for spoken sentence, `L` for label (like a button).

- Deal with contractions. See https://github.com/kootenpv/contractions

- Test an unsupervised sub-word tokenizer with something like [sentencepiece](https://github.com/google/sentencepiece) to create a fixed word space (vocab size) using sub-words

- Test a [syllable-based tokenizer](https://gist.github.com/bradmerlin/5693904) (English specific)
or a naive language-general approach of splitting after every vowel (or multiple vowels). For languages without vowels (Arabic, Chinese) just split after each character.
Masters thesis by Jonathan Oberl̈ander "[Splitting Word Compounds](https://core.ac.uk/display/97803281)"
 
- Test a suffix-tokenization method. We could extract a list of suffixes for each language from [Wiktionary](https://www.wiktionary.org) (a sister project of Wikipedia):
We simply take all page titles in the [Category:language prefixes](https://en.wiktionary.org/wiki/Category:English_prefixes) and [Category:language suffixes](https://en.wiktionary.org/wiki/Category:English_suffixes) and remove the dash at the beginning of each page title.


