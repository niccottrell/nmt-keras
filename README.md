# README #

This is a test for NMT using Keras

# Environment

## Install Conda

## Install hunspell and dictionaries

### On Mac

```
brew install hunspell
cd ~/Library/Spelling
wget http://cgit.freedesktop.org/libreoffice/dictionaries/plain/en/en_US.aff
wget http://cgit.freedesktop.org/libreoffice/dictionaries/plain/en/en_US.dic
wget https://cgit.freedesktop.org/libreoffice/dictionaries/plain/sv_SE/sv_SE.aff
wget https://cgit.freedesktop.org/libreoffice/dictionaries/plain/sv_SE/sv_SE.dic
```

### Install POS-Tagging for Swedish

```
wget http://stp.lingfil.uu.se/~bea/resources/hunpos/suc-suctags.model.gz
gunzip suc-suctags.model.gz
```
```
wget https://storage.googleapis.com/google-code-archive-downloads/v2/code.google.com/hunpos/en_wsj.model.gz
gunzip en_wsj.model.gz
```
## Install Python packages

Use pip to read from requirements.txt


Initial code from https://machinelearningmastery.com/develop-neural-machine-translation-system-keras/

Download content at http://www.manythings.org/anki/swe-eng.zip

From http://opus.nlpl.eu/Europarl.php
http://opus.nlpl.eu/download.php?f=Europarl/en-sv.xml.gz
or
http://opus.nlpl.eu/download.php?f=Europarl/en-sv.tmx.gz


http://opus.nlpl.eu/download.php?f=Wikipedia/de-en.txt.zip
http://opus.nlpl.eu/download.php?f=GlobalVoices/en-sv.tmx.gz
http://www.manythings.org/anki/
http://www.manythings.org/anki/swe-eng.zip

On MacBook Pro plugged in, epochs start at about 45s with 9000 samples

Install tensorflow from source
https://www.tensorflow.org/install/install_sources
https://docs.nvidia.com/cuda/cuda-installation-guide-mac-os-x/index.html
https://metakermit.com/2017/compiling-tensorflow-with-gpu-support-on-a-macbook-pro/
https://medium.com/@dhillonkannabhiran/installing-tensorflow-1-4-with-gpu-support-cuda-8-cudnn-6-on-os-x-10-12-6-dca75235417c
Install CUDA drivers from https://docs.nvidia.com/cuda/cuda-installation-guide-mac-os-x/index.html
e.g. https://developer.nvidia.com/compute/cuda/9.1/Prod/network_installers/cuda_9.1.128_mac_network
On mac
brew cask install nvidia-cuda
pip install --upgrade pip setuptools


# Download dictionaries

https://cgit.freedesktop.org/libreoffice/dictionaries/tree/sv_SE/sv_SE.dic
https://cgit.freedesktop.org/libreoffice/dictionaries/tree/en/en_US.dic

# Libraries used

## Pyphen

Pyphen is a pure Python module to hyphenate text using included or external Hunspell hyphenation dictionaries.

http://pyphen.org/


Check version:

# python

```
import tensorflow as tf
hello = tf.constant('Hello, TensorFlow!')
sess = tf.Session()
print(sess.run(hello))
tf.VERSION
```

Schiller [2005] find that in a newspaper corpus [in German?], 5.5% of all tokens and 43% of all types were compounds.
Anne Schiller. German compound analysis with wfsc. In International Work- shop on Finite-State Methods and Natural Language Processing, pages 239–246. Springer, 2005.

The German word for eye drops is Augentropfen, consisting of Auge (eye), tropfen (drop), and an n in the middle.

in a few languages, such as German, Dutch, Hungarian, Greek, and the Scandinavian languages the resulting words, or compounds, are written as a single word without any special characters or whitespace in-between.
The most frequent use of compounding by far [Baroni et al., 2002] is com- pounds consisting of two nouns, but adjectives and verbs form compounds as well.

!!! A Schulbuch (school book) consists of Schule (school) and Buch (book), but the final e of Schule is removed when using it as the first part of a compound.

Koehn and Knight [2002] learn splitting rules from both monolingual as well as parallel corpora. They generate all possible splits of a given word, and take the one that maximizes the geometric mean of the word frequencies of its parts, although they find that this process often leads to both oversplitting words into more common parts (e.g. Freitag (friday) into frei (free) and tag (day)), as well as not splitting some words that should be split, because the compound is more frequent than the geometric mean of the frequencies of its parts.

Soricut and Och [2015] use vector representations of words to uncover morpho- logical processes in an unsupervised manner. Their method is language-agnostic, and can be applied to rare words or out-of-vocabulary tokens (OOVs).
Morphological transformations (e.g. rained = rain + ed) are learned from −−→ −−−→ the word vectors themselves, by observing that, for instance, rain is to rained
 
# Corpora
 
 The EMEA corpus [Tiedemann, 2009] is a parallel corpus based on documents by the European Medicines Agency3. 
 
as walk is to walked.

They find that the ASV toolbox [Biemann et al., 2008] delivers the best results.

# Work

Started with https://machinelearningmastery.com/develop-neural-machine-translation-system-keras/

# Problems

preserve punctuation, e.g. commas will change semantics aand exclamation the emphasis

preserve capitalization (> word space) but not for the first word


# Ideas for tokenization

* Insert a pre-token to denote context, H for heading, W for written sentence, S for spoken sentence, L for label (like a button)?
(or is having punctuation at the end enough?)

Deal with contractions:
https://github.com/kootenpv/contractions

* Lowercase first word in sentence IF it's not a proper noun (but how)

1. sub-words (tokenize plural suffixes, compound words) + Preserve spaces  -- useful for German, Swedish, Finnish?  and for small sets

- will reduce word count? (need to confirm)

syllables: https://gist.github.com/bradmerlin/5693904 (English specific)
naive: split after vowel (or multiple vowels) - for languages without vowels (Arabic?, Chinese) just split after each characters
Masters thesis by Jonathan Oberl̈ander "Splitting Word Compounds"

2. chunk noun phrases (but will actually INCREASE word space?)
https://datascience.stackexchange.com/questions/17294/nlp-what-are-some-popular-packages-for-multi-word-tokenization
OR
https://github.com/travisbrady/word2phrase (python port)

3. replace Proper Nouns with placeholders: Name.M, Name.F, Name.Place, Name.Org etc. again to reduce word space

4. add POS-tags to each word before training and encoding/decoding and then remove at the end (larger word space, but less training epochs??)
     e.g. loved.V, love.N  etc.
     what about ['he', 'PRP', 'loved', 'V', 'with', 'P', 'love', 'N'...] ... or with that make it worse
     what about ['he', 'PRP', 'love', 'V.Past', 'with', 'P', 'love', 'N'...] (might be very useful to reduce word space for inflected languages like German, Finnish, Icelandic)
     See https://stackoverflow.com/questions/25534214/nltk-wordnet-lemmatizer-shouldnt-it-lemmatize-all-inflections-of-a-word
5. unsupervised sub-word tokenizer with something like https://github.com/google/sentencepiece to create a fixed word space (vocab size) using sub-words

Other direction

## Suffixes
We extract a list of suffixes for each language from Wiktionary (a sister project of Wikipedia):
We simply take all page titles in the Category:language prefixes and Category:language suffixes6 and remove the dash at the beginning of each page title.


## Splitting

Language
German
s e en nen ens es ns er
Hemdsa ̈rmel Hundehu ̈tte Strahlentherapie Lehrerinnenausbildung Herzenswunsch Haaresbreite Willensbildung Schilderwald

Swedish
s
utg ̊angsdatum

Hungarian
o ́  o ̋ ba  ́ıt ̋o es s i a
old ́oszer gyu ̋jto ̋doboz forgalombahozatali  ́edes ́ıto ̋szerk ́ent k ́ekesbarna szu ̈rk ́esbarna  ́ızu ̈letifa ́jdalom koraszu ̈l ̈ott
    

# TODO:
https://github.com/lvapeab/nmt-keras


Useful tools
https://bitbucket.org/fhaxbox66/pyhyphen
https://cran.r-project.org/web/packages/hunspell/vignettes/intro.html

# Papers

Parallel Corpora, Parallel Worlds: Selected Papers from a Symposium on Parallel and Comparable Corpora at Uppsala University, Sweden, 22-23 April, 1999
https://books.google.de/books?id=-XCX7SRubY4C&dq=swedish+english+sentence+pairs&source=gbs_navlinks_s
