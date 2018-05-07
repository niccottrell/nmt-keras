# README #

This is a test for NMT using Keras

Initial code from https://machinelearningmastery.com/develop-neural-machine-translation-system-keras/

Download content at http://www.manythings.org/anki/swe-eng.zip

From http://opus.nlpl.eu/Europarl.php
http://opus.nlpl.eu/download.php?f=Europarl/en-sv.xml.gz
or
http://opus.nlpl.eu/download.php?f=Europarl/en-sv.tmx.gz


http://opus.nlpl.eu/download.php?f=Wikipedia/de-en.txt.zip


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


Check version:

# python

```
import tensorflow as tf
hello = tf.constant('Hello, TensorFlow!')
sess = tf.Session()
print(sess.run(hello))
tf.VERSION
```

Problems

preserve punctuation, e.g. commas will change semantics aand exclamation the emphasis

preserve capitalization (> word space) but not for the first word


IDeas for tokenization
* Insert a pre-token to denote context, H for heading, W for written sentence, S for spoken sentence, L for label (like a button)?
(or is having punctuation at the end enough?)

Deal with contractions:
https://github.com/kootenpv/contractions

* Lowercase first word in sentence IF it's not a proper noun (but how)

1. sub-words (tokenize plural suffixes, compound words) + Preserve spaces  -- useful for German, Swedish, Finnish?  and for small sets

- will reduce word count? (need to confirm)

syllables: https://gist.github.com/bradmerlin/5693904 (English specific)
naive: split after vowel (or multiple vowels) - for languages without vowels (Arabic?, Chinese) just split after each characters

2. chunk noun phrases (but will actually INCREASE word space?)
https://datascience.stackexchange.com/questions/17294/nlp-what-are-some-popular-packages-for-multi-word-tokenization

3. replace Proper Nouns with placeholders: Name.M, Name.F, Name.Place, Name.Org etc. again to reduce word space

4. add POS-tags to each word before training and encoding/decoding and then remove at the end (larger word space, but less training epochs??)
     e.g. loved.V, love.N  etc.
     what about ['he', 'PRP', 'loved', 'V', 'with', 'P', 'love', 'N'...] ... or with that make it worse
     what about ['he', 'PRP', 'love', 'V.Past', 'with', 'P', 'love', 'N'...] (might be very useful to reduce word space for inflected languages like German, Finnish, Icelandic)
     See https://stackoverflow.com/questions/25534214/nltk-wordnet-lemmatizer-shouldnt-it-lemmatize-all-inflections-of-a-word
5. unsupervised sub-word tokenizer with something like https://github.com/google/sentencepiece to create a fixed word space (vocab size) using sub-words

Other direction


TODO:
https://github.com/lvapeab/nmt-keras

