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

preserve punctation, e.g. commas will change semantics aand exclamation the emphasis

preserve capitalization (> word space) but not for the first word


IDeas for tokenization

1. sub-words (tokenize plural suffixes, compound words) + Preserve spaces  -- useful for German, Swedish, Finnish?  and for small sets
- will reduce word count? (need to confirm)

2. chunk noun phrases (but will actually INCREASE word space?)
https://datascience.stackexchange.com/questions/17294/nlp-what-are-some-popular-packages-for-multi-word-tokenization

3. replace Proper Nouns with placeholders: Name.M, Name.F, Name.Place, Name.Org etc. again to reduce word space

4. add POS-tags to each word before training and encoding/decoding and then remove at the end (larger word space, but less training epochs??)

5. unsupervised sub-word tokenizer with something like https://github.com/google/sentencepiece to create a fixed word space (vocab size) using sub-words

Other direction

