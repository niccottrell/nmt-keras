#!/usr/bin/env bash
set -e

mkdir -p nltk_data
curl -o nltk_data/punkt.zip https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/packages/tokenizers/punkt.zip
curl -o nltk_data/averaged_perceptron_tagger.zip https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/packages/taggers/averaged_perceptron_tagger.zip
unzip nltk_data/*

mkdir -p hunspell
curl -o hunspell/sv_SE.dic https://cgit.freedesktop.org/libreoffice/dictionaries/plain/sv_SE/sv_SE.dic
curl -o hunspell/sv_SE.aff https://cgit.freedesktop.org/libreoffice/dictionaries/plain/sv_SE/sv_SE.aff
curl -o hunspell/en_US.dic https://cgit.freedesktop.org/libreoffice/dictionaries/plain/en/en_US.dic
curl -o hunspell/en_US.aff https://cgit.freedesktop.org/libreoffice/dictionaries/plain/en/en_US.aff

mkdir -p dicts
curl -o dicts/sv.txt https://raw.githubusercontent.com/hermitdave/FrequencyWords/master/content/2016/sv/sv_50k.txt
curl -o dicts/en.txt https://raw.githubusercontent.com/hermitdave/FrequencyWords/master/content/2016/en/en_50k.txt

wget http://stp.lingfil.uu.se/~bea/resources/hunpos/suc-suctags.model.gz
gunzip -f suc-suctags.model.gz

wget https://storage.googleapis.com/google-code-archive-downloads/v2/code.google.com/hunpos/en_wsj.model.gz
gunzip -f en_wsj.model.gz