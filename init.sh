#!/usr/bin/env bash

##
# Setup the environment: download required resources and install python packages
##

sudo yum install -y wget unzip hunspell hunspell-devel python

wget http://www.manythings.org/anki/swe-eng.zip
unzip -f swe-eng.zip

mkdir hunspell
curl -o hunspell/sv_SE.dic https://cgit.freedesktop.org/libreoffice/dictionaries/plain/sv_SE/sv_SE.dic
curl -o hunspell/sv_SE.aff https://cgit.freedesktop.org/libreoffice/dictionaries/plain/sv_SE/sv_SE.aff
curl -o hunspell/en_US.dic https://cgit.freedesktop.org/libreoffice/dictionaries/plain/en/en_US.dic
curl -o hunspell/en_US.aff https://cgit.freedesktop.org/libreoffice/dictionaries/plain/en/en_US.aff

wget http://stp.lingfil.uu.se/~bea/resources/hunpos/suc-suctags.model.gz
gunzip -f suc-suctags.model.gz

wget https://storage.googleapis.com/google-code-archive-downloads/v2/code.google.com/hunpos/en_wsj.model.gz
gunzip -f en_wsj.model.gz

# sudo pip install --upgrade pip

sudo pip install -r requirements.txt

sudo python setup.py install

mkdir checkpoints

run.sh