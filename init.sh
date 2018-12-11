#!/usr/bin/env bash

##
# Setup the environment: download required resources and install python packages
##

sudo yum install -y wget unzip hunspell hunspell-devel python

# Install hunspell
ln -s /usr/local/Cellar/hunspell/1.6.2/lib/libhunspell-1.6.0.dylib /usr/local/Cellar/hunspell/1.6.2/lib/libhunspell.dylib
CFLAGS=$(pkg-config --cflags hunspell) LDFLAGS=$(pkg-config --libs hunspell) pip install hunspell

# Install wget
brew install wget
wget http://www.manythings.org/anki/swe-eng.zip
unzip swe-eng.zip

mkdir hunspell
curl -o hunspell/sv_SE.dic https://cgit.freedesktop.org/libreoffice/dictionaries/plain/sv_SE/sv_SE.dic
curl -o hunspell/sv_SE.aff https://cgit.freedesktop.org/libreoffice/dictionaries/plain/sv_SE/sv_SE.aff
curl -o hunspell/en_US.dic https://cgit.freedesktop.org/libreoffice/dictionaries/plain/en/en_US.dic
curl -o hunspell/en_US.aff https://cgit.freedesktop.org/libreoffice/dictionaries/plain/en/en_US.aff

wget http://stp.lingfil.uu.se/~bea/resources/hunpos/suc-suctags.model.gz
gunzip suc-suctags.model.gz

wget https://storage.googleapis.com/google-code-archive-downloads/v2/code.google.com/hunpos/en_wsj.model.gz
gunzip en_wsj.model.gz

# sudo pip install --upgrade pip

pip install -r requirements.txt

sudo python setup.py install

run.sh