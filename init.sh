#!/usr/bin/env bash

yum install wget unzip

wget http://www.manythings.org/anki/swe-eng.zip
unzip swe-eng.zip

wget https://cgit.freedesktop.org/libreoffice/dictionaries/tree/sv_SE/sv_SE.dic
wget https://cgit.freedesktop.org/libreoffice/dictionaries/tree/en/en_US.dic

wget http://stp.lingfil.uu.se/~bea/resources/hunpos/suc-suctags.model.gz
gunzip suc-suctags.model.gz

wget https://storage.googleapis.com/google-code-archive-downloads/v2/code.google.com/hunpos/en_wsj.model.gz
gunzip en_wsj.model.gz

pip install -r requirements.txt

python setup.py
python clean.py
python prepare.py
python train.py
python evaluate.py