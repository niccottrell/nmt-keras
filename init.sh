#!/usr/bin/env bash

##
# Setup the environment: download required resources and install python packages
##

# sudo yum install -y wget unzip hunspell hunspell-devel python
# ./init-mac.sh
./init-dl-data.sh

# sudo pip install --upgrade pip

pip install -r requirements.txt

sudo python setup.py install

mkdir checkpoints

run.sh