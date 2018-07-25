#!/usr/bin/env bash

##
# Runs the entire workflow (assuming that all the initialization has worked correctly)
##

python clean.py
python prepare.py
python train.py
python evaluate.py