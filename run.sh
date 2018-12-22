#!/usr/bin/env bash
set -e

##
# Runs the entire workflow (assuming that all the initialization has worked correctly)
##

python clean.py > clean.out
python prepare.py > prepare.out
python train.py > train.out
python evaluate.py > evaluate.out