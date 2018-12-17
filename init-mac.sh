#!/usr/bin/env bash

##
# Setup the environment: download required resources and install python packages
##

# Install hunspell
ln -s /usr/local/Cellar/hunspell/1.6.2/lib/libhunspell-1.6.0.dylib /usr/local/Cellar/hunspell/1.6.2/lib/libhunspell.dylib
CFLAGS=$(pkg-config --cflags hunspell) LDFLAGS=$(pkg-config --libs hunspell) pip install hunspell

# Install wget
brew install wget
wget http://www.manythings.org/anki/swe-eng.zip
unzip -f swe-eng.zip

# Install pre-reqs for brew
brew install autoconf automake libtool gettext
brew link gettext --force

# Note gcc version 8 doesn't seem to work
brew uninstall hunspell
brew install gcc@6 hunspell@1.6.2
# this is needed because -lhunspell is passed to clang, and brew does not maintain such a link
# ln -s /usr/local/Cellar/hunspell/1.6.1/lib/libhunspell-1.6.0.dylib /usr/local/Cellar/hunspell/1.6.1/lib/libhunspell.dylib
# check that libhunspell-1.7.a actually exists
ln -s /usr/local/Cellar/hunspell/1.7.0/lib/libhunspell-1.7.a /usr/local/Cellar/hunspell/1.7.0/lib/libhunspell.a
ln -s /usr/local/Cellar/hunspell/1.7.0/lib/libhunspell-1.7.a /usr/local/Cellar/hunspell/1.7.0/lib/libhunspell-1.7.0.a
echo $(pkg-config --cflags hunspell)
echo $(pkg-config --libs hunspell)
# Build custom hunspell (note: must be in the right conda env!!!)
CFLAGS=$(pkg-config --cflags hunspell) LDFLAGS=$(pkg-config --libs hunspell) CC=/usr/local/bin/gcc-6 MACOSX_DEPLOYMENT_TARGET=10.14  pip install hunspell --force;
