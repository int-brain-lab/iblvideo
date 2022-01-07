#!/bin/bash

# update the iblvideo package
cd ~/Documents/PYTHON/iblvideo
git fetch --all
git checkout -f master
git reset --hard
git pull

# update the environment
source ~/Documents/PYTHON/envs/dlcenv/bin/activate
pip uninstall -y ibllib
pip install git+https://github.com/int-brain-lab/ibllib.git@master
pip install -U tensorflow
pip install -U deeplabcut