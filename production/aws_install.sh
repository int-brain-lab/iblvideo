#!/bin/bash

# Use p2.xlarge instance with Deep Learning AMI (Ubuntu 18.04) Version 36.0
# Install requirements for IBL https://docs.google.com/document/d/1_31_4oafqSY8YsOweMXA2jMJOT_4hBrcF1YQWPLCYnc/edit#
# As well as requirements for DLC https://github.com/int-brain-lab/iblvide

# First remove anaconda to make space, we will use our own PYTHON
rm -rf /home/ubuntu/anaconda3

# Make a directory for the virutalenv (consistent with server install)
mkdir -p ~/Documents/PYTHON
cd ~/Documents/PYTHON

# Install Ubuntu packages
sudo apt install -y tmux
sudo apt install -y virtualenv
sudo apt install -y git
sudo apt install -y python3-dev build-essential
sudo apt install -y python3-distutils
sudo apt install -y python3-tk
sudo apt install -y python3-pip
sudo apt install -y sox
sudo apt install -y tk-dev
sudo apt install -y libnotify-dev libsdl1.2-dev
sudo apt install -y ffmpeg

# Install Python
# Here we need to use Python 3.7 as DLC does not currently tensorflow 2, and tensorflow 1 does not work on Python 3.8
sudo apt update -y
sudo apt install -y software-properties-common
sudo add-apt-repository -y ppa:deadsnakes/ppa
sudo apt install -y python3.7 python3.7-dev

# Create virtual env
mkdir -p ~/Documents/PYTHON/envs
cd ~/Documents/PYTHON/envs
virtualenv dlcenv --python=python3.7
source ~/Documents/PYTHON/envs/dlcenv/bin/activate

#Install dlc and ibllib (in this order, ignore the numpy version warning for deeplabcut)
pip install https://extras.wxpython.org/wxPython4/extras/linux/gtk3/ubuntu-18.04/wxPython-4.0.7-cp37-cp37m-linux_x86_64.whl
pip install tensorflow-gpu==1.13.1
pip install deeplabcut
pip install ibllib

# clone iblvideo
mkdir ~/dlc
cd ~/dlc
git clone https://github.com/int-brain-lab/iblvideo.git --branch aws
echo
echo "Installation finished."
echo "Run source /home/ubuntu/dlc/iblvideo/production/sourcefile to set paths."
echo
