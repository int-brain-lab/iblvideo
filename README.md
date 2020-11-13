
## Video acquisition in IBL 

Mice are filmed in training rigs and recording rigs. In training rigs there is only one side camera recording at full resolution (1280x1024) and 30 Hz. In the recording rigs, there are three cameras, one called 'left' at full resolution 1280x1024 and 60 Hz filming the mouse from one side, one called 'right' at half resolution (640x512) and 150 Hz filming the mouse symmetrically from the other side, and one called 'body' filming the trunk of the mouse from above.    
 
## Feature-tracking using DeepLabCut	 	 

DeepLabCut (DLC) is used for markerless tracking of animal parts in these videos, returning for each frame x,y coordinates in px for each point and a likelihood (how certain was the network to have found that point in a specific frame). For each side video we track the following points:	

`'pupil_top_r', 'pupil_right_r', 'pupil_bottom_r', 'pupil_left_r', 'nose_tip', 'tube_top', 'tube_bottom', 'tongue_end_r', 'tongue_end_l', 'paw_r', 'paw_l'`

<img src="https://github.com/int-brain-lab/iblvideo/blob/master/DLC_IBL.png" width="50%" height="50%">

In addition, we track the tail end in the body videos:

<img src="https://github.com/int-brain-lab/iblvideo/blob/master/Screenshot%20from%202020-11-13%2011-00-15.png" width="50%" height="50%">

## Accessing results

DLC results are stored on the Flatrion server, with the `dataset_type` being `camera.dlc` and can be searched as any other IBL datatype via ONE. See https://int-brain-lab.github.io/iblenv/ for details. There is further a script to produce labelled videos as seen in the images above for the inspection of particular trials: https://github.com/int-brain-lab/iblapps/blob/develop/dlc/DLC_labeled_video.py

## Installing DLC locally on an IBL server

Official installation instructions for deeplabcut: https://github.com/AlexEMG/DeepLabCut/blob/master/docs/installation.md

### Pre-requisites

Install spike sorting server as per https://docs.google.com/document/d/1NYVlVD8OkwRYUaPeHo3ZFPuwpv_E5zgUVjLsV0V5Ko4

### Installation of cuda

Install CUDA libraries matching the TensorFlow version (10.0 for DeepLabCut 2.0 as of January 2020) https://docs.google.com/document/d/1UyXUOx21mwrpBtCcS51avnikmyCPCzXEtTRaTetH-Mo

As mentioned in the CUDA install guide, the installation procedure may differ a bit given the fact that we need to support several CUDA toolkit versions.
In order to install DLC properly we will have to:

Install an extension of the Cuda Toolkit for deep neural networks: cuDNN
Install Tensorflow, a highly optimized and highly popular framework for deep neural nets running on GPU
Install DLC itself

Compatibility: DLC depends on a specific version of Tensorflow that itself depends on a specific version of CUDA toolkit and cuDNN. In January 2020, the prescribed setup was:
Cuda 10.0
Tensorflow 1.13.1
cuDNN for Cuda 10.0
Cuda cuDNN
Get the cuDNN version here:
https://drive.google.com/open?id=1Ofad1VvHR5s4m-F30jjBUn_9zWjTNwd8

Then unpack the archive and copy the libraries with appropriate permissions:

`
wget --user iblmember -- password check_your_one_settings http://ibl.flatironinstitute.org/resources/cudnn-10.0-linux-x64-v7.6.5.32.tgz
tar -xvf cudnn-10.0-linux-x64-v7.6.5.32.tgz
sudo cp cuda/include/cudnn.h /usr/local/cuda-10.0/include
sudo cp cuda/lib64/libcudnn* /usr/local/cuda-10.0/lib64
sudo chmod a+r /usr/local/cuda-10.0/include/cudnn.h /usr/local/cuda-10.0/lib64/libcudnn*`

### Install the Python environment with Tensor Flow and Deep Lab Cut 2.1.5.2

Install a few things system wide

`
sudo apt-get install tk-dev
`

Install python3.7

`
sudo apt update
sudo apt install software-properties-common
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt install python3.7 python3.7-dev
python3.7 --version
`

Create and activate a Python 3.7 environment 

`
cd ~/Documents/PYTHON/envs
virtualenv dlc --python=python3.7
source ./dlc/bin/activate
`

Install packages (please observe order of those commands!)	

`
pip install ibllib
pip install https://extras.wxpython.org/wxPython4/extras/linux/gtk3/ubuntu-18.04/wxPython-4.0.7-cp37-cp37m-linux_x86_64.whl
pip install tensorflow-gpu==1.13.1
pip install deeplabcut
`

### Test if installation was successful 

Before starting a DLC program or console, make sure you link the proper CUDA version through the LD_LIBRARY_PATH environment variable.
Then the simplest test to check if it works properly is to import deeplabcut as shown above

`
export LD_LIBRARY_PATH=/usr/local/cuda-10.0/lib64:/usr/local/cuda-10.0/extras/CUPTI/lib64:/lib/nccl/cuda-10:$LD_LIBRARY_PATH

source ~/Documents/PYTHON/envs/dlc/bin/activate
`

`
ipython
import tensorflow as tf
import deeplabcut
`




