# DeepLabCut (DLC) applied to IBL data
## Video acquisition in IBL

Mice are filmed in training rigs and recording rigs. In training rigs there is only one side camera recording at full resolution (1280x1024) and 30 Hz. In the recording rigs, there are three cameras, one called 'left' at full resolution 1280x1024 and 60 Hz filming the mouse from one side, one called 'right' at half resolution (640x512) and 150 Hz filming the mouse symmetrically from the other side, and one called 'body' filming the trunk of the mouse from above.    

## Feature-tracking using DLC 	 

DLC is used for markerless tracking of animal parts in these videos, returning for each frame x,y coordinates in px for each point and a likelihood (how certain was the network to have found that point in a specific frame). For each side video we track the following points:

```python
'pupil_top_r', 'pupil_right_r', 'pupil_bottom_r', 'pupil_left_r',
'nose_tip', 'tongue_end_r', 'tongue_end_l', 'paw_r', 'paw_l'
```

<img src="https://github.com/int-brain-lab/iblvideo/blob/master/_static/side_view.png" width="50%" height="50%">

In addition, we track the `'tail_start'` in the body videos:

<img src="https://github.com/int-brain-lab/iblvideo/blob/master/_static/body_view.png" width="50%" height="50%">

## Accessing results

DLC results are stored on the Flatrion server, with the `dataset_type` being `camera.dlc` and can be searched as any other IBL datatype via ONE. See https://int-brain-lab.github.io/iblenv/ for details. There is further a script to produce labelled videos as seen in the images above for the inspection of particular trials: https://github.com/int-brain-lab/iblapps/blob/develop/dlc/DLC_labeled_video.py

## Installing DLC locally on an IBL server

### Pre-requisites

Install local server as per [this instruction](https://docs.google.com/document/d/1NYVlVD8OkwRYUaPeHo3ZFPuwpv_E5zgUVjLsV0V5Ko4).

Install CUDA 10.0 libraries as documented [here](https://docs.google.com/document/d/1UyXUOx21mwrpBtCcS51avnikmyCPCzXEtTRaTetH-Mo) to match the TensorFlow version 1.13.1 required for DLC.

Install cuDNN, an extension of the Cuda Toolkit for deep neural networks: Download cuDNN from FlatIron as shown below, or find it online.

```bash
wget --user iblmember -- password check_your_one_settings http://ibl.flatironinstitute.org/resources/cudnn-10.0-linux-x64-v7.6.5.32.tgz  
tar -xvf cudnn-10.0-linux-x64-v7.6.5.32.tgz  
sudo cp cuda/include/cudnn.h /usr/local/cuda-10.0/include  
sudo cp cuda/lib64/libcudnn* /usr/local/cuda-10.0/lib64  
sudo chmod a+r /usr/local/cuda-10.0/include/cudnn.h /usr/local/cuda-10.0/lib64/libcudnn*  
```

Before starting a DLC program or console, make sure you point to the proper CUDA version through the LD_LIBRARY_PATH environment variable.

```bash
export LD_LIBRARY_PATH=/usr/local/cuda-10.0/lib64:/usr/local/cuda-10.0/extras/CUPTI/lib64:/lib/nccl/cuda-10:$LD_LIBRARY_PATH  
```

### Create a Python environment with TensorFlow and DLC

Install a few things system wide and then python3.7

```bash
sudo apt install tk-dev  
sudo apt update  
sudo apt install software-properties-common  
sudo add-apt-repository ppa:deadsnakes/ppa  
sudo apt install python3.7 python3.7-dev  
```

Create and activate a Python 3.7 environment called e.g. dlcenv

```bash
mkdir -p ~/Documents/PYTHON/envs
cd ~/Documents/PYTHON/envs
virtualenv dlcenv --python=python3.7
source ~/Documents/PYTHON/envs/dlcenv/bin/activate
```

Install packages (please observe order of those commands!)

```bash
pip install "dask[complete]"
pip install ibllib  
pip install https://extras.wxpython.org/wxPython4/extras/linux/gtk3/ubuntu-18.04/wxPython-4.0.7-cp37-cp37m-linux_x86_64.whl  
pip install tensorflow-gpu==1.13.1  
pip install deeplabcut  
```

### Test if installation was successful

The simplest test to check if it works properly is to import deeplabcut.

First, if you haven't yet done so, point to CUDA libraries and source the virtual environment
```bash
export LD_LIBRARY_PATH=/usr/local/cuda-10.0/lib64:/usr/local/cuda-10.0/extras/CUPTI/lib64:/lib/nccl/cuda-10:$LD_LIBRARY_PATH  
source ~/Documents/PYTHON/envs/dlcenv/bin/activate
```

Then start Python
```bash
ipython
```

And try importing tensorflow and deeplabcut
```python
import deeplabcut 
import tensorflow as tf  
```
