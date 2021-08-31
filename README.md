# DeepLabCut (DLC) applied to IBL data
## Video acquisition in IBL

Mice are filmed in training rigs and recording rigs. In training rigs there is only one side camera recording at full resolution (1280x1024) and 30 Hz. In the recording rigs, there are three cameras, one called 'left' at full resolution 1280x1024 and 60 Hz filming the mouse from one side, one called 'right' at half resolution (640x512) and 150 Hz filming the mouse symmetrically from the other side, and one called 'body' filming the trunk of the mouse from above.    

## Feature-tracking using DLC

DLC is used for markerless tracking of animal parts in these videos, returning for each frame x,y coordinates in px for each point and a likelihood (how certain was the network to have found that point in a specific frame). For each side video we track the following points: `'pupil_top_r'`, `'pupil_right_r'`, `'pupil_bottom_r'`, `'pupil_left_r'`, `'nose_tip'`, `'tongue_end_r'`, `'tongue_end_l'`, `'paw_r'`, `'paw_l'`

<img src="https://github.com/int-brain-lab/iblvideo/blob/master/_static/side_view.png" width="50%" height="50%">

In addition, we track the `'tail_start'` in the body videos:

<img src="https://github.com/int-brain-lab/iblvideo/blob/master/_static/body_view.png" width="50%" height="50%">

## Getting started
### Running DLC for one mp4 video - stand-alone local run
```python
from iblvideo import dlc
output = dlc("Path/to/file.mp4")
```

### Running DLC for one session given its EID
```python
from iblvideo import run_session
run_session("db156b70-8ef8-4479-a519-ba6f8c4a73ee")
```
### Running 10 sessions from the queue
```python
from iblvideo import run_queue
run_queue(n_sessions=10)
```
## Accessing results

DLC results are stored on the Flatrion server, with the `dataset_type` being `camera.dlc` and can be searched as any other IBL datatype via ONE. See https://int-brain-lab.github.io/iblenv/ for details. There is a script to produce labeled videos as seen in the images above for the inspection of particular trials (requires the legnthy download of full videos): https://github.com/int-brain-lab/iblapps/blob/develop/dlc/DLC_labeled_video.py and one to produce trial-averaged behavioral activity plots using DLC traces (fast, as this is downloading DLC traces and wheel data only): https://github.com/int-brain-lab/iblapps/blob/master/dlc/overview_plot_dlc.py 

## Installing DLC locally on an IBL server

### Pre-requisites

Install local server as per [this instruction](https://docs.google.com/document/d/1NYVlVD8OkwRYUaPeHo3ZFPuwpv_E5zgUVjLsV0V5Ko4).

Install CUDA 10.0 libraries as documented [here](https://docs.google.com/document/d/1UyXUOx21mwrpBtCcS51avnikmyCPCzXEtTRaTetH-Mo) to match the TensorFlow version 1.13.1 required for DLC.

Install cuDNN, an extension of the Cuda Toolkit for deep neural networks: Download cuDNN from FlatIron as shown below, or find it online.

```bash
wget --user iblmember --password check_your_one_settings http://ibl.flatironinstitute.org/resources/cudnn-10.0-linux-x64-v7.6.5.32.tgz  
tar -xvf cudnn-10.0-linux-x64-v7.6.5.32.tgz  
sudo cp cuda/include/cudnn.h /usr/local/cuda-10.0/include  
sudo cp cuda/lib64/libcudnn* /usr/local/cuda-10.0/lib64  
sudo chmod a+r /usr/local/cuda-10.0/include/cudnn.h /usr/local/cuda-10.0/lib64/libcudnn*  
```

(optional): check CUDNN installation or for troubleshooting only)
Download and unzip https://ibl.flatironinstitute.org/resources/cudnn_samples_v7.zip
If necessary, setup your CUDA environment variables with the version you want to test

```
cd cudnn_samples_v7/mnistCUDNN/
make clean && make
./mnistCUDNN
```

Should print a message that finishes with Test passed !

### Create a Python environment with TensorFlow and DLC

Install a few things system wide and then python3.7

```bash
sudo apt update  
sudo apt install software-properties-common  
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt-get install python3.7-tk  
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
pip install -U setuptools
pip install git+https://github.com/int-brain-lab/ibllib.git
pip install https://extras.wxpython.org/wxPython4/extras/linux/gtk3/ubuntu-18.04/wxPython-4.0.7-cp37-cp37m-linux_x86_64.whl  
pip install tensorflow-gpu==1.13.1  
pip install deeplabcut==2.1.10  
```

### Test if tensorflow and deeplabcut installation was successful

Export environment variable to avoid tensorflow issue
```bash
export TF_FORCE_GPU_ALLOW_GROWTH='true'
```
 
Point to CUDA libraries and source the virtual environment
```bash
export LD_LIBRARY_PATH=/usr/local/cuda-10.0/lib64:/usr/local/cuda-10.0/extras/CUPTI/lib64:/lib/nccl/cuda-10:$LD_LIBRARY_PATH  
source ~/Documents/PYTHON/envs/dlcenv/bin/activate
```

Try importing tensorflow and deeplabcut
```
python -c 'import deeplabcut, tensorflow'
```

### Clone and install iblvideo
```
git clone https://github.com/int-brain-lab/iblvideo.git
cd iblvideo
pip install -e .
```

Test if you install was successful
```
python -c 'import iblvideo'
```
## Releasing a new version (for devs)

We use semantic versioning MAJOR.MINOR.PATCH. If you update the version, see below for what to adapt.

### Any version update
Update the version in
```
iblvideo/iblvideo/__init__.py
```
Afterwards, tag the new version on Github.


### Update MINOR or MAJOR
The version of DLC weights and DLC test data are synchronized with the MAJOR.MINOR version of this code. (Note that the patch version is not included in the directory names)

If you update any of the DLC weights, you also need to update the MINOR version of the code and the DLC test data, and vice versa.
1. For the weights, create a new directory called `weights_v{MAJOR}.{MINOR}` and copy the new weights, plus any unchanged weights into it.
2. Make a new `dlc_test_data_v{MAJOR}.{MINOR}` directory, with subdirectories `input` and `output`.
3. Copy the three videos from the `input` folder of the previous version dlc_test_data to the new one.
4. Create the three parquet files to go in `output` by running iblvideo.dlc() with the new weights folder as `path_dlc`, and each of the videos in the new `input` folder as `file_mp4`.
5. Zip and upload the new weights and test data folders to FlatIron :
```
/resources/dlc/weights_v{MAJOR}.{MINOR}.zip
/integration/dlc/test_data/dlc_test_data_v{MAJOR}.{MINOR}.zip
```
6. Delete your local weights and test data and run tests/test_choiceworld.py to make sure everything worked.
