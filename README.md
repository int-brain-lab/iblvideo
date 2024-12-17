# Lightning Pose (LP) applied to IBL data

You can find the README for pose tracking with DLC [here](README_DLC.md). 

## Video acquisition in IBL

Mice are filmed in training rigs and recording rigs. In training rigs there is only one side camera recording at full resolution (1280x1024) and 30 Hz. In the recording rigs, there are three cameras, one called 'left' at full resolution 1280x1024 and 60 Hz filming the mouse from one side, one called 'right' at half resolution (640x512) and 150 Hz filming the mouse symmetrically from the other side, and one called 'body' filming the trunk of the mouse from above.

Find more details in the [video white paper](https://figshare.com/articles/online_resource/Video_hardware_and_software_for_the_International_Brain_Laboratory/19694452).   

## Feature-tracking using LP

LP is used for markerless tracking of animal parts in these videos, returning for each frame x,y coordinates in px for each point and a likelihood (how certain was the network to have found that point in a specific frame). For each side video we track the following points: `'pupil_top_r'`, `'pupil_right_r'`, `'pupil_bottom_r'`, `'pupil_left_r'`, `'nose_tip'`, `'tongue_end_r'`, `'tongue_end_l'`, `'paw_r'`, `'paw_l'`

The following is an example frame from camera 'left'. Camera 'right' is flipped for labelling such that the frame looks the same as for the 'left' camera and the same LP network can be applied. Hence 'paw_r' for each camera is the paw that is closer to the camera, respectively. I.e. seen on the right side in the example frame. Analogously for other right (\_r) left (\_l) suffixes. The green rectangle indicates the whisker pad region, for which motion energy is computed. We further compute motion energy for the complete mouse body seen from the body camera.

<img src="https://github.com/int-brain-lab/iblvideo/blob/master/_static/side_view2.png" width="50%" height="50%">

In addition, we track the `'tail_start'` in the body videos:

<img src="https://github.com/int-brain-lab/iblvideo/blob/master/_static/body_view.png" width="50%" height="50%">

## Getting started

### Running LP for one mp4 video - stand-alone local run
```python
from one.api import ONE
from iblvideo import download_lit_models, lightning_pose

# Download the lightning pose models using ONE
one = ONE()
path_models = download_lit_models()

# Run lightning pose on a video
output = lightning_pose("Path/to/file.mp4", ckpts_path=path_models)
```

## Installing LP locally on an IBL server

### Pre-requisites

Install local server as per [this instruction](https://docs.google.com/document/d/1NYVlVD8OkwRYUaPeHo3ZFPuwpv_E5zgUVjLsV0V5Ko4).

Install CUDA 11.8 libraries as documented [here](https://docs.google.com/document/d/1UyXUOx21mwrpBtCcS51avnikmyCPCzXEtTRaTetH-Mo/edit#heading=h.39mk45fhbn1l). No need to set up the library paths yet, as we will do it below.

### Create a Python environment with Lightning Pose

Install python3.10 (required by `lightning-pose`)
```bash
sudo apt update -y 
sudo apt install software-properties-common -y  
sudo add-apt-repository ppa:deadsnakes/ppa -y
sudo apt update -y
sudo apt install python3.10 -y
sudo apt install python3.10-dev python3.10-distutils python3.10-venv -y
```

Create an environment called e.g. `litpose`
```bash
mkdir -p ~/Documents/PYTHON/envs
cd ~/Documents/PYTHON/envs
python3.10 -m venv litpose
```

Activate the environment and install packages
```bash
#CUDA_VERSION=11.8
#export PATH=/usr/local/cuda-$CUDA_VERSION/bin:$PATH
#export LD_LIBRARY_PATH=/usr/local/cuda-$CUDA_VERSION/lib64:/usr/local/cuda-$CUDA_VERSION/extras/CUPTI/lib64:$LD_LIBRARY_PATH  
source ~/Documents/PYTHON/envs/litpose/bin/activate

pip install ibllib
pip install lightning-pose==1.6.1
pip install ensemble-kalman-smoother==3.0.0
# TODO: change to master once it's merged:
pip install git+https://github.com/int-brain-lab/iblvideo.git@litpose
```

Test if your install was successful. Make sure that the environment is activated and that you have the correct CUDA version set.
```
python -c 'import iblvideo'
```

Once the import goes through without errors, you can set up an alias in your .bashrc file to easily enter the litpose environment:
```bash
nano ~/.bashrc
```
Enter this line under the other aliases:
```bash
alias litpose='export CUDA_VERSION=11.8; export PATH=/usr/local/cuda-"$CUDA_VERSION"/bin:$PATH; export LD_LIBRARY_PATH=/usr/local/cuda-"$CUDA_VERSION"/lib64:/usr/local/cuda-"$CUDA_VERSION"/extras/CUPTI/lib64:$LD_LIBRARY_PATH; source ~/Documents/PYTHON/envs/litpose/bin/activate'
```
After opening a new terminal you should be able to type `litpose` and end up in an environment in which you can import lightning-pose like above.


Eventually run the tests. You need to find the iblvideo installation path (e.g. using `python -c "import iblvideo; print(iblvideo.__file__)"`) and navigate into it. Then run:
```bash
pytest ./tests/test_pose_lp.py
```


## Releasing a new version (for devs)

We use semantic versioning, with a prefix: `iblvideo_MAJOR.MINOR.PATCH`. If you update the version, see below for what to adapt.

### Any version update
Update the version in
```
iblvideo/iblvideo/__init__.py
```
Afterwards, tag the new version on Github.


### Network model versioning
For lightning pose, we are no longer linking the versioning of the networks models with the code version 
(as was done for DLC). When the models are updated the test data should also be updated. Then upload both to the private 
and public S3 bucket in resources/lightning_pose with filename `networks_vX.Y.zip` and `lp_test_data_vX.Y.zip` respectively. 
Always keep the old models for reproducibility. Then update the default version number to the new version `vX.Y` in the 
functions `iblvideo.weights.download_lit_model` and `iblvideo.tests.download_test_data._download_lp_test_data`.

You should always also bump the code version in `iblvideo/__init__.py` when you update the models (at least the PATCH). 
This way, the code version that is stored in the alyx task can always be linked to a specific model version.