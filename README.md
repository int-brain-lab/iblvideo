# DeepLabCut (DLC) applied to IBL data
## Video acquisition in IBL

Mice are filmed in training rigs and recording rigs. In training rigs there is only one side camera recording at full resolution (1280x1024) and 30 Hz. In the recording rigs, there are three cameras, one called 'left' at full resolution 1280x1024 and 60 Hz filming the mouse from one side, one called 'right' at half resolution (640x512) and 150 Hz filming the mouse symmetrically from the other side, and one called 'body' filming the trunk of the mouse from above.

Find more details in the [video white paper](https://figshare.com/articles/online_resource/Video_hardware_and_software_for_the_International_Brain_Laboratory/19694452).   

## Feature-tracking using DLC

DLC is used for markerless tracking of animal parts in these videos, returning for each frame x,y coordinates in px for each point and a likelihood (how certain was the network to have found that point in a specific frame). For each side video we track the following points: `'pupil_top_r'`, `'pupil_right_r'`, `'pupil_bottom_r'`, `'pupil_left_r'`, `'nose_tip'`, `'tongue_end_r'`, `'tongue_end_l'`, `'paw_r'`, `'paw_l'`

The following is an example frame from camera 'left'. Camera 'right' is flipped for labelling such that the frame looks the same as for the 'left' camera and the same DLC network can be applied. Hence 'paw_r' for each camera is the paw that is closer to the camera, respectively. I.e. seen on the right side in the example frame. Analogously for other right (\_r) left (\_l) suffixes. The green rectangle indicates the whisker pad region, for which motion energy is computed. We further compute motion energy for the complete mouse body seen from the body camera.

<img src="https://github.com/int-brain-lab/iblvideo/blob/master/_static/side_view2.png" width="50%" height="50%">

In addition, we track the `'tail_start'` in the body videos:

<img src="https://github.com/int-brain-lab/iblvideo/blob/master/_static/body_view.png" width="50%" height="50%">

## Getting started
### Running DLC for one mp4 video - stand-alone local run
```python
from iblvideo import dlc
output = dlc("Path/to/file.mp4")
```

### Running DLC for one session using ONE
```python
from iblvideo import run_session
run_session("db156b70-8ef8-4479-a519-ba6f8c4a73ee")
```
### Running the queue using ONE
```python
from iblvideo import run_queue 
run_queue(machine='mymachine')
```
### Updating the environment
```bash
# Inside the main repository
chmod 775 update_env.sh
# If you installed your environment and repo in a different place than the example, 
# # you need to open and adapt this script
./update_env.sh
```

## Accessing results

DLC results are stored on the Flatrion server, with the `dataset_type` being `camera.dlc` and can be searched as any other IBL datatype via ONE. See https://int-brain-lab.github.io/iblenv/ for details. There is a script to produce labeled videos as seen in the images above for the inspection of particular trials (requires the legnthy download of full videos): https://github.com/int-brain-lab/iblapps/blob/develop/dlc/DLC_labeled_video.py and one to produce trial-averaged behavioral activity plots using DLC traces (fast, as this is downloading DLC traces and wheel data only): https://github.com/int-brain-lab/iblapps/blob/master/dlc/overview_plot_dlc.py 


Tensor flow dictates the versions of Python / cuDNN and CUDA while `deeplabcut` keeps up to date with [tensorflow](https://www.tensorflow.org/install/source#gpu) (as of May 2023)

| Version           | Python version | Compiler  | Build tools | cuDNN | CUDA |
|-------------------|----------------|-----------|-------------|------|------|
| tensorflow-2.12.0 | 3.8-3.11       | GCC 9.3.1 | Bazel 5.3.0 | 8.6  | 11.8 |
| tensorflow-2.11.0 | 3.7-3.10       | GCC 9.3.1 | Bazel 5.3.0 | 8.1  | 11.2 |

## Installing DLC locally on an IBL server - tensorflow 2.12.0

### Pre-requisites

Install local server as per [this instruction](https://docs.google.com/document/d/1NYVlVD8OkwRYUaPeHo3ZFPuwpv_E5zgUVjLsV0V5Ko4).

Install CUDA 11.8 libraries as documented [here](https://docs.google.com/document/d/1UyXUOx21mwrpBtCcS51avnikmyCPCzXEtTRaTetH-Mo/edit#heading=h.39mk45fhbn1l). No need to set up the library paths yet, as we will do it below.

Install cuDNN 8.6, an extension of the Cuda Toolkit for deep neural networks: Download cuDNN from FlatIron as shown below, or find it online.

```bash
# get the install archive
CUDA_VERSION=11.8
CUDNN_ARCHIVE=cudnn-linux-x86_64-8.9.1.23_cuda11-archive
wget --user iblmember --password check_your_one_settings http://ibl.flatironinstitute.org/resources/$CUDNN_ARCHIVE
# unpack the archive and copy libraries to the CUDA library path
tar -xvf $CUDNN_ARCHIVE.tar.xz
sudo cp $CUDNN_ARCHIVE/include/cudnn.h /usr/local/cuda-$CUDA_VERSION/include  
sudo cp $CUDNN_ARCHIVE/lib/libcudnn* /usr/local/cuda-$CUDA_VERSION/lib64  
sudo chmod a+r /usr/local/cuda-$CUDA_VERSION/include/cudnn.h /usr/local/cuda-$CUDA_VERSION/lib64/libcudnn*
```

### Create a Python environment with TensorFlow and DLC

Install python3.10 (NB: `torch`, `deeplabcut` and `python3.11` didn't play well for us, we tested 3.10)

```bash
sudo apt update -y 
sudo apt install software-properties-common -y  
sudo add-apt-repository ppa:deadsnakes/ppa -y
sudo apt-get install python3.10-tk -y  
sudo apt install python3.10 python3.10-dev -y 
sudo apt install python3.10-distutils -y
```

Create an environment called e.g. dlcenv

```bash
mkdir -p ~/Documents/PYTHON/envs
cd ~/Documents/PYTHON/envs
virtualenv dlcenv --python=python3.10
```

Activate the environment and install packages

```bash
source ~/Documents/PYTHON/envs/dlcenv/bin/activate
pip install setuptools==65
pip install ibllib
pip install torch==1.12
pip install deeplabcut[tf]
```

### Test if tensorflow and deeplabcut installation was successful

Export environment variables for testing
```bash
$CUDA_VERSION=11.8
export PATH=/usr/local/cuda-$CUDA_VERSION/bin:$PATH
export TF_FORCE_GPU_ALLOW_GROWTH='true'
export LD_LIBRARY_PATH=/usr/local/cuda-$CUDA_VERSION/lib64:/usr/local/cuda-$CUDA_VERSION/extras/CUPTI/lib64:$LD_LIBRARY_PATH  
```

Try to import deeplabcut and tensorflow (don't forget that dlcenv has to be active)
```
python -c 'import deeplabcut, tensorflow'
```

Once the import goes through without errors (it is ok to get the warning that you cannot use the GUI), you can set up an alias in your .bashrc file to easily enter the dlcenv environment:
```bash
nano ~/.bashrc
```
Enter this line under the other aliases:
```bash
alias dlcenv="CUDA_VERSION=11.8; export PATH=/usr/local/cuda-%CUDA_VERSION/bin:$PATH; export TF_FORCE_GPU_ALLOW_GROWTH='true'; export LD_LIBRARY_PATH=/usr/local/cuda-$CUDA_VERSION/lib64:/usr/local/cuda-$CUDA_VERSION/extras/CUPTI/lib64:$LD_LIBRARY_PATH; source ~/Documents/PYTHON/envs/dlcenv/bin/activate"
```
After opening a new terminal you should be able to type `dlcenv` and end up in an environment in which you can import tensorflow and deeplabcut like above.

### Clone and install iblvideo

Make sure to be in the Documents/PYTHON folder and that the dlcenv environment is activated
```bash
cd ~/Documents/PYTHON
dlcenv
```
Then clone and install iblvideo
```
git clone https://github.com/int-brain-lab/iblvideo.git
cd iblvideo
pip install -e .
```

Test if you install was successful
```
python -c 'import iblvideo'
```

Eventually run the tests:
```shell
pytest ./iblvideo/tests/test_choiceworld.py
pytest ./iblvideo/tests/test_motion_energy.py
```

## Releasing a new version (for devs)

We use semantic versioning, with a prefix: `iblvideo_MAJOR.MINOR.PATCH`. If you update the version, see below for what to adapt.

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
2. Make a new `dlc_test_data_v{MAJOR}.{MINOR}` directory.
3. Copy the three videos from the `input` folder of the previous version dlc_test_data to the new one.
4. Create the three parquet files to go in by running iblvideo.dlc() with the new weights folder as `path_dlc`, and each of the videos in the new `input` folder as `file_mp4`.
5. Rename the newly created folder `alf` inside the dlc_test_data folder into `output`.
6. Zip and upload the new weights and test data folders to FlatIron :
```
/resources/dlc/weights_v{MAJOR}.{MINOR}.zip
/resources/dlc/dlc_test_data_v{MAJOR}.{MINOR}.zip
```
6. Delete your local weights and test data and run tests/test_choiceworld.py to make sure everything worked.
