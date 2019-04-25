# DeepLabCut setup 1804

# Python and virtual environments
sudo apt update
sudo apt install python3-dev python3-pip python3-tk
sudo -EH pip install virtualenv
sudo -EH pip install virtualenvwrapper

# Install nvidia drivers
sudo apt install nvidia-384 nvidia-384-dev

# Install cuda 9.0
wget https://developer.nvidia.com/compute/cuda/9.0/Prod/local_installers/cuda_9.0.176_384.81_linux-run
chmod +x cuda_9.0.176_384.81_linux-run 	# Follow defaults
sudo ./cuda_9.0.176_384.81_linux-run --override

# Install cuDNN
# Download 7.4.2 cuDNN for cuda 9.0 (can't be automatically downloaded)
tar -xzvf cudnn-9.0-linux-x64-v7.4.2.24.tgz
sudo cp -P cuda/include/cudnn.h /usr/local/cuda-9.0/include          
sudo cp -P cuda/lib64/libcudnn* /usr/local/cuda-9.0/lib64/
sudo chmod a+r /usr/local/cuda-9.0/lib64/libcudnn*
sudo reboot

# run this
nvidia-smi
nvcc -V

# Create virtual environment
mkvirtualenv deeplabcut -p python3
pip install tensorflow-gpu==1.8
pip install deeplabcut
pip install ipykernel
python -m ipykernel install --user --name deeplabcut --display-name "deeplabcut"
pip install https://extras.wxpython.org/wxPython4/extras/linux/gtk3/ubuntu-16.04/wxPython-4.0.3-cp36-cp36m-linux_x86_64.whl

# Other fixes to make deeplabcut work
wget http://se.archive.ubuntu.com/ubuntu/pool/main/libp/libpng/libpng12-0_1.2.54-1ubuntu1_amd64.deb
sudo dpkg -i libpng12-0_1.2.54-1ubuntu1_amd64.deb
