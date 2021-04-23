# DOCKER INSTRUCTIONS
Here we provide a docker file to build an image and run in develop and production.
This is an advanced operation, the intended audience here are system admins or cloud computing managers wishing to run large queues on unix systems.

## Run Instructions
Copy (or create if not available) the one parameters from home to the repository folder

Run the test container
``` shell
docker-compose run test
```

Run the queue using container
``` shell
docker-compose run queue
```

## Installation Instructions
### Requirements
- Ubuntu 20.04 but any unix system with support for Docker will do
- nvidia driver installed
- a docker installation
- support for nvidia runtime in docker

### Docker Installation instructions on Ubuntu 20.04, as of April 2021
**TODO check and detail** (at next server install) 😬 Feel free to refer to online documentation
-	Step 1: make sure Nvidia driver is installed by typing `nvidia-smi`
-	Step 2: install docker if not already installed
-	Step 3: install nvidia support for Docker: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#docker

## Update the Docker image
### Build release image instructions
From the iblvideo repository
```shell
cd ./docker
VERSION=v1.0
WEIGHTS_DIR=/datadisk/FlatIron/resources/dlc/weights_$VERSION
TEST_DIR=/datadisk/FlatIron/integration/dlc/test_data 

cp ~/.one_params .one_params
cp -r $WEIGHTS_DIR ./
cp -r $TEST_DIR/dlc_test_data_$VERSION ./
cp -r $TEST_DIR/me_test_data ./
docker build -t internationalbrainlab/dlc:$VERSION -f Dockerfile.$VERSION
docker push internationalbrainlab/dlc:base
```

### Build base image instructions

```shell
cd ~/Documents/PYTHON
git clone https://github.com/int-brain-lab/iblvideo
cd iblvideo/docker
docker build -t internationalbrainlab/dlc:base -f Dockerfile.base  # this one will take a long time
```

Test the image by accessing a shell inside of the container:
```shell
docker run -it --rm --gpus all -u $(id -u):$(id -g) -v /mnt/s0/Data/FlatIron:/mnt/s0/Data/FlatIron -v ~/Documents/PYTHON/00_IBL/iblvideo:/root ibl/dlc:base
python3
import deeplabcut
```

Eventually send the image to dockerhub
```shell
docker login
docker push internationalbrainlab/dlc:base
```
