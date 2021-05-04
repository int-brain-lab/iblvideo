# DOCKER INSTRUCTIONS
Here we provide a docker file to build an image and run in develop and production.
This is an advanced operation, the intended audience here are system admins or cloud computing managers wishing to run large queues on unix systems.

## Run Instructions
This assumes that docker is installed, the iblvideo repository is cloned on the system and that the `.one_params` file is in the home directory of the current user. See installation instructions below.

```shell
cd ~/Documents/PYTHON/iblvideo/docker/
git pull
docker-compose pull
```

Run the test container

``` shell
docker-compose run test
```

Run the queue using container

``` shell
docker-compose run queue
```

## IBL DLC Docker Installation Instructions
### Requirements
- Ubuntu 20.04 but any unix system with support for Docker will do
- nvidia driver installed
- a docker installation 
- support for nvidia runtime in docker
- ONE parameters setup to connect to the database [here](https://int-brain-lab.github.io/iblenv/one_docs/one_credentials.html) 

### Docker Installation instructions on Ubuntu 20.04, as of April 2021
-	Step 1: make sure Nvidia driver is installed by typing `nvidia-smi`
-	Step 2: install docker if not already installed, [instructions here](https://docs.docker.com/engine/install/ubuntu/)
-	Step 3: install nvidia support for Docker, [instructions here](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#docker)
-   Step 4: install docker compose, the version needs to be above 1.21. [instructions here](https://docs.docker.com/compose/install/)
-   Step 5: handle post installs steps
    ```shell
    sudo groupadd docker
    sudo usermod -aG docker $USER
    ```
    And then log out and log back in again
-   Step 5: setup the IBL DLC docker by cloning the repository and running the tests:

```shell
# clone the iblvideo repository
cd ~/Documents/PYTHON
git clone https://github.com/int-brain-lab/iblvideo.git
# copy one parameters from home to
cd iblvideo
cp ~/.one_params .one_params
# run the tests
cd docker
docker-compose run test
```

And then make sure that the `CACHE_DIR` parameter is set to `"/mnt/s0/Data/FlatIron"`

You're all setup, you can go to the run section of this document.

## Contributer info: update the Docker image
### Build release image instructions
This pre-supposes that you have already
-	a working dev environment
-	tagged and commit the version in the master branch
-	run the tests in local, with the weights and test data still present in the `$SDSC_PATH` below.

```shell
cd ~/Documents/PYTHON/iblvideo/docker/
VERSION=v1.0
SDSC_PATH=/mnt/s0/Data/FlatIron
WEIGHTS_DIR=$SDSC_PATH/resources/dlc/weights_$VERSION
TEST_DIR=$SDSC_PATH/integration/dlc/test_data 

cp -r $WEIGHTS_DIR .
cp -r $TEST_DIR/dlc_test_data_$VERSION .
cp -r $TEST_DIR/me_test_data .
docker build -t internationalbrainlab/dlc:$VERSION -f Dockerfile.$VERSION --no-cache .
```

Eventually push the image in dockerhub

```shell
VERSION=v1.0
docker login
docker push internationalbrainlab/dlc:$VERSION
```

```shell
docker run -it --rm --gpus all -u $(id -u):$(id -g) -v /mnt/s0/Data/FlatIron:/mnt/s0/Data/FlatIron -v ~/Documents/PYTHON/iblvideo/docker:/root internationalbrainlab/dlc:v1.0
python3
from iblvideo import run_queue
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
docker run -it --rm --gpus all -u $(id -u):$(id -g) -v /mnt/s0/Data/FlatIron:/mnt/s0/Data/FlatIron -v ~/Documents/PYTHON/iblvideo:/root internationalbrainlab/dlc:base
python3
from import deeplabcut
```

Eventually send the image to dockerhub

```shell
docker login
docker push internationalbrainlab/dlc:base
```
