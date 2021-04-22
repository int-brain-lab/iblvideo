# DOCKER INSTRUCTIONS
Here we provide a docker file to build an image and run in develop and production.
This is an advanced operation, the intended audience here are system admins or cloud computing managers wishing to run large queues on unix systems.

## Run Instructions
Copy (or create if not available) the one parameters from home to the repository folder

	`cp ~/.one_params ./.one_params`

Run the test container

	`docker-compose run test`

Run the queue using container

	`docker-compose run queue`


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
-	Step 3: install nvidia support for Docker
-	Step 4: build or download the docker image

## Update the Docker image
### Build image instructions
From the repository directory cd to the docker dir

	`cd docker`

Build the image (this will download large files and take a while)

	`docker build . -t ibl/dlc:base`

To access the image for development

	`docker run -it --rm --gpus all -u $(id -u):$(id -g) -v /datadisk:/datadisk -v ~/Documents/PYTHON/00_IBL/iblvideo:/root ibl/dlc:base`
