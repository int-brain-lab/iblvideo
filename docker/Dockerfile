#docker build -t ibl/dlc:base .
FROM nvidia/cuda:10.0-cudnn7-devel-ubuntu18.04

# link the cuda libraries
ENV LD_LIBRARY_PATH /usr/local/cuda/extras/CUPTI/lib64:$LD_LIBRARY_PATH

# setup time zone for tz
ENV TZ=Europe/Paris
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# Install python3.7
RUN apt-get update
RUN apt-get install -y software-properties-common
RUN add-apt-repository ppa:deadsnakes/ppa
RUN apt-get update
RUN apt-get install -y python3.7 python3.7-dev python3.7-tk python3-pip python3.7-venv git ffmpeg libgtk-3-dev

# Install Python dependencies
ARG PYTHON=python3.7
ENV LANG C.UTF-8
RUN ln -sf /usr/bin/${PYTHON} /usr/local/bin/python3
RUN python3 -m pip install --upgrade pip
RUN python3 -m pip install "dask[complete]"
RUN python3 -m pip install ibllib
RUN python3 -m pip install https://extras.wxpython.org/wxPython4/extras/linux/gtk3/ubuntu-18.04/wxPython-4.0.7-cp37-cp37m-linux_x86_64.whl
RUN python3 -m pip install tensorflow-gpu==1.13.1
RUN python3 -m pip install deeplabcut
