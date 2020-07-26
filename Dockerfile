FROM nvidia/cuda

COPY . /pcl
WORKDIR /pcl

RUN apt-get update
RUN apt-get --assume-yes upgrade

RUN apt-get install python3
RUN apt-get install python3-pip libhdf5-dev
RUN pip install h5py torch torchvision

RUN python3 models/nndistance/build.py install
