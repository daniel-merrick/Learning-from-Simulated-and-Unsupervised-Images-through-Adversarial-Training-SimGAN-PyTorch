FROM nvidia/cuda:10.1-cudnn7-devel-ubuntu16.04 

ENV http_proxy http://internet.ford.com:83
ENV https_proxy http://internet.ford.com:83

RUN apt-get update && apt-get -y upgrade
RUN apt-get install -y --no-install-recommends \
	 python \
 	 python-pip \
     	 python-dev \
	 python-setuptools \
	 git-core \
	 vim

RUN pip install --upgrade pip
RUN pip install \
	numpy \
	matplotlib \
	torch \
	torchvision \
	pillow \
	scipy \
	opencv-python

RUN apt-get install -y libglib2.0-0
RUN apt-get install -y libsm6 libxext6 libxrender-dev
