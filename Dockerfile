FROM nvidia/cuda:11.1.1-cudnn8-runtime-ubuntu18.04

ENV LANG=C.UTF-8 LC_ALL=C.UTF-8
ENV PATH /opt/conda/bin:$PATH

RUN apt-get update
RUN apt-get install -y wget git
RUN apt-get install -y --no-install-recommends default-jre default-jdk
RUN apt-get install -y libopenmpi-dev freeglut3-dev
RUN DEBIAN_FRONTEND=noninteractive apt-get install -y keyboard-configuration
RUN apt-get install -y xorg openbox x11-apps
RUN apt-get install -y vim

RUN apt-get update --fix-missing && \
    apt-get install -y wget bzip2 curl git && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN wget \
    https://repo.anaconda.com/archive/Anaconda3-2020.11-Linux-x86_64.sh \
    && mkdir /root/.conda \
    && bash Anaconda3-2020.11-Linux-x86_64.sh -b \
    && rm -f Anaconda3-2020.11-Linux-x86_64.sh
ENV PATH="/root/anaconda3/bin:${PATH}"

#RUN echo "source activate" > ~/.bashrc
#RUN /bin/bash ~/.bashrc
RUN pip install --upgrade pip setuptools wheel
