# 1) choose base container
ARG BASE_CONTAINER=ucsdets/datahub-base-notebook:2022.3-stable

FROM $BASE_CONTAINER

LABEL maintainer="Sirui Tao"


# 2) change to root to install packages
USER root

RUN apt update


# 3) install packages using notebook user
RUN conda install pytorch==1.12.1 -c pytorch
RUN conda install pyg -c pyg

RUN pip install --no-cache-dir \
    pyyaml~=6.0 \
    numpy==1.23.5


# 4) copy files to user home directory
WORKDIR /

COPY src src
COPY config config
COPY test test

COPY run.py run.py


# Override command to disable running jupyter notebook at launch
CMD ["/bin/bash"]