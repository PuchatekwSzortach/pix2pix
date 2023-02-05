# syntax = docker/dockerfile:experimental
FROM tensorflow/tensorflow:2.11.0-gpu

RUN apt install -y libcairo2-dev

# Install python environment
COPY ./requirements.txt /tmp/requirements.txt
RUN --mount=type=cache,mode=0777,target=/root/.cache/pip pip install -r /tmp/requirements.txt

# Setup bashrc
COPY ./docker/bashrc /root/.bashrc

# Setup PYTHONPATH
ENV PYTHONPATH=.

# Tensorflow keeps on using deprecated APIs ^^
ENV PYTHONWARNINGS="ignore::DeprecationWarning:tensorflow"

# # Set up working directory
WORKDIR /app
