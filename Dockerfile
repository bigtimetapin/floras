FROM python:3.7

## linux env
RUN apt-get update
RUN apt-get install vim -y
RUN apt-get install ffmpeg libsm6 libxext6 -y

## python depdencines
ARG TF_VERSION=2.5.0
WORKDIR ./root
COPY requirements.txt ./
RUN pip install -r requirements.txt
RUN pip install tensorflow==${TF_VERSION}

## src
COPY data ./data
COPY src ./src

ENTRYPOINT ~/../bin/bash