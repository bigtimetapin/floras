FROM python:3.7

## linux env
RUN apt-get update
RUN apt-get install -y vim

## python depdencines
ARG TF_VERSION=2.5.0
WORKDIR ./root
COPY requirements.txt ./
RUN pip install -r requirements.txt
RUN pip install tensorflow==${TF_VERSION}

## src
COPY src ./src

ENTRYPOINT ~/../bin/bash