FROM nvcr.io/nvidia/tensorflow:20.03-tf2-py3

## linux env
RUN apt-get update
RUN apt-get install vim -y
RUN apt-get install ffmpeg libsm6 libxext6 -y

## python depdencines
WORKDIR ./root
COPY requirements.txt ./
RUN pip install -r requirements.txt

## src
COPY data/in ./data/in
COPY data/out ./data/out
COPY ml/src ./src

# run
RUN python src/main/python/train.py

ENTRYPOINT ~/../bin/bash