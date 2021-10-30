FROM ubuntu:latest

## linux env
RUN apt-get update
RUN apt-get install ffmpeg -y

COPY data/mov ./tmp/data/mov
COPY data/mp4 ./tmp/data/mp4
COPY discretizer/mov2mp4.sh ./tmp

ARG DIR=03
RUN sh ./tmp/mov2mp4.sh ${DIR}

ENTRYPOINT ~/../bin/bash
