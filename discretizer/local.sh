#!/usr/bin/env sh
## build
sudo docker build -f discretizer/Dockerfile -t mrmizz/discretizer:latest .
## copy artifacts from image to local
sudo docker create -ti --name dummy mrmizz/discretizer:latest
sudo docker cp dummy:/tmp/data/mp4 data/
sudo docker rm -f dummy
