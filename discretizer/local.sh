#!/usr/bin/env sh
## build
docker build -f discretizer/Dockerfile -t mrmizz/discretizer:latest .
## copy artifacts from image to local
docker create -ti --name dummy mrmizz/discretizer:latest
docker cp dummy:/tmp/data/mp4 data/
docker rm -f dummy
