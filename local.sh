#!/usr/bin/env sh
## build
docker build -t mrmizz/floras:latest .
## copy artifacts from image to local
docker create -ti --name dummy mrmizz/floras:latest
docker cp dummy:/root/data/in data/
docker cp dummy:/root/data/out data/
docker rm -f dummy
