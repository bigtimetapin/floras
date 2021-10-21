#!/usr/bin/env sh

DIR=$1
ffmpeg -i tmp/data/mov/${DIR}/in.mov -q:v 0 tmp/data/mp4/${DIR}/in.mp4
