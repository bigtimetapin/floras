#!/usr/bin/env sh

DIR=$1
echo ../data/mov/${DIR}/
ffmpeg -i ../data/mov/${DIR}/in.mov -q:v 0 ../data/mp4/${DIR}/in.mp4
