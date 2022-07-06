#!/usr/bin/env zsh

ffmpeg -framerate 30 -pattern_type glob -i './data/out/*.png' \
  -c:v libx264 -pix_fmt yuv420p out.mp4
