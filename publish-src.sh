#!/usr/bin/env zsh

aws s3 sync ./ml s3://big-time-floras/ --profile tap-in

