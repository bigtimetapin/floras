#!/usr/bin/env zsh

#aws s3 cp ./data/in/xx/composite/a/images.tfrecords s3://big-time-floras/data/in/ --profile tap-in
#aws s3 sync ./data/png/01/x/ s3://big-time-floras/data/png/01/x/ --profile tap-in
aws s3 sync ./data/png/02/x/ s3://big-time-floras/data/png/02/x/ --profile tap-in
aws s3 sync ./data/png/03/x/ s3://big-time-floras/data/png/03/x/ --profile tap-in
aws s3 sync ./data/png/04/x/ s3://big-time-floras/data/png/04/x/ --profile tap-in
aws s3 sync ./data/png/05/x/ s3://big-time-floras/data/png/05/x/ --profile tap-in
aws s3 sync ./data/png/06/x/ s3://big-time-floras/data/png/06/x/ --profile tap-in
aws s3 sync ./data/png/07/x/ s3://big-time-floras/data/png/07/x/ --profile tap-in
aws s3 sync ./data/png/08/x/ s3://big-time-floras/data/png/08/x/ --profile tap-in
aws s3 sync ./data/png/10/x/ s3://big-time-floras/data/png/10/x/ --profile tap-in