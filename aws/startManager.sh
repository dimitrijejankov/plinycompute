#!/bin/bash

# Like all the scripts in the 'aws' directory, this script assumes that the current working
# directory is the top-level plinycompute directory.

./aws/mountAndBuild.sh $1
cd ../nvme # This way, the manager's data will be stored on the mounted NVME drive
../plinycompute/bin/pdb-node -m
