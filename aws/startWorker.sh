#!/bin/bash

# Like all the scripts in the 'aws' directory, this script assumes that the current working
# directory is the top-level plinycompute directory.

# Usage:
# ./aws/startWorker.sh diskName workerPrivateIP managerPrivateIP
#
# Please see aws/README.md for details on how to find the instances' private IP addresses.

./aws/mountAndBuild.sh $1
cd ../nvme # This way, the worker's data will be stored on the mounted NVME drive
../plinycompute/bin/pdb-node -p 8109 -r ./pdbWorker1 -t 4 -i $2 -d $3
