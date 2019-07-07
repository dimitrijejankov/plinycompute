#!/bin/bash

# Like all the scripts in the 'aws' directory, this script assumes that the current working
# directory is the top-level plinycompute directory.

# This script is NOT MEANT TO BE DIRECTLY USED; it is instead used in aws/startManager.sh
# and aws/startWorker.sh.

# This is a generic setup script for use with AWS instances that have an NVMe drive.
# It mounts the drive and builds the main PDB target. When using this script,
# you should also build any PDB applications/tests that you want to run, and then
# change the working directory to the mounted 'nvme' directory. THIS IS IMPORTANT. 
# The data stored in the cluster will be stored, by default, in whatever the current working 
# directory is when you run pdb-node. 



# Mount the SSD to directory 'nvme'
echo "lsblk:"
lsblk
# NOTE: lsblk is only run so that you can double check that everything is working correctly.
# Make sure that the row with name "nvme1n1" has the largest size; for the r5d.xlarge it should be
# around 140 GB.

cd ..
mkdir nvme
sudo mkfs.ext3 /dev/$1 # about 20-30 seconds. Note: shouldn't be asking for permission, if it does then nvme1n1 is the wrong argument
sudo mount /dev/$1 nvme
cd nvme
sudo chmod 777 -R .

# Build PDB
cd ../plinycompute
git pull
cmake .
make pdb-node -j
