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
cd ..
mkdir nvme
# Create a single partition on the disk
sudo fdisk /dev/$1 # Interactive. Press: 'n', nothing, nothing, nothing, nothing, 'w'
# Reformat the partition with a new filesystem
sudo mkfs.ext3 /dev/$1p1 # about 20-30 seconds. Note: shouldn't be asking for permission, if it does then the input arg is wrong
# Mount the partition to directory nvme
sudo mount -t ext3 /dev/$1p1 nvme # This is how the mount(8) man page recommends calling mount
cd nvme
sudo chmod 777 -R .

# Build PDB
cd ../plinycompute
git pull
cmake .
make pdb-node -j
