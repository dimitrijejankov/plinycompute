#!/bin/bash

# NOTE: This script assumes that the current working directory is the top-level plinycompute directory.

sudo apt-get update
sudo apt install -y cmake

# Python3 comes with Ubuntu Bionic: https://askubuntu.com/a/865569
python3 scripts/internal/setupDependencies.py # 40-ish seconds

# Build google benchmark from source. The below installation instructions
# were found here: https://github.com/google/benchmark
cd ..
git clone https://github.com/google/benchmark.git
git clone https://github.com/google/googletest.git benchmark/googletest
mkdir build && cd build
cmake ../benchmark # <10 seconds
make # 50-ish seconds
sudo make install
cd ../plinycompute

