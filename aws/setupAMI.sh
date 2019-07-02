#!/bin/bash

sudo apt install -y cmake

git clone https://github.com/dimitrijejankov/plinycompute.git
cd plinycompute

# Python3 comes with Ubuntu Bionic: https://askubuntu.com/a/865569
python3 scripts/internal/setupDependencies.py # 40-ish seconds

# Install google benchmark. The below installation instructions
# were found here: https://github.com/google/benchmark
cd ..
git clone https://github.com/google/benchmark.git
git clone https://github.com/google/googletest.git benchmark/googletest
mkdir build && cd build
cmake ../benchmark # <10 seconds
make # 50-ish seconds
sudo make install
cd ../plinycompute

