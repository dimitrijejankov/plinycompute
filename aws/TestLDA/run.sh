#!/bin/bash

# Like all the scripts in the 'aws' directory, this script assumes that the current working
# directory is the top-level plinycompute directory.

make TestLDA -j
./bin/TestLDA localhost 10 900 50 N N 50
