#!/bin/bash

# Like all the scripts in the 'aws' directory, this script assumes that the current working
# directory is the top-level plinycompute directory.

make BenchAggregationBalanced -j
./bin/BenchAggregationBalanced -v
