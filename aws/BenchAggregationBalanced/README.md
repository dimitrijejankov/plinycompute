# BenchAggregationBalanced
This file contains instructions for running BenchAggregationBalanced on a cluster.
1. Create the manager/worker instances and start them as documented in aws/README.md.
2. Start a new SSH session to the manager and do the following:
```
cd plinycompute
./aws/BenchAggregationBalanced/run.sh
```
3. Record the output table in timings.csv and commit it to version control.
4. Update any relevant areas in the Results section below.

Formatting for timings.csv:
Each column is labeled with two numbers. The first number is the number of departments, and the second number is the total number of employees in all departments. Each department has the same number of employees.
The rows are the time, in seconds, that it took to run the Aggregation. The same Aggregation is run num_rows times on the same instance of PDB, and the times are listed in order from top to bottom.
All whitespace should be ignored when parsing the table as a CSV (except newlines, of course).

## Results
Benchmark was run on 7/16/19. It was run on a cluster of 4 worker nodes, each of which had 4 cores, 32 GiB memory, and a 150 GiB NVMe SSD. Each Employee takes up approximately 154.35 bytes, although this is likely a slight overestimate. The largest data set consisted of 10 million Employees, which roughly comes out to 1.5 gigabytes of data distributed across the cluster. It may be useful to try benchmarking larger data sets, to get an idea of how the cluster performs when data cannot fit in memory. I tried doing this but for some reason the cluster was dying because one worker got an exception.

These benchmarks are not particularly useful on their own; the goal is to run the same workload using the old PDB so we can compare the performance. 
