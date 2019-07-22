# AWS Setup
This document exists to show what I have done to set up the AWS environment for performance benchmarking. It was written by Vicram Rajagopalan.

The layout of this document is as follows. If all you want to do is launch a cluster using the configurations I've already set up, you only need to read the **Running the Cluster** section. The rest of the document goes over how and why I set everything up that way, with sufficient detail that you would be able to replicate the process yourself.

## Running the Cluster
Note: this uses On-Demand Instances. If you want to use Spot Instances, you'll have to configure it yourself.

1. Log in to AWS, change the region to "Ohio", and go to the EC2 dashboard.
2. Click "Launch Instance".
3. Click the "My AMIs" tab and select "PDB Benchmark Image".
4. Select an instance type which includes an NVMe SSD drive. I recommend using "r5d.xlarge"; that's the instance type I have used for my benchmarks.
5. Change "Number of instances" to be the number of worker nodes you want, plus 1 (for the manager). I recommend 5 (4 workers and a manager).
6. Under "Network", select the VPC named "vpc-0021b44045f7ffd68 | benchmarking".
7. Under "Subnet", select the subnet named "Public subnet".
8. Set "Auto-assign Public IP" to "Enable".
9. At the bottom right-hand corner of the screen, click "Next: Add Storage" and then "Next: Add Tags" and "Next: Configure Security Group".
10. You should be on the screen that says "Step 6: Configure Security Group". Choose "Select an existing security group" and then choose the security group named "default".
11. Click "Review and Launch" then "Launch".
12. Go to the EC2 dashboard and you should see the instances starting up. I recommend naming them "manager", "worker1", "worker2", etc. This will make subsequent steps easier.
13. For each instance, note down the Public DNS (IPv4) and the Private IP fields.
14. Connect to the instances via SSH, using the Public DNS. 
15. Run the following commands on each instance. I recommend waiting until the 2 status checks have finished before doing this, although I'm not actually sure whether it would make a difference.
```
cd plinycompute
# Optionally, check out a branch of your choosing
git pull
lsblk
```
16. These next parts are pretty cumbersome. I apologize in advance. The output of `lsblk` should look something like this:
```
NAME        MAJ:MIN RM   SIZE RO TYPE MOUNTPOINT
loop0         7:0    0    18M  1 loop /snap/amazon-ssm-agent/1335
loop1         7:1    0  88.4M  1 loop /snap/core/7169
loop2         7:2    0  88.5M  1 loop /snap/core/7270
loop3         7:3    0    18M  1 loop /snap/amazon-ssm-agent/1455
nvme0n1     259:0    0 139.7G  0 disk
nvme1n1     259:1    0    16G  0 disk
└─nvme1n1p1 259:2    0    16G  0 part /
```
Note that the above numbers correspond to an r5d.xlarge instance. You need to find the name of the NVMe drive. This is the name of the drive with the largest size. In my experience, the name is usually `nvme0n1` but it's sometimes `nvme1n1`. Either way, note this down for each instance. There does not appear to be a way to predict which it will be, and it can vary from instance to instance.

17. On the manager node, run `./aws/startManager.sh diskName`, where `diskName` is either `nvme0n1` or `nvme1n1`, whichever you noted down for the manager.
18. Allow the manager to start up. You know that the manager has finished starting when you see it output these lines:
```
Starting a new storage!
Waiting for the server to start accepting requests.
Waiting for the server to start accepting requests.
Distributed storage manager server started!
Distributed storage manager server started!
```
19. On each worker node, run `./aws/startWorker.sh diskName workerPrivateIP managerPrivateIP`, where `diskName` is the noted NVMe drive name for that worker, `workerPrivateIP` is the private IP for that worker (from step 13), and `managerPrivateIP` is the private IP for the manager instance. NOTE: I have hard-coded the number of threads for each worker to correspond to the number of cores in the r5d.xlarge instance type (4). See **Instance Type**, below, for more.
20. Wait until you see the printouts from step 18 for each worker. Congrats! You have successfully started a distributed Plinycompute cluster.

To remember all the commands to run and all the info for each instance, I open up the following template in a text editor and edit the parts that change. This is what the entries looked like for one of my runs:
```
PRIVATE IPs
manager IP: 10.0.0.82
worker1: 10.0.0.196
worker2: 10.0.0.171
worker3: 10.0.0.92
worker4: 10.0.0.65

cd plinycompute
# If necessary, check out a branch
git pull
lsblk
./aws/startManager.sh nvme0n1

# Or, for a worker:
# ./aws/startWorker.sh nvme0n1 workerPrivateIP managerPrivateIP

./aws/startWorker.sh nvme0n1 10.0.0.196 10.0.0.82
./aws/startWorker.sh nvme0n1 10.0.0.171 10.0.0.82
./aws/startWorker.sh nvme1n1 10.0.0.92 10.0.0.82
./aws/startWorker.sh nvme0n1 10.0.0.65 10.0.0.82
```
Of course, this is optional, but I find it helpful.

## VPC
I set up a VPC using the steps [here](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/get-set-up-for-amazon-ec2.html#create-a-vpc). I first tried creating it in the N. Virginia region, but I got an error message saying "you've reached your limit of VPCs" or something like that, so I tried again with Ohio and it worked. **This means that you need to set your region to Ohio in order to use the VPC.**
* I named the VPC "benchmarking".
* The VPC ID is "vpc-0021b44045f7ffd68".
* The IPv4 CIDR block was automatically set to "10.0.0.0/16".

### Subnet
According to some of the documentation I've read, in order to use a VPC you may need to set up a subnet associated with that VPC. But if you go to the Subnets section of the VPC Dashboard (with region set to Ohio), you'll see that there's already a subnet associated with the VPC, called "Public subnet". So that should be sufficient. Its Availability Zone is "us-east-2c" which probably means that all instances will be launched in the same availability zone, which is good for benchmark reproducibility.

**NOTE** It would be convenient for the subnet have auto-assign public IP enabled by default, but this subnet has it disabled by default. You may want to look into changing this or creating a new subnet with this setting.

### Security Groups
I'm under the impression that security groups are pretty important for setting up a cluster, because they determine what ports the instances are allowed to communicate on (among other things). Fortunately, this is set up by default: according to the documentation [here](https://docs.aws.amazon.com/vpc/latest/userguide/VPC_SecurityGroups.html#DefaultSecurityGroup), the VPC comes with a default security group that should allow instances to communicate with each other on any ports *as long as they are all in that security group*. 

I also added the following inbound rule:
```
Type: SSH        Protocol: TCP        Port Range: 22        Source: 0.0.0.0/0
```
Without this rule, you wouldn't be able to SSH into the instances. For reference, I copied this rule from one of the "launch-wizard-xx" autogenerated security groups.

I have given the VPC's default security group the name "benchmarking-security-group".

## Launching an Instance
See the documentation [here](https://docs.aws.amazon.com/vpc/latest/userguide/working-with-vpcs.html#VPC_Launch_Instance).
1. Select the AMI/instance type as usual.
2. Under "Configure Instance Details", set "Network" to the "benchmarking-security-group" VPC and set Subnet to the "Public subnet" subnet. Also set "Auto-assign Public IP" to "Enable".
3. Under "Configure Security Group", set the security group to the one with name "default".

## Creating the AMI
It would be useful to have an AMI which has all of the necessary dependencies installed, so we don't need to start from scratch every time we want to benchmark. Here are all the requirements/constraints I can think of for the AMI:
* To minimize data storage costs, we need to ensure that the root EBS device for the AMI is as small as possible (we'll be using an ephemeral NVMe drive for the actual benchmarks; see section **Instance Type** below). But it needs to be big enough to store all the installed libraries we're using.
* Should have the Plinycompute repo cloned already. However, note that because the AMI will be static, you should probably always do a `git pull` to make sure you're getting the most up-to-date version of the repo.
* Also because the repo will likely be up-to-date, it's probably not worthwhile to build PDB before creating the AMI; users will likely have to rebuild everything.

Given these constraints, I've created an AMI. Here are the steps I took to do this:
1. Switched my region to Ohio and went to the EC2 Dashboard. (This is important because the AMI will only be available in the region you create it in, and you'll need to spend something like [2 cents per gigabyte](https://datapath.io/resources/blog/what-are-aws-data-transfer-costs-and-how-to-minimize-them/) to move it to a different region.)
2. Clicked "Launch Instance".
3. Selected the AMI: "Ubuntu Server 18.04 LTS (HVM), SSD Volume Type - ami-0c55b159cbfafe1f0 (64-bit x86)"
4. Selected the t3.large instance type (2 cores, 8 GiB memory). This was mostly arbitrary, because the AMI will outlive the instance. I chose an instance that a) is cheap but probably has enough resources to install things quickly enough, and b) is EBS-only to ensure everything will be installed to the EBS disk.
5. I configured the network settings as explained in steps 6-10 of **Running the Cluster**, above.
6. Under "Add Storage", I gave the root device 16 GiB to leave room for installed libraries and PDB binary files. The specific number 16 was fairly arbitrary.
7. Waited until the status checks completed before SSHing into the instance. (You can see this on the EC2 Dashboard.)
8. Ran the following commands:
```
git clone https://github.com/dimitrijejankov/plinycompute.git
cd plinycompute
sudo ./aws/setupAMI.sh
```
9. Went to the EC2 Dashboard and created an AMI from the instance, naming it "PDB Benchmark Image". See [here](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/creating-an-ami-ebs.html) for details.
10. Terminated the instance.

**NOTE:** When you're done with the AMI, you should deregister it. But this doesn't delete the EBS snapshot that was used to create the AMI, so you'll need to manually delete that or else AWS will keep charging you for it. See [here](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/deregister-ami.html#clean-up-ebs-ami) for more.

## Instance Type
My recommendation of r5d.xlarge is somewhat arbitrary. It's a mid-sized instance at a reasonable cost: 4 cores, 32 GiB memory, 150 GiB NVMe drive, and [$0.288 per Hour](https://aws.amazon.com/ec2/pricing/on-demand/). As mentioned above, I have hard-coded the `startWorker.sh` script to use 4 threads.

As a side note, because the r5d.xlarge uses Intel processors, you have to watch out when setting the number of threads to use, because hyperthreading means that each core can have 2 threads. I did a bit of Googling, and [this Stackexchange post](https://unix.stackexchange.com/questions/218074/how-to-know-number-of-cores-of-a-system-in-linux) has a good discussion of interpreting the output of `lscpu` given this context. The main takeaway is that the number of threads is the number of threads per core, times the number of cores per socket, times the number of sockets.

Given this information, let's examine the `lscpu` output for an r5d.xlarge:
```
ubuntu@ip-10-0-0-245:~$ lscpu
Architecture:        x86_64
CPU op-mode(s):      32-bit, 64-bit
Byte Order:          Little Endian
CPU(s):              4
On-line CPU(s) list: 0-3
Thread(s) per core:  2
Core(s) per socket:  2
Socket(s):           1
NUMA node(s):        1
Vendor ID:           GenuineIntel
CPU family:          6
Model:               85
Model name:          Intel(R) Xeon(R) Platinum 8175M CPU @ 2.50GHz
Stepping:            4
CPU MHz:             3200.768
BogoMIPS:            5000.00
Hypervisor vendor:   KVM
Virtualization type: full
L1d cache:           32K
L1i cache:           32K
L2 cache:            1024K
L3 cache:            33792K
NUMA node0 CPU(s):   0-3
Flags:               fpu vme de pse tsc msr pae mce cx8 apic sep mtrr pge mca cmov pat pse36 clflush mmx fxsr sse sse2 ss ht syscall nx pdpe1gb rdtscp lm constant_tsc rep_good nopl xtopology nonstop_tsc cpuid aperfmperf tsc_known_freq pni pclmulqdq ssse3 fma cx16 pcid sse4_1 sse4_2 x2apic movbe popcnt tsc_deadline_timer aes xsave avx f16c rdrand hypervisor lahf_lm abm 3dnowprefetch invpcid_single pti fsgsbase tsc_adjust bmi1 hle avx2 smep bmi2 erms invpcid rtm mpx avx512f avx512dq rdseed adx smap clflushopt clwb avx512cd avx512bw avx512vl xsaveopt xsavec xgetbv1 xsaves ida arat pku ospke
```
As you can see, Threads/Core * Cores/Socket * Sockets = 4, which is the number that we expected. This confirms that 4 is indeed the right number of threads to use per worker.
