# AWS Setup
This document exists to show what I have done to set up the AWS environment that I use for performance benchmarking. 
## VPC
I set up a VPC using the steps [here](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/get-set-up-for-amazon-ec2.html#create-a-vpc). I first tried creating it in the N. Virginia region, but I got an error message saying "you've reached your limit of VPCs" or something like that, so I tried again with Ohio and it worked. **This means that you need to set your region to Ohio in order to use the VPC.**
* I named the VPC "benchmarking".
* The VPC ID is "vpc-0021b44045f7ffd68".
* The IPv4 CIDR block was automatically set to "10.0.0.0/16".

### Subnet
According to some of the documentation I've read, in order to use a VPC you may need to set up a subnet associated with that VPC. But if you go to the Subnets section of the VPC Dashboard (with region set to Ohio), you'll see that there's already a subnet associated with the VPC, called "Public subnet". So that should be sufficient. Its Availability Zone is "us-east-2c" which probably means that all instances will be launched in the same availability zone, which is good for benchmark reproducibility.

### Security Groups
I'm under the impression that security groups are pretty important for setting up a cluster, because they determine what ports the instances are allowed to communicate on (among other things). Fortunately, this is set up by default: according to the documentation [here](https://docs.aws.amazon.com/vpc/latest/userguide/VPC_SecurityGroups.html#DefaultSecurityGroup), the VPC comes with a default security group that should allow instances to communicate with each other on any ports *as long as they are all in that security group*. 

I have given the VPC's default security group the name "benchmarking-security-group".

## Launching an Instance
See the documentation [here](https://docs.aws.amazon.com/vpc/latest/userguide/working-with-vpcs.html#VPC_Launch_Instance). I won't repeat the steps here because they're spelled out in the link, but basically to launch an EC2 instance into the VPC, you select the AMI/instance type as usual, then under "Configure Instance Details" you select the desired VPC, subnet, and security group.

## Creating the AMI
It would be useful to have an AMI which has all of the necessary dependencies installed, so we don't need to start from scratch every time we want to benchmark. Here are all the requirements/constraints I can think of for the AMI:
* To minimize data storage costs, we need to ensure that the root EBS device for the AMI is as small as possible (we'll be using an ephemeral NVMe drive for the actual benchmarks; see section **Instance Type** below). But it needs to be big enough to store all the installed libraries we're using.
* Should have the Plinycompute repo cloned already. However, note that because the AMI will be static, you should probably always do a `git pull` to make sure you're getting the most up-to-date version of the repo.
* Also because the repo will likely be up-to-date, it's probably not worthwhile to build PDB before creating the AMI; users will likely have to rebuild everything.

## Instance Type
