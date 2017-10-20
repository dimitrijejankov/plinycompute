#  Copyright 2018 Rice University                                           
#                                                                           
#  Licensed under the Apache License, Version 2.0 (the "License");          
#  you may not use this file except in compliance with the License.         
#  You may obtain a copy of the License at                                  
#                                                                           
#      http://www.apache.org/licenses/LICENSE-2.0                           
#                                                                           
#  Unless required by applicable law or agreed to in writing, software      
#  distributed under the License is distributed on an "AS IS" BASIS,        
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. 
#  See the License for the specific language governing permissions and      
#  limitations under the License.                                           
#  ======================================================================== 
#!python
#JiaNote: to run integrated tests on standalone node for storing and querying data


import subprocess
import time
class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

#Parameters to tune for performance
threadNum = "1"
sharedMemorySize = "2048"

def startPseudoCluster():
    try:
        #run bin/pdb-cluster
        print (bcolors.OKBLUE + "start a pdbServer as the coordinator" + bcolors.ENDC)
        serverProcess = subprocess.Popen(['bin/pdb-cluster', 'localhost', '8108', 'Y'])
        print (bcolors.OKBLUE + "waiting for 9 seconds for server to be fully started..." + bcolors.ENDC)
        time.sleep(9)
        #run bin/pdb-server for worker
        num = 0;
        with open('conf/serverlist.test') as f:
            for each_line in f:
                print (bcolors.OKBLUE + "start a pdbServer at " + each_line + "as " + str(num) + "-th worker" + bcolors.ENDC)
                num = num + 1
                serverProcess = subprocess.Popen(['bin/pdb-server', threadNum, sharedMemorySize, 'localhost:8108', each_line])
                print (bcolors.OKBLUE + "waiting for 9 seconds for server to be fully started..." + bcolors.ENDC)
                time.sleep(9)
                each_line = each_line.split(':')
                port = int(each_line[1])
                subprocess.check_call(['bin/CatalogTests',  '--port', '8108', '--serverAddress', 'localhost', '--command', 'register-node', '--node-ip', 'localhost', '--node-port', str(port), '--node-name', 'worker', '--node-type', 'worker'])


    except subprocess.CalledProcessError as e:
        print (bcolors.FAIL + "[ERROR] in starting peudo cluster" + bcolors.ENDC)
        print (e.returncode)


print("#################################")
print("REQUIRE 8192 MB MEMORY TO RUN scons test")
print("#################################")



print("#################################")
print("CLEAN UP THE TESTING ENVIRONMENT")
print("#################################")
subprocess.call(['bash', './scripts/cleanupNode.sh'])
numTotal = 4
numErrors = 0
numPassed = 0

subprocess.call(['bash', './scripts/cleanupNode.sh'])
print (bcolors.OKBLUE + "waiting for 5 seconds for server to be fully cleaned up...")
time.sleep(5)


print("#################################")
print("RUN AGGREGATION AND SELECTION MIXED TEST ON G-2 PIPELINE")
print("#################################")

try:
    #start pseudo cluster
    startPseudoCluster()

    #run bin/test74
    print (bcolors.OKBLUE + "start a query client to store and query data from pdb cluster" + bcolors.ENDC)
    subprocess.check_call(['bin/test74', 'Y', 'Y', '1024', 'localhost', 'Y'])

except subprocess.CalledProcessError as e:
    print (bcolors.FAIL + "[ERROR] in running distributed aggregation and selection mixed test" + bcolors.ENDC)
    print (e.returncode)
    numErrors = numErrors + 1

else:
    print (bcolors.OKBLUE + "[PASSED] G2-distributed aggregation and selection mixed test" + bcolors.ENDC)
    numPassed = numPassed + 1



subprocess.call(['bash', './scripts/cleanupNode.sh'])

print (bcolors.OKBLUE + "waiting for 5 seconds for server to be fully cleaned up...")
time.sleep(5)

print("#################################")
print("RUN JOIN TEST ON G-2 PIPELINE")
print("#################################")

try:
    #start pseudo cluster
    startPseudoCluster()

    #run bin/test79
    print (bcolors.OKBLUE + "start a query client to store and query data from pdb cluster" + bcolors.ENDC)
    subprocess.check_call(['bin/test79', 'Y', 'Y', '1024', 'localhost', 'Y'])

except subprocess.CalledProcessError as e:
    print (bcolors.FAIL + "[ERROR] in running distributed join test" + bcolors.ENDC)
    print (e.returncode)
    numErrors = numErrors + 1

else:
    print (bcolors.OKBLUE + "[PASSED] G2-distributed join test" + bcolors.ENDC)
    numPassed = numPassed + 1



subprocess.call(['bash', './scripts/cleanupNode.sh'])

print (bcolors.OKBLUE + "waiting for 5 seconds for server to be fully cleaned up...")
time.sleep(5)

print("#################################")
print("RUN SELECTION AND JOIN MIXED TEST ON G-2 PIPELINE")
print("#################################")

try:
    #start pseudo cluster
    startPseudoCluster()

    #run bin/test78
    print (bcolors.OKBLUE + "start a query client to store and query data from pdb cluster" + bcolors.ENDC)
    subprocess.check_call(['bin/test78', 'Y', 'Y', '1024', 'localhost', 'Y'])

except subprocess.CalledProcessError as e:
    print (bcolors.FAIL + "[ERROR] in running distributed selection and join mixed test" + bcolors.ENDC)
    print (e.returncode)
    numErrors = numErrors + 1

else:
    print (bcolors.OKBLUE + "[PASSED] G2-distributed selection and join mixed test" + bcolors.ENDC)
    numPassed = numPassed + 1



subprocess.call(['bash', './scripts/cleanupNode.sh'])

print (bcolors.OKBLUE + "waiting for 5 seconds for server to be fully cleaned up...")
time.sleep(5)

print("#################################")
print("RUN COMPLEX AGGREGATION TEST ON G-2 PIPELINE")
print("#################################")

try:
    #start pseudo cluster
    startPseudoCluster()

    #run bin/test90
    print (bcolors.OKBLUE + "start a query client to store and query data from pdb cluster" + bcolors.ENDC)
    subprocess.check_call(['bin/test90', 'Y', 'Y', '1024', 'localhost', 'Y'])

except subprocess.CalledProcessError as e:
    print (bcolors.FAIL + "[ERROR] in running distributed selection and join mixed test" + bcolors.ENDC)
    print (e.returncode)
    numErrors = numErrors + 1

else:
    print (bcolors.OKBLUE + "[PASSED] G2-distributed selection and join mixed test" + bcolors.ENDC)
    numPassed = numPassed + 1



print("#################################")
print("SUMMRY")
print("#################################")
print (bcolors.OKGREEN + "TOTAL TESTS: " + str(numTotal) + bcolors.ENDC)
print (bcolors.OKGREEN + "PASSED TESTS: " + str(numPassed) + bcolors.ENDC)
print (bcolors.FAIL + "FAILED TESTS: " + str(numErrors) + bcolors.ENDC)
                                              
