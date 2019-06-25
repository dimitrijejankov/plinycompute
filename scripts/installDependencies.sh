#!/usr/bin/env bash
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

# script that installs dependencies required by PlinyCompute
#
#Name	Homepage	Ubutnu Packages
#Snappy	https://github.com/google/snappy	libsnappy1v5, libsnappy-dev
#GSL	https://www.gnu.org/software/gsl/	libgsl-dev
#Eigen                                          libeigen3-dev
#Boost	http://www.boost.org/	libboost-dev, libboost-program-options-dev, libboost-filesystem-dev, libboost-system-dev
#Bison	https://www.gnu.org/software/bison/	bison
#Flex	https://github.com/westes/flex	flex
#

if [[ "$OSTYPE" == "darwin"* ]]; then

    brew install ossp-uuid
    brew install gsl
    brew install snappy
    brew install bison
    brew install flex
    brew install eigen
    
elif [[ "$OSTYPE" == "linux-gnu" ]]; then
	
    sudo apt-get -y install uuid-dev
    sudo apt-get -y install libeigen3-dev
    sudo apt-get -y install libgsl-dev
    sudo apt-get -y install libsnappy1v5 libsnappy-dev
    sudo apt-get -y install libboost-dev libboost-program-options-dev libboost-filesystem-dev libboost-system-dev
    sudo apt-get -y install bison flex
    sudo apt-get -y install clang
    sudo apt-get -y install libgtest-dev

    # todo:: remove this, unless you have to build it from scratch.
    # install google test
    cd /tmp
    cd googletest
    cmake .
    make
    sudo make install

    # todo:: remove this, unless you are actually using this.
    # install google benchmark
    cd /tmp
    cd benchmark
    cmake .
    make
    sudo make install

fi