/*****************************************************************************
 *                                                                           *
 *  Copyright 2018 Rice University                                           *
 *                                                                           *
 *  Licensed under the Apache License, Version 2.0 (the "License");          *
 *  you may not use this file except in compliance with the License.         *
 *  You may obtain a copy of the License at                                  *
 *                                                                           *
 *      http://www.apache.org/licenses/LICENSE-2.0                           *
 *                                                                           *
 *  Unless required by applicable law or agreed to in writing, software      *
 *  distributed under the License is distributed on an "AS IS" BASIS,        *
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. *
 *  See the License for the specific language governing permissions and      *
 *  limitations under the License.                                           *
 *                                                                           *
 *****************************************************************************/


#include <gtest/gtest.h>

#include "Handle.h"
#include "PDBVector.h"
#include "InterfaceFunctions.h"
#include "Employee.h"
#include "Supervisor.h"

//String class test

using namespace pdb;

TEST(StringTest, TestAll) {

    // just some random string
    std::string ipStd = "localhost";

    // init from cpp string
    String ip = ipStd;
    String ipConstChar = "localhost";

    // check if they are all equal
    EXPECT_TRUE(ip == ipStd);
    EXPECT_TRUE(ipConstChar == ipStd);

    // convert pdb string to regular string and append
    String nodeAddress = std::string(ip) + ":8108";

    // check if it is ok
    EXPECT_TRUE(nodeAddress == "localhost:8108");
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}