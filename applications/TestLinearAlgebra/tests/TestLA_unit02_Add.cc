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
#ifndef TEST_LA_02_CC
#define TEST_LA_02_CC


// by Binhang, May 2017
// to test matrix add implemented by join;
#include <ctime>
#include <chrono>

#include "PDBDebug.h"
#include "PDBString.h"
#include "Lambda.h"
#include "PDBClient.h"
#include "LAScanMatrixBlockSet.h"
#include "LAWriteMatrixBlockSet.h"
#include "MatrixBlock.h"
#include "LAAddJoin.h"


using namespace pdb;
int main(int argc, char* argv[]) {
    bool printResult = true;
    std::cout << "Usage: #printResult[Y/N] #managerIp #addData[Y/N]" << std::endl;
    if (argc > 1) {
        if (strcmp(argv[1], "N") == 0) {
            printResult = false;
            std::cout << "You successfully disabled printing result." << std::endl;
        } else {
            printResult = true;
            std::cout << "Will print result." << std::endl;
        }
    } else {
        std::cout << "Will print result. If you don't want to print result, you can add N as the "
                     "first parameter to disable result printing." << std::endl;
    }

    std::string managerIp = "localhost";
    if (argc > 2) {
        managerIp = argv[2];
    }
    std::cout << "Manager IP Address is " << managerIp << std::endl;

    bool whetherToAddData = true;
    if (argc > 3) {
        if (strcmp(argv[3], "N") == 0) {
            whetherToAddData = false;
        }
    }
    PDBClient pdbClient(8108, managerIp);

    int blockSize = 64;  // Force it to be 64 by now.
    std::cout << "To add data with size: " << blockSize << "MB" << std::endl;

    if (whetherToAddData) {
        // Step 1. Create Database and Set
        // now, register a type for user data
        pdbClient.registerType("libraries/libMatrixBlock.so");

        // now, create a new database
        pdbClient.createDatabase("LA02_db");

        // now, create the first matrix set in that database
        pdbClient.createSet<MatrixBlock>("LA02_db", "LA_input_set1");

        // now, create the first matrix set in that database
        pdbClient.createSet<MatrixBlock>("LA02_db", "LA_input_set2");

        // Step 2. Add data
        int matrixRowNums = 4;
        int matrixColNums = 4;
        int blockRowNums = 10;
        int blockColNums = 5;

        int total = 0;

        // Add Matrix 1
        pdb::makeObjectAllocatorBlock(blockSize * 1024 * 1024, true);
        pdb::Handle<pdb::Vector<pdb::Handle<MatrixBlock>>> storeMatrix1 =
            pdb::makeObject<pdb::Vector<pdb::Handle<MatrixBlock>>>();

        for (int i = 0; i < matrixRowNums; i++) {
            for (int j = 0; j < matrixColNums; j++) {
                pdb::Handle<MatrixBlock> myData =
                    pdb::makeObject<MatrixBlock>(i, j, blockRowNums, blockColNums);
                // Foo initialization
                for (int ii = 0; ii < blockRowNums; ii++) {
                    for (int jj = 0; jj < blockColNums; jj++) {
                        (*(myData->getRawDataHandle()))[ii * blockColNums + jj] =
                            i + j + ii + jj + 1.0;
                    }
                }

                std::cout << "New block: " << total << std::endl;
                myData->print();
                storeMatrix1->push_back(myData);
                total++;
            }
        }

        pdbClient.sendData<MatrixBlock>("LA02_db", "LA_input_set1", storeMatrix1);
        PDB_COUT << total << " MatrixBlock data sent to dispatcher server~~" << std::endl;

        // Add Matrix 2
        total = 0;
        pdb::makeObjectAllocatorBlock(blockSize * 1024 * 1024, true);
        pdb::Handle<pdb::Vector<pdb::Handle<MatrixBlock>>> storeMatrix2 =
            pdb::makeObject<pdb::Vector<pdb::Handle<MatrixBlock>>>();

        for (int i = 0; i < matrixRowNums; i++) {
            for (int j = 0; j < matrixColNums; j++) {
                pdb::Handle<MatrixBlock> myData =
                    pdb::makeObject<MatrixBlock>(i, j, blockRowNums, blockColNums);
                // Foo initialization
                for (int ii = 0; ii < blockRowNums; ii++) {
                    for (int jj = 0; jj < blockColNums; jj++) {
                        (*(myData->getRawDataHandle()))[ii * blockColNums + jj] =
                            -1.0 * (i + j + ii + jj);
                    }
                }
                std::cout << "New block: " << total << std::endl;
                myData->print();
                storeMatrix2->push_back(myData);
                total++;
            }
        }

        pdbClient.sendData<MatrixBlock>("LA02_db", "LA_input_set2", storeMatrix2);
        PDB_COUT << total << " MatrixBlock data sent to dispatcher server~~" << std::endl;
    }
    // now, create a new set in that database to store output data

    PDB_COUT << "to create a new set for storing output data" << std::endl;
    pdbClient.createSet<MatrixBlock>("LA02_db", "LA_sum_set");

    // Step 3. To execute a Query
    // for allocations
    const UseTemporaryAllocationBlock tempBlock{1024 * 1024 * 128};

    // register this query class
    pdbClient.registerType("libraries/libLAAddJoin.so");
    pdbClient.registerType("libraries/libLAScanMatrixBlockSet.so");
    pdbClient.registerType("libraries/libLAWriteMatrixBlockSet.so");

    Handle<Computation> myMatrixSet1 = makeObject<LAScanMatrixBlockSet>("LA02_db", "LA_input_set1");
    Handle<Computation> myMatrixSet2 = makeObject<LAScanMatrixBlockSet>("LA02_db", "LA_input_set2");

    Handle<Computation> myAddJoin = makeObject<LAAddJoin>();
    myAddJoin->setInput(0, myMatrixSet1);
    myAddJoin->setInput(1, myMatrixSet2);

    Handle<Computation> mySumWriteSet = makeObject<LAWriteMatrixBlockSet>("LA02_db", "LA_sum_set");
    mySumWriteSet->setInput(myAddJoin);

    auto begin = std::chrono::high_resolution_clock::now();

    pdbClient.executeComputations({mySumWriteSet});
    std::cout << std::endl;

    auto end = std::chrono::high_resolution_clock::now();

    std::cout << std::endl;
    // print the results
    if (printResult) {
        std::cout << "to print result..." << std::endl;

        auto input1_iter = pdbClient.getSetIterator<MatrixBlock>("LA02_db", "LA_input_set1");
        std::cout << "Input Matrix 1 " << std::endl;
        int countIn1 = 0;
        while(input1_iter->hasNextRecord()){
            countIn1++;
            std::cout << countIn1 << ":";
            auto r = input1_iter->getNextRecord();
            r->print();
            std::cout << std::endl;
        }
        std::cout << "Matrix input1 block nums:" << countIn1 << std::endl;

        auto input2_iter = pdbClient.getSetIterator<MatrixBlock>("LA02_db", "LA_input_set2");
        std::cout << "Input Matrix 2 " << std::endl;
        int countIn2 = 0;
        while(input2_iter->hasNextRecord()){
            countIn2++;
            std::cout << countIn2 << ":";
            auto r = input2_iter->getNextRecord();
            r->print();
            std::cout << std::endl;
        }
        std::cout << "Matrix input2 block nums:" << countIn2 << std::endl;


        auto output_iter = pdbClient.getSetIterator<MatrixBlock>("LA02_db", "LA_sum_set");
        std::cout << "Sum query results: " << std::endl;
        int countOut = 0;
        while(output_iter->hasNextRecord()){
            countOut++;
            std::cout << countOut << ":";
            auto r = output_iter->getNextRecord();
            r->print();

            std::cout << std::endl;
        }
        std::cout << "Sum output count:" << countOut << "\n";
    }

    pdbClient.removeSet("LA02_db", "LA_sum_set");

    int code = system("scripts/cleanupSoFiles.sh force");
    if (code < 0) {
        std::cout << "Can't cleanup so files" << std::endl;
    }
    std::cout << "Time Duration: "
              << std::chrono::duration_cast<std::chrono::duration<float>>(end - begin).count()
              << " secs." << std::endl;
}

#endif
