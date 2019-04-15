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

#ifndef TEST_62_H
#define TEST_62_H

#include "PDBDebug.h"
#include "PDBString.h"
#include "Lambda.h"
#include "ScanEmployeeSet.h"
#include "SharedEmployee.h"
#include <ctime>
#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <chrono>
#include <fcntl.h>
#include <EmployeeBuiltInIdentitySelection.h>
#include <WriteBuiltinEmployeeSet.h>
#include <ComputePlan.h>

using namespace pdb;
int main(int argc, char *argv[]) {

  // create all of the computation objects
  // Handle<Computation> myScanSet = makeObject<ScanEmployeeSet>("chris_db", "chris_set");
  Handle<Computation> myScanSet = makeObject<ScanEmployeeSet>();
  Handle<Computation> myQuery = makeObject<EmployeeBuiltInIdentitySelection>();
  myQuery->setInput(myScanSet);
  Handle<Computation> myWriteSet = makeObject<WriteBuiltinEmployeeSet>("by8_db", "output_set");
  myWriteSet->setInput(myQuery);

  std::string inputTupleSetName = "";
  std::vector<std::string> inputColumnNames;
  std::vector<std::string> inputColumnsToApply;

  std::string outputTupleSetName;
  std::vector<std::string> outputColumnNames;
  std::string addedOutputColumnName;

  InputTupleSetSpecifier inputTupleSet1(inputTupleSetName, inputColumnNames, inputColumnsToApply);
  std::vector<InputTupleSetSpecifier> inputTupleSets1;
  inputTupleSets1.push_back(inputTupleSet1);

  std::string myScanSetTcapString = myScanSet->toTCAPString(inputTupleSets1,
                                                            0,
                                                            outputTupleSetName,
                                                            outputColumnNames,
                                                            addedOutputColumnName);

  inputTupleSetName = outputTupleSetName;
  inputColumnNames.clear();
  for (const auto &outputColumnName : outputColumnNames) {
    inputColumnNames.push_back(outputColumnName);
  }

  inputColumnsToApply.clear();
  inputColumnsToApply.push_back(addedOutputColumnName);
  outputTupleSetName = "";
  outputColumnNames.clear();
  addedOutputColumnName = "";

  PDB_COUT << "INPUT COLUMN NAMES:" << std::endl;
  for (const auto &inputColumnName : inputColumnNames) {
    PDB_COUT << inputColumnName << std::endl;
  }

  PDB_COUT << "INPUT COLUMNS TO APPLY:" << std::endl;
  for (const auto &i : inputColumnsToApply) {
    PDB_COUT << i << std::endl;
  }

  InputTupleSetSpecifier inputTupleSet2(inputTupleSetName, inputColumnNames, inputColumnsToApply);
  std::vector<InputTupleSetSpecifier> inputTupleSets2;
  inputTupleSets2.push_back(inputTupleSet2);

  std::string myQueryString = myQuery->toTCAPString(inputTupleSets2,
                                                    1,
                                                    outputTupleSetName,
                                                    outputColumnNames,
                                                    addedOutputColumnName);

  inputTupleSetName = outputTupleSetName;
  inputColumnNames.clear();

  for (const auto &outputColumnName : outputColumnNames) {
    inputColumnNames.push_back(outputColumnName);
  }

  inputColumnsToApply.clear();
  inputColumnsToApply.push_back(addedOutputColumnName);
  outputTupleSetName = "";
  outputColumnNames.clear();
  addedOutputColumnName = "";

  PDB_COUT << "INPUT COLUMN NAMES:" << std::endl;
  for (const auto &inputColumnName : inputColumnNames) {
    PDB_COUT << inputColumnName << std::endl;
  }

  PDB_COUT << "INPUT COLUMNS TO APPLY:" << std::endl;
  for (const auto &i : inputColumnsToApply) {
    PDB_COUT << i << std::endl;
  }

  InputTupleSetSpecifier inputTupleSet3(inputTupleSetName, inputColumnNames, inputColumnsToApply);
  std::vector<InputTupleSetSpecifier> inputTupleSets3;
  inputTupleSets3.push_back(inputTupleSet3);

  std::string myWriteSetTcapString = myWriteSet->toTCAPString(inputTupleSets3,
                                                              2,
                                                              outputTupleSetName,
                                                              outputColumnNames,
                                                              addedOutputColumnName);

  std::cout << myScanSetTcapString << std::endl;
  std::cout << myQueryString << std::endl;
  std::cout << myWriteSetTcapString << std::endl;

  int code = system("scripts/cleanupSoFiles.sh force");
  if (code < 0) {
    std::cout << "Can't cleanup so files" << std::endl;
  }
}

#endif