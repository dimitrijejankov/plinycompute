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
#include <SillyReadOfA.h>
#include <SillyReadOfB.h>
#include <SillyJoinIntString.h>
#include <SillyWriteIntString.h>
#include <QueryGraphAnalyzer.h>
#include <ScanSupervisorSet.h>
#include <SillyQuery.h>
#include <SillyAgg.h>
#include <WriteSalaries.h>
#include <FinalQuery.h>
#include <gtest/gtest.h>

using namespace pdb;

TEST(TestTcapGeneration, Test1) {

  const pdb::UseTemporaryAllocationBlock tempBlock{1024 * 1024};

  // create all of the computation objects
  // make the scan set
  Handle<Computation> myScanSet = makeObject<ScanSupervisorSet>();

  // make the first filter
  Handle<Computation> myFilter = makeObject<SillyQuery>();
  myFilter->setInput(myScanSet);

  // make the aggregation
  Handle<Computation> myAgg = makeObject<SillyAgg>();
  myAgg->setInput(myFilter);

  // make the final filter
  Handle<Computation> myFinalFilter1 = makeObject<FinalQuery>();
  myFinalFilter1->setInput(myAgg);

  // make the set writer
  Handle<Computation> myWrite1 = makeObject<WriteSalaries>();
  myWrite1->setInput(myFinalFilter1);

  // make the final filter
  Handle<Computation> myFinalFilter2 = makeObject<FinalQuery>();
  myFinalFilter2->setInput(myAgg);

  // make the set writer
  Handle<Computation> myWrite2 = makeObject<WriteSalaries>();
  myWrite2->setInput(myFinalFilter2);


  // the query graph has only the aggregation
  std::vector<Handle<Computation>> queryGraph = { myWrite1, myWrite2 };

  // create the graph analyzer
  pdb::QueryGraphAnalyzer queryAnalyzer(queryGraph);

  // parse the tcap string
  std::string tcapString = queryAnalyzer.parseTCAPString();

  std::cout << tcapString << std::endl;
}

TEST(TestTcapGeneration, Test2) {

  const pdb::UseTemporaryAllocationBlock tempBlock{1024 * 1024};

  // here is the list of computations
  Handle <Computation> readA = makeObject <SillyReadOfA>();
  Handle <Computation> readB = makeObject <SillyReadOfB>();
  Handle <Computation> join = makeObject <SillyJoinIntString>();
  join->setInput(0, readA);
  join->setInput(1, readB);
  Handle <Computation> write = makeObject <SillyWriteIntString>();
  write->setInput(0, join);

  // the query graph has only the aggregation
  std::vector<Handle<Computation>> queryGraph = { write };

  // create the graph analyzer
  pdb::QueryGraphAnalyzer queryAnalyzer(queryGraph);

  // parse the tcap string
  std::string tcapString = queryAnalyzer.parseTCAPString();

  std::cout << tcapString << std::endl;
}

#endif