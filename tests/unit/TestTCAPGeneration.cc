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
#include <QueryGraphAnalyzer.h>
#include <ReadInt.h>
#include <ReadStringIntPair.h>
#include <StringSelectionOfStringIntPair.h>
#include <IntSimpleJoin.h>
#include <WriteSumResult.h>
#include <IntAggregation.h>
#include <WriteBuiltinEmployeeSet.h>
#include <EmployeeBuiltInIdentitySelection.h>
#include <ScanEmployeeSet.h>
#include <IntUnion.h>
#include <IntWriter.h>

#include "../../../applications/TestMatrixMultiply/sharedLibraries/headers/MatrixBlock.h"
#include "../../../applications/TestMatrixMultiply/sharedLibraries/headers/MatrixScanner.h"
#include "../../../applications/TestMatrixMultiply/sharedLibraries/headers/MatrixMultiplyJoin.h"
#include "../../../applications/TestMatrixMultiply/sharedLibraries/headers/MatrixMultiplyAggregation.h"
#include "../../../applications/TestMatrixMultiply/sharedLibraries/headers/MatrixWriter.h"

using namespace pdb;
using namespace pdb::matrix;

//TEST(TestTcapGeneration, Test1) {
//
//  const pdb::UseTemporaryAllocationBlock tempBlock{1024 * 1024};
//
//  // create all of the computation objects
//  // make the scan set
//  Handle<Computation> myScanSet = makeObject<ScanSupervisorSet>();
//
//  // make the first filter
//  Handle<Computation> myFilter = makeObject<SillyQuery>();
//  myFilter->setInput(myScanSet);
//
//  // make the aggregation
//  Handle<Computation> myAgg = makeObject<SillyAgg>();
//  myAgg->setInput(myFilter);
//
//  // make the final filter
//  Handle<Computation> myFinalFilter1 = makeObject<FinalQuery>();
//  myFinalFilter1->setInput(myAgg);
//
//  // make the set writer
//  Handle<Computation> myWrite1 = makeObject<WriteSalaries>();
//  myWrite1->setInput(myFinalFilter1);
//
//  // the query graph has only the aggregation
//  std::vector<Handle<Computation>> queryGraph = { myWrite1 };
//
//  // create the graph analyzer
//  pdb::QueryGraphAnalyzer queryAnalyzer(queryGraph);
//
//  // parse the tcap string
//  std::string tcapString = queryAnalyzer.parseTCAPString();
//
//  std::cout << tcapString << std::endl;
//}
//
//TEST(TestTcapGeneration, Test2) {
//
//  const pdb::UseTemporaryAllocationBlock tempBlock{1024 * 1024};
//
//  // here is the list of computations
//  Handle <Computation> readA = makeObject <ReadInt>();
//  Handle <Computation> readB = makeObject <ReadStringIntPair>();
//  Handle <Computation> join = makeObject <SillyJoinIntString>();
//  join->setInput(0, readA);
//  join->setInput(1, readB);
//  Handle <Computation> write = makeObject <SillyWriteIntString>();
//  write->setInput(0, join);
//
//  // the query graph has only the aggregation
//  std::vector<Handle<Computation>> queryGraph = { write };
//
//  // create the graph analyzer
//  pdb::QueryGraphAnalyzer queryAnalyzer(queryGraph);
//
//  // parse the tcap string
//  std::string tcapString = queryAnalyzer.parseTCAPString();
//
//  std::cout << tcapString << std::endl;
//}
//
//TEST(TestTcapGeneration, Test3) {
//
//  const pdb::UseTemporaryAllocationBlock tempBlock{1024 * 1024};
//
//  Handle <Computation> readStringIntPair = makeObject <ReadStringIntPair>();
//  Handle <Computation> multiSelection = makeObject <StringIntPairMultiSelection>();
//  multiSelection->setInput(0, readStringIntPair);
//  Handle <Computation> writeStringIntPair = makeObject <WriteStringIntPair>();
//  writeStringIntPair->setInput(0, multiSelection);
//
//  // the query graph has only the aggregation
//  std::vector<Handle<Computation>> queryGraph = { writeStringIntPair };
//
//  // create the graph analyzer
//  pdb::QueryGraphAnalyzer queryAnalyzer(queryGraph);
//
//  // parse the tcap string
//  std::string tcapString = queryAnalyzer.parseTCAPString();
//
//  std::cout << tcapString << std::endl;
//}
//
//TEST(TestTcapGeneration, Test4) {
//
//  const pdb::UseTemporaryAllocationBlock tempBlock{1024 * 1024};
//
//  Handle<Computation> myScanSet1 = makeObject<ReadInt>("test78_db", "test78_set1");
//  Handle<Computation> myScanSet2 = makeObject<ReadStringIntPair>("test78_db", "test78_set2");
//  Handle<Computation> mySelection = makeObject<StringSelectionOfStringIntPair>();
//  mySelection->setInput(myScanSet2);
//  Handle<Computation> myJoin = makeObject<IntSimpleJoin>();
//  myJoin->setInput(0, myScanSet1);
//  myJoin->setInput(1, myScanSet2);
//  myJoin->setInput(2, mySelection);
//  Handle<Computation> myAggregation = makeObject<IntAggregation>();
//  myAggregation->setInput(myJoin);
//  Handle<Computation> myWriter = makeObject<WriteSumResult>("test78_db", "output_set1");
//  myWriter->setInput(myAggregation);
//
//  // the query graph has only the aggregation
//  std::vector<Handle<Computation>> queryGraph = { myWriter };
//
//  // create the graph analyzer
//  pdb::QueryGraphAnalyzer queryAnalyzer(queryGraph);
//
//  // parse the tcap string
//  std::string tcapString = queryAnalyzer.parseTCAPString();
//
//  std::cout << tcapString << std::endl;
//}
//

TEST(TestTcapGeneration, Test5) {

  const pdb::UseTemporaryAllocationBlock tempBlock{1024 * 1024};

  // create all of the computation objects
  Handle<Computation> myScanSet1 = makeObject<ReadInt>("chris_db", "input_set1");
  Handle<Computation> myScanSet2 = makeObject<ReadInt>("chris_db", "input_set2");
  Handle<Computation> myQuery = makeObject<IntUnion>();
  myQuery->setInput(0, myScanSet1);
  myQuery->setInput(1, myScanSet2);
  Handle<Computation> myWriteSet = makeObject<IntWriter>("chris_db", "outputSet");
  myWriteSet->setInput(myQuery);

  // the query graph has only the aggregation
  std::vector<Handle<Computation>> queryGraph = { myWriteSet };

  // create the graph analyzer
  pdb::QueryGraphAnalyzer queryAnalyzer(queryGraph);

  // here is the list of computations
  Handle<Vector<Handle<Computation>>> myComputations = makeObject<Vector<Handle<Computation>>>();

  // parse the tcap string
  std::cout << '\n';
  std::string tcapString = queryAnalyzer.parseTCAPString(*myComputations);

  std::cout << tcapString << std::endl;
}