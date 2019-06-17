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

#include <utility>
#include <gmock/gmock-generated-function-mockers.h>
#include <PDBBufferManagerImpl.h>
#include <gmock/gmock-more-actions.h>
#include <ReadStringIntPair.h>
#include <StringIntPairMultiSelection.h>
#include <WriteStringIntPair.h>
#include "Handle.h"
#include "Lambda.h"
#include "Supervisor.h"
#include "Employee.h"
#include "LambdaCreationFunctions.h"
#include "UseTemporaryAllocationBlock.h"
#include "pipeline/Pipeline.h"
#include "SetWriter.h"
#include "SelectionComp.h"
#include "AggregateComp.h"
#include "SetScanner.h"
#include "DepartmentTotal.h"
#include "VectorSink.h"
#include "MapTupleSetIterator.h"
#include "VectorTupleSetIterator.h"
#include "ComputePlan.h"
#include "ScanEmployeeSet.h"
#include "EmployeeBuiltInIdentitySelection.h"
#include "WriteBuiltinEmployeeSet.h"

// to run the aggregate, the system first passes each through the hash operation...
// then the system
using namespace pdb;

class MockPageSetReader : public pdb::PDBAbstractPageSet {
 public:

  MOCK_METHOD1(getNextPage, PDBPageHandle(size_t workerID));

  MOCK_METHOD0(getNewPage, PDBPageHandle());

  MOCK_METHOD0(getNumPages, size_t ());

  MOCK_METHOD0(resetPageSet, void ());
};

class MockPageSetWriter: public pdb::PDBAnonymousPageSet {
 public:

  MOCK_METHOD1(getNextPage, PDBPageHandle(size_t workerID));

  MOCK_METHOD0(getNewPage, PDBPageHandle());

  MOCK_METHOD1(removePage, void(PDBPageHandle pageHandle));

  MOCK_METHOD0(getNumPages, size_t ());
};

PDBPageHandle getSetPageWithData(pdb::PDBBufferManagerImpl &myMgr) {

  // create a page, loading it with random data
  auto page = myMgr.getPage();
  {
    // this implementation only serves six pages
    static int numPages = 0;
    if (numPages == 6)
      return nullptr;

    // create a page, loading it with random data
    {
      const UseTemporaryAllocationBlock tempBlock{page->getBytes(), 1024 * 1024};

      // write a bunch of supervisors to it
      Handle <Vector <Handle <StringIntPair>>> data = makeObject <Vector <Handle <StringIntPair>>> ();

      for (int i = 0; i < 8000; i++) {
        std::ostringstream oss;
        oss << "My string is " << i;
        oss.str();
        Handle <StringIntPair> myPair = makeObject <StringIntPair> (oss.str (), i);
        data->push_back (myPair);
      }

      getRecord(data);
    }
    numPages++;
  }

  return page;
}

TEST(PipelineTest, TestSelection) {

  /// 1. Create the buffer manager that is going to provide the pages to the pipeline

  // create the buffer manager
  pdb::PDBBufferManagerImpl myMgr;
  myMgr.initialize("tempDSFSD", 2 * 1024 * 1024, 16, "metadata", ".");

  // this is the object allocation block where all of this stuff will reside
  makeObjectAllocatorBlock(1024 * 1024, true);

  /// 2. Create the computation and the corresponding TCAP

  // here is the list of computations
  Vector<Handle<Computation>> myComputations;

  Handle <Computation> readStringIntPair = pdb::makeObject <ReadStringIntPair>();
  Handle <Computation> multiSelection = pdb::makeObject <StringIntPairMultiSelection>();
  multiSelection->setInput(0, readStringIntPair);
  Handle <Computation> writeStringIntPair = pdb::makeObject <WriteStringIntPair>();
  writeStringIntPair->setInput(0, multiSelection);

  // put them in the list of computations
  myComputations.push_back(readStringIntPair);
  myComputations.push_back(multiSelection);
  myComputations.push_back(writeStringIntPair);

  // now we create the TCAP string
  String myTCAPString = "inputDataForSetScanner_0(in0) <= SCAN ('db', 'set', 'SetScanner_0')\n"
                        "nativ_0OutForMultiSelectionComp1(in0,nativ_0_1OutFor) <= APPLY (inputDataForSetScanner_0(in0), inputDataForSetScanner_0(in0), 'MultiSelectionComp_1', 'native_lambda_0', [('lambdaType', 'native_lambda')])\n"
                        "filteredInputForMultiSelectionComp1(in0) <= FILTER (nativ_0OutForMultiSelectionComp1(nativ_0_1OutFor), nativ_0OutForMultiSelectionComp1(in0), 'MultiSelectionComp_1')\n"
                        "nativ_1OutForMultiSelectionComp1 (nativ_1_1OutFor) <= APPLY (filteredInputForMultiSelectionComp1(in0), filteredInputForMultiSelectionComp1(), 'MultiSelectionComp_1', 'native_lambda_1', [('lambdaType', 'native_lambda')])\n"
                        "flattenedOutForMultiSelectionComp1(flattened_nativ_1_1OutFor) <= FLATTEN (nativ_1OutForMultiSelectionComp1(nativ_1_1OutFor), nativ_1OutForMultiSelectionComp1(), 'MultiSelectionComp_1')\n"
                        "flattenedOutForMultiSelectionComp1_out( ) <= OUTPUT ( flattenedOutForMultiSelectionComp1 ( flattened_nativ_1_1OutFor ), '', '', 'SetWriter_2')\n";

  // and create a query object that contains all of this stuff
  ComputePlan myPlan(myTCAPString, myComputations);
  LogicalPlanPtr logicalPlan = myPlan.getPlan();
  AtomicComputationList computationList = logicalPlan->getComputations();

  /// 3. Setup the mock calls to the PageSets for the input and the output

  // empty computations parameters
  std::map<ComputeInfoType, ComputeInfoPtr> params = {{ ComputeInfoType::SOURCE_SET_INFO, std::make_shared<pdb::SourceSetArg>(std::make_shared<PDBCatalogSet>("db", "set", "", 0, PDB_CATALOG_SET_VECTOR_CONTAINER)) }};

  // the page set that is gonna provide stuff
  std::shared_ptr<MockPageSetReader> pageReader = std::make_shared<MockPageSetReader>();

  // make the function return pages with Employee objects
  ON_CALL(*pageReader, getNextPage(testing::An<size_t>())).WillByDefault(testing::Invoke([&](size_t workerID) {
    return getSetPageWithData(myMgr);
  }));

  // it should call send object exactly six times
  EXPECT_CALL(*pageReader, getNextPage(testing::An<size_t>())).Times(7);

  // the page set that is gonna provide stuff
  std::shared_ptr<MockPageSetWriter> pageWriter = std::make_shared<MockPageSetWriter>();

  std::unordered_map<uint64_t, PDBPageHandle> writePages;
  ON_CALL(*pageWriter, getNewPage).WillByDefault(testing::Invoke(
      [&]() {

        // store the page
        auto page = myMgr.getPage();
        writePages[page->whichPage()] = page;

        return page;
      }));

  // it should call this method many times
  EXPECT_CALL(*pageWriter, getNewPage).Times(testing::AtLeast(1));

  ON_CALL(*pageWriter, removePage(testing::An<PDBPageHandle>())).WillByDefault(testing::Invoke(
      [&](PDBPageHandle pageHandle) {
        writePages.erase(pageHandle->whichPage());
      }));

  // it should call send object exactly six times
  EXPECT_CALL(*pageWriter, removePage).Times(testing::Exactly(0));


  /// 4. Build the pipeline

  // now, let's pretend that myPlan has been sent over the network, and we want to execute it... first we build
  // a pipeline into the aggregation operation
  PipelinePtr myPipeline = myPlan.buildPipeline(std::string("inputDataForSetScanner_0"), /* this is the TupleSet the pipeline starts with */
                                                std::string("flattenedOutForMultiSelectionComp1_out"),     /* this is the TupleSet the pipeline ends with */
                                                pageReader,
                                                pageWriter,
                                                params,
                                                20,
                                                1,
                                                1,
                                                0);

  // and now, simply run the pipeline and then destroy it!!!
  myPipeline->run();
  myPipeline = nullptr;

  // and be sure to delete the contents of the ComputePlan object... this always needs to be done
  // before the object is written to disk or sent accross the network, so that we don't end up
  // moving around a C++ smart pointer, which would be bad
  myPlan.nullifyPlanPointer();

  /// 5. Check the results
  std::unordered_map<int, int> counts;
  for(int i = 0; i < 8000; ++i) { counts[i] = 6;}
  for(auto &page : writePages) {

    Handle<Vector<Handle<StringIntPair>>> myHashTable = ((Record<Vector<Handle<StringIntPair>>> *) page.second->getBytes())->getRootObject();
    for (int i = 0; i < myHashTable->size(); i++) {

      EXPECT_EQ(std::string("Hi"), (*myHashTable)[i]->myString->c_str());
      counts[(*myHashTable)[i]->myInt]--;
    }
  }

  // make sure we had every record
  for_each (counts.begin(), counts.end(), [&](auto &count) { EXPECT_EQ(count.second, 0); });
}