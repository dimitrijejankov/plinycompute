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

TEST(PipelineTest, TestSelection) {

  /// 1. Create the buffer manager that is going to provide the pages to the pipeline

  // create the buffer manager
  pdb::PDBBufferManagerImpl myMgr;
  myMgr.initialize("tempDSFSD", 64 * 1024, 16, "metadata", ".");

  // this is the object allocation block where all of this stuff will reside
  makeObjectAllocatorBlock(1024 * 1024, true);

  /// 2. Create the computation and the corresponding TCAP

  // here is the list of computations
  Vector<Handle<Computation>> myComputations;

  Handle<Computation> myScanSet = makeObject<ScanEmployeeSet>();
  Handle<Computation> myQuery = makeObject<EmployeeBuiltInIdentitySelection>();
  myQuery->setInput(myScanSet);
  Handle<Computation> myWriteSet = makeObject<WriteBuiltinEmployeeSet>("by8_db", "output_set");
  myWriteSet->setInput(myQuery);

  // put them in the list of computations
  myComputations.push_back(myScanSet);
  myComputations.push_back(myQuery);
  myComputations.push_back(myWriteSet);

  // now we create the TCAP string
  String myTCAPString =
  "inputDataForScanSet_0(in0) <= SCAN ('by8_db', 'input_set', 'SetScanner_0') \n"\
  "nativ_0OutForSelectionComp1(in0,nativ_0_1OutFor) <= APPLY (inputDataForScanSet_0(in0), inputDataForScanSet_0(in0), 'SelectionComp_1', 'native_lambda_0', [('lambdaType', 'native_lambda')]) \n"\
  "filteredInputForSelectionComp1(in0) <= FILTER (nativ_0OutForSelectionComp1(nativ_0_1OutFor), nativ_0OutForSelectionComp1(in0), 'SelectionComp_1') \n"\
  "nativ_1OutForSelectionComp1 (nativ_1_1OutFor) <= APPLY (filteredInputForSelectionComp1(in0), filteredInputForSelectionComp1(), 'SelectionComp_1', 'native_lambda_1', [('lambdaType', 'native_lambda')]) \n"\
  "nativ_1OutForSelectionComp1_out() <= OUTPUT ( nativ_1OutForSelectionComp1 ( nativ_1_1OutFor ), 'output_set', 'by8_db', 'SetWriter_2') \n";

  // and create a query object that contains all of this stuff
  ComputePlan myPlan(myTCAPString, myComputations);
  LogicalPlanPtr logicalPlan = myPlan.getPlan();
  AtomicComputationList computationList = logicalPlan->getComputations();

  /// 3. Setup the mock calls to the PageSets for the input and the output

  // empty computations parameters
  std::map<ComputeInfoType, ComputeInfoPtr> params = {{ ComputeInfoType::SOURCE_SET_INFO, std::make_shared<pdb::SourceSetArg>(std::make_shared<PDBCatalogSet>("by8_db", "input_set", "", 0, PDB_CATALOG_SET_VECTOR_CONTAINER)) }};

  // the page set that is gonna provide stuff
  std::shared_ptr<MockPageSetReader> pageReader = std::make_shared<MockPageSetReader>();

  // make the function return pages with Employee objects
  ON_CALL(*pageReader, getNextPage(testing::An<size_t>())).WillByDefault(testing::Invoke(
      [&](size_t workerID) {

        // this implementation only serves six pages
        static int numPages = 0;
        if (numPages == 6)
          return (PDBPageHandle) nullptr;

        // create a page, loading it with random data
        auto page = myMgr.getPage();
        {
          const pdb::UseTemporaryAllocationBlock tempBlock{page->getBytes(), 64 * 1024};

          // write a bunch of supervisors to it
          pdb::Handle<pdb::Vector<pdb::Handle<pdb::Employee>>> employees = pdb::makeObject<pdb::Vector<pdb::Handle<pdb::Employee>>>();

          // this will build up the department
          char first = 'A', second = 'B';
          char myString[3];
          myString[2] = 0;

          try {
            for (int i = 0; true; i++) {

              myString[0] = first;
              myString[1] = second;

              // this will allow us to cycle through "AA", "AB", "AC", "BA", ...
              first++;
              if (first == 'D') {
                first = 'A';
                second++;
                if (second == 'D')
                  second = 'A';
              }

              if(i % 2 == 0) {

                pdb::Handle<pdb::Employee> temp = pdb::makeObject<pdb::Employee>("Steve Stevens", 20 + ((i) % 29), std::string(myString), i * 3.54);
                employees->push_back(temp);
              }
              else {
                pdb::Handle<pdb::Employee> temp = pdb::makeObject<pdb::Employee>("Ninja Turtles", 20 + ((i) % 29), std::string(myString), i * 3.54);
                employees->push_back(temp);
              }
            }
          } catch (pdb::NotEnoughSpace &e) {

            getRecord (employees);
          }
        }
        numPages++;
        return page;
      }
  ));

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
  PipelinePtr myPipeline = myPlan.buildPipeline(std::string("inputDataForScanSet_0"), /* this is the TupleSet the pipeline starts with */
                                                std::string("nativ_1OutForSelectionComp1_out"),     /* this is the TupleSet the pipeline ends with */
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

  EXPECT_TRUE(!writePages.empty());
  for(auto &page : writePages) {

    Handle<Vector<Handle<Employee>>> myHashTable = ((Record<Vector<Handle<Employee>>> *) page.second->getBytes())->getRootObject();
    for (int i = 0; i < myHashTable->size(); i++) {
      EXPECT_TRUE(*(((*myHashTable)[i])->getName()) == "Steve Stevens" || *(((*myHashTable)[i])->getName()) == "Ninja Turtles");
    }
  }

}