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

#include "Handle.h"
#include "Lambda.h"
#include "Supervisor.h"
#include "Employee.h"
#include "LambdaCreationFunctions.h"
#include "UseTemporaryAllocationBlock.h"
#include "Pipeline.h"
#include "SetWriter.h"
#include "SelectionComp.h"
#include "AggregateComp.h"
#include "SetScanner.h"
#include "DepartmentTotal.h"
#include "VectorSink.h"
#include "HashSink.h"
#include "MapTupleSetIterator.h"
#include "VectorTupleSetIterator.h"
#include "ComputePlan.h"
#include "PDBAnonymousPageSet.h"
#include "PDBBufferManagerImpl.h"

#include "objects/FinalQuery.h"
#include "objects/SillyAgg.h"
#include "objects/SillyQuery.h"

#include <gtest/gtest.h>
#include <gmock/gmock.h>

using namespace pdb;

class MockPageSetReader : public pdb::PDBAbstractPageSet {
 public:

  MOCK_METHOD1(getNextPage, PDBPageHandle(size_t workerID));

  MOCK_METHOD0(getNewPage, PDBPageHandle());

  MOCK_METHOD0(getNumPages, size_t ());

};

class MockPageSetWriter: public pdb::PDBAnonymousPageSet {
 public:

  MOCK_METHOD1(getNextPage, PDBPageHandle(size_t workerID));

  MOCK_METHOD0(getNewPage, PDBPageHandle());

  MOCK_METHOD1(removePage, void(PDBPageHandle pageHandle));

  MOCK_METHOD0(getNumPages, size_t ());
};


int main() {

  /// 1. Create the buffer manager that is going to provide the pages to the pipeline

  // create the buffer manager
  pdb::PDBBufferManagerImpl myMgr;
  myMgr.initialize("tempDSFSD", 64 * 1024, 16, "metadata", ".");

  // this is the object allocation block where all of this stuff will reside
  const UseTemporaryAllocationBlock tmp{1024 * 1024};

  // here is the list of computations
  Vector<Handle<Computation>> myComputations;

  /// 2. Create the computation and the corresponding TCAP

  // create all of the computation objects
  Handle<Computation> myScanSet = makeObject<SetScanner<Supervisor>>();
  Handle<Computation> myFilter = makeObject<SillyQuery>();
  Handle<Computation> myAgg = makeObject<SillyAgg>();
  Handle<Computation> myFinalFilter = makeObject<FinalQuery>();
  Handle<Computation> myWrite = makeObject<SetWriter<double>>();

  // put them in the list of computations
  myComputations.push_back(myScanSet);
  myComputations.push_back(myFilter);
  myComputations.push_back(myAgg);
  myComputations.push_back(myFinalFilter);
  myComputations.push_back(myWrite);

  // now we create the TCAP string
  String myTCAPString =
      "inputData (in) <= SCAN ('mySet', 'myData', 'SetScanner_0', []) \n\
       inputWithAtt (in, att) <= APPLY (inputData (in), inputData (in), 'SelectionComp_1', 'methodCall_0', []) \n\
       inputWithAttAndMethod (in, att, method) <= APPLY (inputWithAtt (in), inputWithAtt (in, att), 'SelectionComp_1', 'attAccess_1', []) \n\
       inputWithBool (in, bool) <= APPLY (inputWithAttAndMethod (att, method), inputWithAttAndMethod (in), 'SelectionComp_1', '==_2', []) \n\
       filteredInput (in) <= FILTER (inputWithBool (bool), inputWithBool (in), 'SelectionComp_1', []) \n\
       projectedInputWithPtr (out) <= APPLY (filteredInput (in), filteredInput (), 'SelectionComp_1', 'methodCall_3', []) \n\
       projectedInput (out) <= APPLY (projectedInputWithPtr (out), projectedInputWithPtr (), 'SelectionComp_1', 'deref_4', []) \n\
       aggWithKeyWithPtr (out, key) <= APPLY (projectedInput (out), projectedInput (out), 'AggregationComp_2', 'attAccess_0', []) \n\
       aggWithKey (out, key) <= APPLY (aggWithKeyWithPtr (key), aggWithKeyWithPtr (out), 'AggregationComp_2', 'deref_1', []) \n\
       aggWithValue (key, value) <= APPLY (aggWithKey (out), aggWithKey (key), 'AggregationComp_2', 'methodCall_2', []) \n\
       agg (aggOut) <=	AGGREGATE (aggWithValue (key, value), 'AggregationComp_2', []) \n\
       checkSales (aggOut, isSales) <= APPLY (agg (aggOut), agg (aggOut), 'SelectionComp_3', 'methodCall_0', []) \n\
       justSales (aggOut, isSales) <= FILTER (checkSales (isSales), checkSales (aggOut), 'SelectionComp_3', []) \n\
       final (result) <= APPLY (justSales (aggOut), justSales (), 'SelectionComp_3', 'methodCall_1', []) \n\
	   write () <= OUTPUT (final (result), 'outSet', 'myDB', 'SetWriter_4', [])";


  // and create a query object that contains all of this stuff
  Handle<ComputePlan> myPlan = makeObject<ComputePlan>(myTCAPString, myComputations);
  LogicalPlanPtr logicalPlan = myPlan->getPlan();
  AtomicComputationList computationList = logicalPlan->getComputations();
  std::cout << "to print logical plan:" << std::endl;
  std::cout << computationList << std::endl;

  /// 3. Setup the mock calls to the PageSets for the input and the output

  // empty computations parameters
  std::map<std::string, ComputeInfoPtr> params;

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
          pdb::Handle<pdb::Vector<pdb::Handle<pdb::Supervisor>>> supers = pdb::makeObject<pdb::Vector<pdb::Handle<pdb::Supervisor>>>();

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

              Handle<Supervisor>
                  super = makeObject<Supervisor>("Steve Stevens", 20 + (i % 29), std::string(myString), i * 34.4);
              supers->push_back(super);
              for (int j = 0; j < 10; j++) {

                Handle<Employee> temp;
                if (i % 2 == 0)
                  temp = makeObject<Employee>("Steve Stevens", 20 + ((i + j) % 29), std::string(myString), j * 3.54);
                else
                  temp =
                      makeObject<Employee>("Albert Albertson", 20 + ((i + j) % 29), std::string(myString), j * 3.54);
                (*supers)[i]->addEmp(temp);
              }
            }

          } catch (pdb::NotEnoughSpace &e) {

            getRecord (supers);
          }
        }

        numPages++;
        return page;
      }
  ));

  // it should call send object exactly six times
  EXPECT_CALL(*pageReader, getNextPage(testing::An<size_t>())).Times(7);

  // the page set that is gonna provide stuff
  std::shared_ptr<MockPageSetWriter> hashTableRW = std::make_shared<MockPageSetWriter>();

  PDBPageHandle hashTable;
  ON_CALL(*hashTableRW, getNewPage).WillByDefault(testing::Invoke(
      [&]() {

        // the hash table should not exist
        EXPECT_TRUE(hashTable == nullptr);

        // store the page
        auto page = myMgr.getPage();
        hashTable = page;

        return page;
      }));

  // it should call this method many times
  EXPECT_CALL(*hashTableRW, getNewPage).Times(testing::AtLeast(1));

  ON_CALL(*hashTableRW, removePage(testing::An<PDBPageHandle>())).WillByDefault(testing::Invoke(
      [&](PDBPageHandle pageHandle) {}));

  // it should call send object exactly six times
  EXPECT_CALL(*hashTableRW, removePage).Times(testing::Exactly(0));

  // make the function return pages with Employee objects
  ON_CALL(*hashTableRW, getNextPage(testing::An<size_t>())).WillByDefault(testing::Invoke(
      [&](size_t workerID) {
        hashTable->repin();
        return hashTable;
  }));

  // it should call send object exactly six times
  EXPECT_CALL(*hashTableRW, getNextPage).Times(testing::Exactly(1));

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


  // now, let's pretend that myPlan has been sent over the network, and we want to execute it... first we build
  // a pipeline into the aggregation operation
  PipelinePtr myPipeline = myPlan->buildPipeline(std::string("inputData"), /* this is the TupleSet the pipeline starts with */
                                                 std::string("aggWithValue"),     /* this is the TupleSet the pipeline ends with */
                                                 pageReader,
                                                 hashTableRW,
                                                 params,
                                                 20,
                                                 0);

  // and now, simply run the pipeline and then destroy it!!!
  std::cout << "\nRUNNING PIPELINE\n";
  myPipeline->run();
  myPipeline = nullptr;

  // after the destruction of the pointer, the current allocation block is messed up!

  // at this point, the hash table should be filled up...	so now we can build a second pipeline that covers
  // the second half of the aggregation
  myPipeline = myPlan->buildPipeline(std::string("agg"), /* this is the TupleSet the pipeline starts with */
                                     std::string("write"),     /* this is the TupleSet the pipeline ends with */
                                     hashTableRW,
                                     pageWriter,
                                     params,
                                     20,
                                     0);

  // run and then kill the pipeline
  std::cout << "\nRUNNING PIPELINE\n";
  myPipeline->run();
  myPipeline = nullptr;

  // and be sure to delete the contents of the ComputePlan object... this always needs to be done
  // before the object is written to disk or sent accross the network, so that we don't end up
  // moving around a C++ smart pointer, which would be bad
  myPlan->nullifyPlanPointer();

  /// 5. Check the results

  for(auto &page : writePages) {

    page.second->repin();

    Handle<Vector<Handle<double>>> myHashTable = ((Record<Vector<Handle<double>>> *) page.second->getBytes())->getRootObject();
    for (int i = 0; i < myHashTable->size(); i++) {
      std::cout << "Got double " << *((*myHashTable)[i]) << "\n";
    }
  }
}
