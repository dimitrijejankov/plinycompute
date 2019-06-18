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
#include "PDBAnonymousPageSet.h"
#include "PDBBufferManagerImpl.h"

#include "FinalQuery.h"
#include "SillyAgg.h"
#include "SillyQuery.h"

#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <processors/NullProcessor.h>
#include <processors/PreaggregationPageProcessor.h>
#include <WriteSalaries.h>
#include <ScanSupervisorSet.h>

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

PDBPageHandle getPageWithData(size_t &numRecords, std::shared_ptr<PDBBufferManagerImpl> &myMgr) {

  // did we serve enough records
  if(numRecords >= 720 * 3) {
    return nullptr;
  }

  // create a page, loading it with random data
  auto page = myMgr->getPage();
  {
    const pdb::UseTemporaryAllocationBlock tempBlock{page->getBytes(), 64 * 1024};

    // write a bunch of supervisors to it
    pdb::Handle<pdb::Vector<pdb::Handle<pdb::Supervisor>>> supers = pdb::makeObject<pdb::Vector<pdb::Handle<pdb::Supervisor>>>();

    // this will build up the department
    std::string myString = "aa";

    try {
      for (int i = 0; true; i++) {

        // create the supervisor
        Handle<Supervisor> super = makeObject<Supervisor>("Steve Stevens", numRecords, myString + std::to_string(numRecords % 720), 1);
        supers->push_back(super);

        for (int j = 0; j < 10; j++) {

          Handle<Employee> temp;
          if (i % 2 == 0)
            temp = makeObject<Employee>("Steve Stevens", numRecords, myString + std::to_string(numRecords % 720), j * 3.54);
          else
            temp =
                makeObject<Employee>("Albert Albertson", numRecords, myString + std::to_string(numRecords % 720), j * 3.54);
          (*supers)[i]->addEmp(temp);
        }

        numRecords++;

        // if we served 720 * 3 records time to end this
        if(numRecords == 720 * 3) {
          getRecord (supers);
          break;
        }
      }

    } catch (pdb::NotEnoughSpace &e) {

      supers->pop_back();

      getRecord (supers);
    }
  }

  return page;
}



TEST(PipelineTest, TestAggregation) {

  // So basically we first get the data from the "pageReader" page set, this data is a vector of pdb::Supervisor objects
  // you can check out the getPageWithData to get more details. Next we take that page and do the preaggregation,
  // so that we are grouping by the department and summing up the salary. Now the preaggregation will only aggregate
  // as much as it can fit of the input page, and because of that they need to be combined in the next step.
  // In order to combine them in parallel in the next step we split the preaggregated results into numNodes * threadsPerNode
  // partitions. Because of that the output of the preaggregation is a vector of maps each map corresponds to a particular thread
  // and each page has only one node it belongs to (has to be sent). The pages are sent into appropriate blocking queues
  // in the real system there are going to be a bunch of threads that are grabbing pages from these queues and sending them to
  // appropriate nodes. The page set that is used as a sink for these pages is the "partitionedHashTable" page set, it will only give
  // the pages for a particular node determined by curNode
  // next step is the final aggregation where we combine the partitions of the preaggreation pages into a
  // single hash map, basically each thread on a node grabs the same page and only takes in the map that corresponds
  // to that thread and combines it. The output of this is basically a page for each thread on a node, so numNodes * threadsPerNode
  // pages. Each page is going to contain a map the key is the department and the value is the aggregated DepartmentTotal object.
  // These pages are stored in the "hashTablePageSet" page set.
  // Next we scan these maps get the double value from the department total and put that on a page that has a vector of doubles.
  // These pages are stored in the pageWriter hash set.

  // this is our configuration we are testing
  const uint64_t numNodes = 2;
  const uint64_t threadsPerNode = 2;
  const uint64_t curNode = 1;
  const uint64_t curThread = 1;

  /// 1. Create the buffer manager that is going to provide the pages to the pipeline

  // create the buffer manager
  std::shared_ptr<PDBBufferManagerImpl> myMgr = std::make_shared<PDBBufferManagerImpl>();
  myMgr->initialize("tempDSFSD", 64 * 1024, 16, "metadata", ".");

  // this is the object allocation block where all of this stuff will reside
  const UseTemporaryAllocationBlock tmp{1024 * 1024};

  // here is the list of computations
  Vector<Handle<Computation>> myComputations;

  /// 2. Create the computation and the corresponding TCAP

  // create all of the computation objects
  Handle<Computation> myScanSet = makeObject<ScanSupervisorSet>();
  Handle<Computation> myFilter = makeObject<SillyQuery>();
  Handle<Computation> myAgg = makeObject<SillyAgg>();
  Handle<Computation> myFinalFilter = makeObject<FinalQuery>();
  Handle<Computation> myWrite = makeObject<WriteSalaries>();

  // put them in the list of computations
  myComputations.push_back(myScanSet);
  myComputations.push_back(myFilter);
  myComputations.push_back(myAgg);
  myComputations.push_back(myFinalFilter);
  myComputations.push_back(myWrite);

  // now we create the TCAP string
  String myTCAPString =
      "inputData (in) <= SCAN ('myData', 'mySet', 'SetScanner_0', []) \n"
      "inputWithAtt (in, att) <= APPLY (inputData (in), inputData (in), 'SelectionComp_1', 'methodCall_0', []) \n"
      "inputWithAttAndMethod (in, att, method) <= APPLY (inputWithAtt (in), inputWithAtt (in, att), 'SelectionComp_1', 'attAccess_1', []) \n"
      "inputWithBool (in, bool) <= APPLY (inputWithAttAndMethod (att, method), inputWithAttAndMethod (in), 'SelectionComp_1', '==_2', []) \n"
      "filteredInput (in) <= FILTER (inputWithBool (bool), inputWithBool (in), 'SelectionComp_1', []) \n"
      "projectedInputWithPtr (out) <= APPLY (filteredInput (in), filteredInput (), 'SelectionComp_1', 'methodCall_3', []) \n"
      "projectedInput (out) <= APPLY (projectedInputWithPtr (out), projectedInputWithPtr (), 'SelectionComp_1', 'deref_4', []) \n"
      "aggWithKeyWithPtr (out, key) <= APPLY (projectedInput (out), projectedInput (out), 'AggregationComp_2', 'attAccess_0', []) \n"
      "aggWithKey (out, key) <= APPLY (aggWithKeyWithPtr (key), aggWithKeyWithPtr (out), 'AggregationComp_2', 'deref_1', []) \n"
      "aggWithValue (key, value) <= APPLY (aggWithKey (out), aggWithKey (key), 'AggregationComp_2', 'methodCall_2', []) \n"
      "agg (aggOut) <=	AGGREGATE (aggWithValue (key, value), 'AggregationComp_2', []) \n"
      "checkSales (aggOut, isSales) <= APPLY (agg (aggOut), agg (aggOut), 'SelectionComp_3', 'methodCall_0', []) \n"
      "justSales (aggOut, isSales) <= FILTER (checkSales (isSales), checkSales (aggOut), 'SelectionComp_3', []) \n"
      "final (result) <= APPLY (justSales (aggOut), justSales (), 'SelectionComp_3', 'methodCall_1', []) \n"
      "write () <= OUTPUT (final (result), 'outSet', 'myDB', 'SetWriter_4', [])";

  // and create a query object that contains all of this stuff
  ComputePlan myPlan(myTCAPString, myComputations);
  LogicalPlanPtr logicalPlan = myPlan.getPlan();
  AtomicComputationList computationList = logicalPlan->getComputations();
  std::cout << "to print logical plan:" << std::endl;
  std::cout << computationList << std::endl;

  /// 3. Define a bunch of page sets so we can do the aggregation

  // the page set that is gonna provide stuff
  std::shared_ptr<MockPageSetReader> pageReader = std::make_shared<MockPageSetReader>();

  // make the function return pages with Employee objects
  size_t numRecords = 0;
  ON_CALL(*pageReader, getNextPage(testing::An<size_t>())).WillByDefault(testing::Invoke(
      [&](size_t workerID) { return getPageWithData(numRecords, myMgr); }));

  // it should call send object exactly 61 times providing 60 pages
  EXPECT_CALL(*pageReader, getNextPage(testing::An<size_t>())).Times(61);

  // the page set that is going to contain the partitioned preaggregation results
  std::shared_ptr<MockPageSetWriter> partitionedHashTable = std::make_shared<MockPageSetWriter>();

  ON_CALL(*partitionedHashTable, getNewPage).WillByDefault(testing::Invoke(
      [&]() { return myMgr->getPage(); }));

  // it should call this method many times
  EXPECT_CALL(*partitionedHashTable, getNewPage).Times(testing::AtLeast(1));

  ON_CALL(*partitionedHashTable, removePage(testing::An<PDBPageHandle>())).WillByDefault(testing::Invoke(
      [&](PDBPageHandle pageHandle) {}));

  // it should call send object exactly 0 times
  EXPECT_CALL(*partitionedHashTable, removePage).Times(testing::Exactly(0));

  // make the function return pages with the Vector<Map<String, double>>
  std::vector<PDBPageQueuePtr> pageQueues;
  pageQueues.reserve(numNodes);
  for(int i = 0; i < numNodes; ++i) { pageQueues.emplace_back(std::make_shared<PDBPageQueue>()); }

  ON_CALL(*partitionedHashTable, getNextPage(testing::An<size_t>())).WillByDefault(testing::Invoke(
      [&](size_t workerID) {

        // wait to get the page
        PDBPageHandle page;
        pageQueues[curNode]->wait_dequeue(page);

        if(page == nullptr) {
          return (PDBPageHandle) nullptr;
        }

        // repin the page
        page->repin();

        // return it
        return page;
      }));

  // it should call send object exactly six times
  EXPECT_CALL(*partitionedHashTable, getNextPage).Times(testing::AtLeast(1));

  // the page set that is going to contain the aggregated results of a single thread on a particular node
  std::shared_ptr<MockPageSetWriter> hashTablePageSet = std::make_shared<MockPageSetWriter>();

  PDBPageHandle hashTable;
  ON_CALL(*hashTablePageSet, getNewPage).WillByDefault(testing::Invoke(
      [&]() {

        // the hash table should not exist
        EXPECT_TRUE(hashTable == nullptr);

        // store the page
        auto page = myMgr->getPage();
        hashTable = page;

        return page;
      }));

  // this method is going to be called exactly once
  EXPECT_CALL(*hashTablePageSet, getNewPage).Times(testing::Exactly(1));

  ON_CALL(*hashTablePageSet, removePage(testing::An<PDBPageHandle>())).WillByDefault(testing::Invoke(
      [&](PDBPageHandle pageHandle) {}));

  // it should call send object exactly zero times
  EXPECT_CALL(*hashTablePageSet, removePage).Times(testing::Exactly(0));

  // make the function return pages with Employee objects
  ON_CALL(*hashTablePageSet, getNextPage(testing::An<size_t>())).WillByDefault(testing::Invoke(
      [&](size_t workerID) {
        hashTable->repin();
        return hashTable;
      }));

  // it should call send object exactly once
  EXPECT_CALL(*hashTablePageSet, getNextPage).Times(testing::Exactly(1));

  // this page set is going to contain the final results
  std::shared_ptr<MockPageSetWriter> pageWriter = std::make_shared<MockPageSetWriter>();

  std::unordered_map<uint64_t, PDBPageHandle> writePages;
  ON_CALL(*pageWriter, getNewPage).WillByDefault(testing::Invoke(
      [&]() {

        // store the page
        auto page = myMgr->getPage();
        writePages[page->whichPage()] = page;

        return page;
      }));

  // it should call this method many times
  EXPECT_CALL(*pageWriter, getNewPage).Times(testing::AtLeast(1));

  ON_CALL(*pageWriter, removePage(testing::An<PDBPageHandle>())).WillByDefault(testing::Invoke(
      [&](PDBPageHandle pageHandle) {
        writePages.erase(pageHandle->whichPage());
      }));

  // it should call send object exactly 0 times
  EXPECT_CALL(*pageWriter, removePage).Times(testing::Exactly(0));

  /// 4. Create the pre-aggregation and run it.

  // set the parameters
  std::map<ComputeInfoType, ComputeInfoPtr> params = { { ComputeInfoType::PAGE_PROCESSOR,  std::make_shared<PreaggregationPageProcessor>(2, 2, pageQueues, myMgr) },
                                                       { ComputeInfoType::SOURCE_SET_INFO, std::make_shared<pdb::SourceSetArg>(std::make_shared<PDBCatalogSet>("myData", "mySet", "", 0, PDB_CATALOG_SET_VECTOR_CONTAINER)) } };

  // now, let's pretend that myPlan has been sent over the network, and we want to execute it... first we build
  // a pipeline into the aggregation operation
  PipelinePtr myPipeline = myPlan.buildPipeline(std::string("inputData"), /* this is the TupleSet the pipeline starts with */
                                                std::string("aggWithValue"),     /* this is the TupleSet the pipeline ends with */
                                                pageReader,
                                                partitionedHashTable,
                                                params,
                                                numNodes,
                                                threadsPerNode,
                                                20,
                                                0);

  // and now, simply run the pipeline and then destroy it!!!
  std::cout << "\nRUNNING PIPELINE\n";
  myPipeline->run();
  myPipeline = nullptr;

  // add the null to the priority queue so they can end supplying pages
  for(int i = 0; i < numNodes; ++i) { pageQueues[i]->enqueue(nullptr); }

  /// 5. Create the aggregation and run it

  myPipeline = myPlan.buildAggregationPipeline(std::string("aggWithValue"), partitionedHashTable, hashTablePageSet, curThread);

  // and now, simply run the pipeline and then destroy it!!!
  std::cout << "\nRUNNING PIPELINE\n";
  myPipeline->run();
  myPipeline = nullptr;

  // after the destruction of the pointer, the current allocation block is messed up!

  // set he parameters
  params = {};

  /// 6. Create the selection pipeline and run it!

  // at this point, the hash table should be filled up...	so now we can build a second pipeline that covers
  // the second half of the aggregation
  myPipeline = myPlan.buildPipeline(std::string("agg"), /* this is the TupleSet the pipeline starts with */
                                    std::string("write"),     /* this is the TupleSet the pipeline ends with */
                                    hashTablePageSet,
                                    pageWriter,
                                    params,
                                    numNodes,
                                    curNode,
                                    20,
                                    0);

  // run and then kill the pipeline
  std::cout << "\nRUNNING PIPELINE\n";
  myPipeline->run();
  myPipeline = nullptr;

  /// 5. Check the results

  for(auto &page : writePages) {

    page.second->repin();

    Handle<Vector<Handle<double>>> myHashTable = ((Record<Vector<Handle<double>>> *) page.second->getBytes())->getRootObject();

    // expect all 3.0 doubles
    for (int i = 0; i < myHashTable->size(); i++) {
      EXPECT_EQ((int) *((*myHashTable)[i]) , 3);
    }

    page.second->unpin();
  }

  // and be sure to delete the contents of the ComputePlan object... this always needs to be done
  // before the object is written to disk or sent accross the network, so that we don't end up
  // moving around a C++ smart pointer, which would be bad
  myPlan.nullifyPlanPointer();
}
