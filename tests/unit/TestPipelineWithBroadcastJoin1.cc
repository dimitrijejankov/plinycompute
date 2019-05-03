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
#include "AggregateComp.h"
#include "JoinComp.h"
#include "SetScanner.h"
#include "VectorSink.h"
#include "MapTupleSetIterator.h"
#include "VectorTupleSetIterator.h"
#include "ComputePlan.h"
#include "StringIntPair.h"
#include "BroadcastJoinSide.h"

#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <PDBBufferManagerImpl.h>
#include <processors/BroadcastJoinProcessor.h>

#include "objects/SillyJoin.h"
#include "objects/SillyReadOfA.h"
#include "objects/SillyReadOfB.h"
#include "objects/SillyReadOfC.h"
#include "objects/SillyWrite.h"

// to run the aggregate, the system first passes each through the hash operation...
// then the system
using namespace pdb;

PDBPageHandle getSetAPageWithData(std::shared_ptr<PDBBufferManagerImpl> &myMgr) {

  // create a page, loading it with random data
  auto page = myMgr->getPage();
  {
    // this implementation only serves six pages
    static int numPages = 0;
    if (numPages == 6)
      return nullptr;

    // create a page, loading it with random data
    {
      const UseTemporaryAllocationBlock tempBlock{page->getBytes(), 1024 * 1024};

      // write a bunch of supervisors to it
      Handle<Vector<Handle<int>>> data = makeObject<Vector<Handle<int>>>();
      int i = 0;
      try {
        for (; true; i++) {
          Handle<int> myInt = makeObject<int>(i);
          data->push_back(myInt);
        }
      } catch (NotEnoughSpace &e) {
        std::cout << "got to " << i << " when proucing data for SillyReadOfA.\n";
        getRecord(data);
      }
    }
    numPages++;
  }

  return page;
}

PDBPageHandle getSetBPageWithData(std::shared_ptr<PDBBufferManagerImpl> &myMgr) {

  // create a page, loading it with random data
  auto page = myMgr->getPage();
  {
    // this implementation only serves six pages
    static int numPages = 0;
    if (numPages == 6)
      return nullptr;

    // create a page, loading it with random data
    {
      const UseTemporaryAllocationBlock tempBlock{page->getBytes(), 1024 * 1024};

      // write a bunch of supervisors to it
      Handle<Vector<Handle<StringIntPair>>> data = makeObject<Vector<Handle<StringIntPair>>>();
      int i = 0;
      try {
        for (; true; i++) {
          std::ostringstream oss;
          oss << "My string is " << i;
          oss.str();
          Handle<StringIntPair> myPair = makeObject<StringIntPair>(oss.str(), i);
          data->push_back(myPair);
        }
      } catch (NotEnoughSpace &e) {
        std::cout << "got to " << i << " when proucing data for SillyReadOfB.\n";
        getRecord(data);
      }
    }
    numPages++;
  }

  return page;
}

PDBPageHandle getSetCPageWithData(std::shared_ptr<PDBBufferManagerImpl> &myMgr){

  // create a page, loading it with random data
  auto page = myMgr->getPage();
  {
    // this implementation only serves six pages
    static int numPages = 0;
    if (numPages == 6)
      return nullptr;

    // create a page, loading it with random data
    {
      const UseTemporaryAllocationBlock tempBlock{page->getBytes(), 1024 * 1024};

      // write a bunch of supervisors to it
      Handle<Vector<Handle<String>>> data = makeObject<Vector<Handle<String>>>();
      int i = 0;
      try {
        for (; true; i++) {
          std::ostringstream oss;
          oss << "My string is " << i;
          oss.str();
          Handle<String> myString = makeObject<String>(oss.str());
          data->push_back(myString);
        }
      } catch (NotEnoughSpace &e) {
        std::cout << "got to " << i << " when proucing data for SillyReadOfC.\n";
        getRecord(data);
      }
    }
    numPages++;
  }
  return page;
}
class MockPageSetReader : public pdb::PDBAbstractPageSet {
 public:
  MOCK_METHOD1(getNextPage, PDBPageHandle(size_t
      workerID));
  MOCK_METHOD0(getNewPage, PDBPageHandle());
  MOCK_METHOD0(getNumPages, size_t());
};

class MockPageSetWriter : public pdb::PDBAnonymousPageSet {
 public:
  MOCK_METHOD1(getNextPage, PDBPageHandle(size_t
      workerID));
  MOCK_METHOD0(getNewPage, PDBPageHandle());
  MOCK_METHOD1(removePage, void(PDBPageHandle
      pageHandle));
  MOCK_METHOD0(getNumPages, size_t());
};

TEST(PipelineTest, TestShuffleJoin) {

  // this is our configuration we are testing
  const uint64_t numNodes = 2;
  const uint64_t threadsPerNode = 2;
  const uint64_t curNode = 1;
  const uint64_t curThread = 1;

  /// 1. Create the buffer manager that is going to provide the pages to the pipeline
  // create the buffer manager
  std::shared_ptr<PDBBufferManagerImpl> myMgr = std::make_shared<PDBBufferManagerImpl>();
  myMgr->initialize("tempDSFSD", 2 * 1024 * 1024, 16, "metadata", ".");

  /// 2. Init the page sets
  // the page set that is gonna provide stuff
  std::shared_ptr<MockPageSetReader> setAReader = std::make_shared<MockPageSetReader>();

  // make the function return pages with Employee objects
  ON_CALL(*setAReader, getNextPage(testing::An<size_t>())).WillByDefault(testing::Invoke([&](size_t workerID) {
    return getSetAPageWithData(myMgr);
  }));

  EXPECT_CALL(*setAReader, getNextPage(testing::An<size_t>())).Times(7);

  // make the function return pages with the Vector<JoinMap<JoinRecord>>

  // the page set that is gonna provide stuff
  std::shared_ptr<MockPageSetReader> setBReader = std::make_shared<MockPageSetReader>();

  // make the function return pages with Employee objects
  ON_CALL(*setBReader, getNextPage(testing::An<size_t>())).WillByDefault(testing::Invoke([&](size_t workerID) {
    return getSetBPageWithData(myMgr);
  }));

  // TODO:
  EXPECT_CALL(*setBReader, getNextPage(testing::An<size_t>())).Times(7);

  

  // the page set that is gonna provide stuff
  std::shared_ptr<MockPageSetReader> setCReader = std::make_shared<MockPageSetReader>();

  // make the function return pages with Employee objects
  ON_CALL(*setCReader, getNextPage(testing::An<size_t>())).WillByDefault(testing::Invoke([&](size_t workerID) {
    return getSetCPageWithData(myMgr);
  }));

  // TODO:
  EXPECT_CALL(*setCReader, getNextPage(testing::An<size_t>())).Times(7);





  std::vector<PDBPageQueuePtr> pageQueuesForA;
  pageQueuesForA.reserve(numNodes);
  for (int i = 0; i < numNodes; ++i) { pageQueuesForA.emplace_back(std::make_shared<PDBPageQueue>()); }

  std::vector<PDBPageQueuePtr> pageQueuesForC;
  pageQueuesForC.reserve(numNodes);
  for (int i = 0; i < numNodes; ++i){pageQueuesForC.emplace_back(std::make_shared<PDBPageQueue>()); }





  // TODO: the page set that is going to contain the partitioned preaggregation results
  std::shared_ptr<MockPageSetWriter> partitionedAPageSet = std::make_shared<MockPageSetWriter>();

  ON_CALL(*partitionedAPageSet, getNewPage).WillByDefault(testing::Invoke([&]() {
    return myMgr->getPage();
  }));
  EXPECT_CALL(*partitionedAPageSet, getNewPage).Times(testing::AtLeast(1));
  // TODO: the page set that is going to contain the partitioned preaggregation results
  ON_CALL(*partitionedAPageSet, getNextPage(testing::An<size_t>())).WillByDefault(testing::Invoke(
      [&](size_t workerID) {
        // wait to get the page
        PDBPageHandle page;
        pageQueuesForA[workerID]->wait_dequeue(page);
        if (page == nullptr) {
          return (PDBPageHandle) nullptr;
        }
        // repin the page
        page->repin();
        // return it
        return page;
      }));
  // TODO: it should call send object exactly six times
  EXPECT_CALL(*partitionedAPageSet, getNextPage).Times(testing::AtLeast(1));


  
  
  
  std::shared_ptr<MockPageSetWriter> partitionedCPageSet = std::make_shared<MockPageSetWriter>();

  // TODO:
  ON_CALL(*partitionedCPageSet, getNewPage).WillByDefault(testing::Invoke([&]() {
    return myMgr->getPage();
  }));

  // TODO:
  EXPECT_CALL(*partitionedCPageSet, getNewPage).Times(testing::AtLeast(1));

  // TODO: the page set that is going to contain the partitioned results
  ON_CALL(*partitionedCPageSet, getNextPage(testing::An<size_t>())).WillByDefault(testing::Invoke(
      [&](size_t workerID) {

        // wait to get the page
        PDBPageHandle page;
        pageQueuesForC[workerID]->wait_dequeue(page);

        if (page == nullptr) {
          return (PDBPageHandle) nullptr;
        }

        // repin the page
        page->repin();

        // return it
        return page;
      }));

  // TODO: it should call send object exactly six times
  EXPECT_CALL(*partitionedCPageSet, getNextPage).Times(testing::AtLeast(1));


  
  std::shared_ptr<MockPageSetWriter> BroadcastedAPageSet = std::make_shared<MockPageSetWriter>();
// TODO:
  ON_CALL(*BroadcastedAPageSet, getNewPage).WillByDefault(testing::Invoke([&]() {
    return myMgr->getPage();
  }));
// TODO:
  EXPECT_CALL(*BroadcastedAPageSet, getNewPage).Times(testing::AtLeast(1));


  // the page set that is going to contain the partitioned preaggregation results
  std::shared_ptr<MockPageSetWriter> BroadcastedCPageSet = std::make_shared<MockPageSetWriter>();

  ON_CALL(*BroadcastedCPageSet, getNewPage).WillByDefault(testing::Invoke([&]() {
    return myMgr->getPage();
  }));

  EXPECT_CALL(*BroadcastedCPageSet, getNewPage).Times(testing::AtLeast(1));

  // the page set that is going to contain the partitioned preaggregation results



  std::shared_ptr<MockPageSetWriter> andAndBJoinedPageSet = std::make_shared<MockPageSetWriter>();

  ON_CALL(*andAndBJoinedPageSet, getNewPage).WillByDefault(testing::Invoke([&]() {
    return myMgr->getPage();
  }));

  EXPECT_CALL(*andAndBJoinedPageSet, getNewPage).Times(testing::AtLeast(1));



  /// 3. Create the computations and the TCAP

  // this is the object allocation block where all of this stuff will reside
  makeObjectAllocatorBlock(1024 * 1024, true);

  // here is the list of computations
  Vector<Handle<Computation>> myComputations;

  // create all of the computation objects
  Handle<Computation> readA = makeObject<SillyReadOfA>();
  Handle<Computation> readB = makeObject<SillyReadOfB>();
  Handle<Computation> readC = makeObject<SillyReadOfC>();
  Handle<Computation> myJoin = makeObject<SillyJoin>();
  Handle<Computation> myWriter = makeObject<SillyWrite>();

  // put them in the list of computations
  myComputations.push_back(readA);
  myComputations.push_back(readB);
  myComputations.push_back(readC);
  myComputations.push_back(myJoin);
  myComputations.push_back(myWriter);

  // now we create the TCAP string
  String myTCAPString =
      "/* scan the three inputs */ \n"
      "A (a) <= SCAN ('mySet', 'myData', 'SetScanner_0', []) \n"
      "B (aAndC) <= SCAN ('mySet', 'myData', 'SetScanner_1', []) \n"
      "C (c) <= SCAN ('mySet', 'myData', 'SetScanner_2', []) \n"
      "\n"
      "/* extract and hash a from A */ \n"
      "AWithAExtracted (a, aExtracted) <= APPLY (A (a), A(a), 'JoinComp_3', 'self_0', []) \n"
      "AHashed (a, hash) <= HASHLEFT (AWithAExtracted (aExtracted), A (a), 'JoinComp_3', '==_2', []) \n"
      "\n"
      "/* extract and hash a from B */ \n"
      "BWithAExtracted (aAndC, a) <= APPLY (B (aAndC), B (aAndC), 'JoinComp_3', 'attAccess_1', []) \n"
      "BHashedOnA (aAndC, hash) <= HASHRIGHT (BWithAExtracted (a), BWithAExtracted (aAndC), 'JoinComp_3', '==_2', []) \n"
      "\n"
      "/* now, join the two of them */ \n"
      "AandBJoined (a, aAndC) <= JOIN (AHashed (hash), AHashed (a), BHashedOnA (hash), BHashedOnA (aAndC), 'JoinComp_3', []) \n"
      "\n"
      "/* and extract the two atts and check for equality */ \n"
      "AandBJoinedWithAExtracted (a, aAndC, aExtracted) <= APPLY (AandBJoined (a), AandBJoined (a, aAndC), 'JoinComp_3', 'self_0', []) \n"
      "AandBJoinedWithBothExtracted (a, aAndC, aExtracted, otherA) <= APPLY (AandBJoinedWithAExtracted (aAndC), AandBJoinedWithAExtracted (a, aAndC, aExtracted), 'JoinComp_3', 'attAccess_1', []) \n"
      "AandBJoinedWithBool (aAndC, a, bool) <= APPLY (AandBJoinedWithBothExtracted (aExtracted, otherA), AandBJoinedWithBothExtracted (aAndC, a), 'JoinComp_3', '==_2', []) \n"
      "AandBJoinedFiltered (a, aAndC) <= FILTER (AandBJoinedWithBool (bool), AandBJoinedWithBool (a, aAndC), 'JoinComp_3', []) \n"
      "\n"
      "/* now get ready to join the strings */ \n"
      "AandBJoinedFilteredWithC (a, aAndC, cExtracted) <= APPLY (AandBJoinedFiltered (aAndC), AandBJoinedFiltered (a, aAndC), 'JoinComp_3', 'attAccess_3', []) \n"
      "BHashedOnC (a, aAndC, hash) <= HASHLEFT (AandBJoinedFilteredWithC (cExtracted), AandBJoinedFilteredWithC (a, aAndC), 'JoinComp_3', '==_5', []) \n"
      "CwithCExtracted (c, cExtracted) <= APPLY (C (c), C (c), 'JoinComp_3', 'self_0', []) \n"
      "CHashedOnC (c, hash) <= HASHRIGHT (CwithCExtracted (cExtracted), CwithCExtracted (c), 'JoinComp_3', '==_5', []) \n"
      "\n"
      "/* join the two of them */ \n"
      "BandCJoined (a, aAndC, c) <= JOIN (BHashedOnC (hash), BHashedOnC (a, aAndC), CHashedOnC (hash), CHashedOnC (c), 'JoinComp_3', []) \n"
      "\n"
      "/* and extract the two atts and check for equality */ \n"
      "BandCJoinedWithCExtracted (a, aAndC, c, cFromLeft) <= APPLY (BandCJoined (aAndC), BandCJoined (a, aAndC, c), 'JoinComp_3', 'attAccess_3', []) \n"
      "BandCJoinedWithBoth (a, aAndC, c, cFromLeft, cFromRight) <= APPLY (BandCJoinedWithCExtracted (c), BandCJoinedWithCExtracted (a, aAndC, c, cFromLeft), 'JoinComp_3', 'self_4', []) \n"
      "BandCJoinedWithBool (a, aAndC, c, bool) <= APPLY (BandCJoinedWithBoth (cFromLeft, cFromRight), BandCJoinedWithBoth (a, aAndC, c), 'JoinComp_3', '==_5', []) \n"
      "last (a, aAndC, c) <= FILTER (BandCJoinedWithBool (bool), BandCJoinedWithBool (a, aAndC, c), 'JoinComp_3', []) \n"
      "\n"
      "/* and here is the answer */ \n"
      "almostFinal (result) <= APPLY (last (a, aAndC, c), last (), 'JoinComp_3', 'native_lambda_7', []) \n"
      "nothing () <= OUTPUT (almostFinal (result), 'outSet', 'myDB', 'SetWriter_4', [])";

  std::cout << myTCAPString << std::endl;

  // and create a query object that contains all of this stuff
  ComputePlan myPlan(myTCAPString, myComputations);

  /// 4. Run the pipeline to process the A<int> set. Basically this splits set A into a numNodes * threadsPerNode JoinMaps.
  /// Each page being put into the pageQueue will have threadsPerNode number of JoinMaps. Each join map has records
  /// with the same hash % (numNodes * threadsPerNode). The join map records will be of type JoinTuple<int, char[0]>

  // set the parameters


  std::map<ComputeInfoType, ComputeInfoPtr> params = { { ComputeInfoType::PAGE_PROCESSOR, std::make_shared<BroadcastJoinProcessor>(numNodes, threadsPerNode, pageQueuesForA, myMgr)},
                                                       { ComputeInfoType::JOIN_SIDE, std::make_shared<pdb::BroadcastJoinSide>(BroadcastJoinSideEnum ::PROBE_SIDE)}};
  PipelinePtr myPipeline = myPlan.buildPipeline(std::string("A"), /* this is the TupleSet the pipeline starts with */
                                                std::string("AHashed"),     /* this is the TupleSet the pipeline ends with */
                                                setAReader,
                                                partitionedAPageSet,
                                                params,
                                                numNodes,
                                                threadsPerNode,
                                                20,
                                                curThread);

  // and now, simply run the pipeline and then destroy it!!!
  std::cout << "\nRUNNING PIPELINE\n";
  myPipeline->run();
  std::cout << "\nDONE RUNNING PIPELINE\n";
  myPipeline = nullptr;

  for (int i = 0; i < numNodes; ++i) { pageQueuesForA[i]->enqueue(nullptr); }

  myPipeline = myPlan.buildMergeJoinBroadcastPipeline("AHashed", partitionedAPageSet,BroadcastedAPageSet,threadsPerNode, curThread);

  std::cout << "\nRUNNING PIPELINE\n";
  myPipeline->run();
  std::cout << "\nDONE RUNNING PIPELINE\n";
  myPipeline = nullptr;

















  params = {{ComputeInfoType::JOIN_SIDE, std::make_shared<pdb::BroadcastJoinSide>(BroadcastJoinSideEnum::BUILD_SIDE)}};
  myPipeline = myPlan.buildPipeline(std::string("B"), /* this is the TupleSet the pipeline starts with */
                                    std::string(""),     /* this is the TupleSet the pipeline ends with */
                                    partitionedAPageSet,
                                    partitionedBPageSet,
                                    params,
                                    numNodes,
                                    threadsPerNode,
                                    20,
                                    curThread);

  // and now, simply run the pipeline and then destroy it!!!
  std::cout << "\nRUNNING PIPELINE\n";
  myPipeline->run();
  std::cout << "\nDONE RUNNING PIPELINE\n";
  myPipeline = nullptr;


  /// 7. This pipeline will build the probe side for the other join

  unordered_map<string, JoinArgPtr> hashTables = {{"", std::make_shared<JoinArg>(BroadcastedAPageSet)}};
  // set the parameters
  params = {{ComputeInfoType::PAGE_PROCESSOR,
             std::make_shared<BroadcastJoinProcessor>(numNodes, threadsPerNode, pageQueuesForA, myMgr)},
            {ComputeInfoType::JOIN_SIDE, std::make_shared<pdb::BroadcastJoinSide>(BroadcastJoinSideEnum::PROBE_SIDE)},
            {ComputeInfoType::JOIN_ARGS, std::make_shared<JoinArguments>(hashTables)}};
  myPipeline = myPlan.buildPipeline(std::string("AandBJoined"), /* this is the TupleSet the pipeline starts with */
                                    std::string("BHashedOnC"),     /* this is the TupleSet the pipeline ends with */
                                    partitionedBPageSet,
                                    andAndBJoinedPageSet,
                                    params,
                                    numNodes,
                                    threadsPerNode,
                                    20,
                                    curThread);

  // and now, simply run the pipeline and then destroy it!!!
  std::cout << "\nRUNNING PIPELINE\n";
  myPipeline->run();
  std::cout << "\nDONE RUNNING PIPELINE\n";
  myPipeline = nullptr;

  /*
  for (int i = 0; i < numNodes; ++i) { pageQueuesForB[i]->enqueue(nullptr); }
  myPipeline = myPlan.buildMergeJoinBroadcastPipeline("BHashedOnA", partitionedBPageSet, BroadcastedBPageSet, threadsPerNode, curThread);
  std::cout << "\nRUNNING PIPELINE\n";
  myPipeline->run();
  std::cout << "\nDONE RUNNING PIPELINE\n";
  myPipeline = nullptr;
  */

}
