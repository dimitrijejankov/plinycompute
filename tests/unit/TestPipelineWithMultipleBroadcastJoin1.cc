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

#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <PDBBufferManagerImpl.h>
#include <processors/BroadcastJoinProcessor.h>
#include <processors/NullProcessor.h>

#include "SillyJoin.h"
#include "SillyReadOfB.h"
#include "SillyReadOfC.h"
#include "SillyWrite.h"
#include "ReadInt.h"
#include "ReadStringIntPair.h"

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
      for (int i = 0; i < 16000; i++) {
        Handle<int> myInt = makeObject<int>(i);
        data->push_back(myInt);
      }
      getRecord(data);
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

      for (int i = 0; i < 1000; i++) {
        std::ostringstream oss;
        oss << "My string is " << i;
        oss.str();
        Handle<StringIntPair> myPair = makeObject<StringIntPair>(oss.str(), i);
        data->push_back(myPair);
      }
      getRecord(data);
    }
    numPages++;
  }

  return page;
}

PDBPageHandle getSetCPageWithData(std::shared_ptr<PDBBufferManagerImpl> &myMgr) {

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

      for (int i = 0; i < 4000; i++) {
        std::ostringstream oss;
        oss << "My string is " << i;
        oss.str();
        Handle<String> myString = makeObject<String>(oss.str());
        data->push_back(myString);
      }
      getRecord(data);
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
  MOCK_METHOD0(resetPageSet, void());
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
TEST(PipelineTest, TestBroadcastJoin) {

  // this is our configuration we are testing
  const uint64_t numNodes = 2;
  const uint64_t threadsPerNode = 2;
  uint64_t curNode = 1;
  uint64_t curThread = 1;

  /// 1. Create the buffer manager that is going to provide the pages to the pipeline
  // create the buffer manager
  std::shared_ptr<PDBBufferManagerImpl> myMgr = std::make_shared<PDBBufferManagerImpl>();
  myMgr->initialize("tempDSFSD", 4 * 1024 * 1024, 16, "metadata", ".");

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
  EXPECT_CALL(*setBReader, getNextPage(testing::An<size_t>())).Times(testing::AtLeast(1));



  // the page set that is gonna provide stuff
  std::shared_ptr<MockPageSetReader> setCReader = std::make_shared<MockPageSetReader>();

  // make the function return pages with Employee objects
  ON_CALL(*setCReader, getNextPage(testing::An<size_t>())).WillByDefault(testing::Invoke([&](size_t workerID) {
    return getSetCPageWithData(myMgr);
  }));
  EXPECT_CALL(*setCReader, getNextPage(testing::An<size_t>())).Times(7);

  std::vector<PDBPageQueuePtr> pageQueuesForA;
  pageQueuesForA.reserve(numNodes);
  for (int i = 0; i < numNodes; ++i) { pageQueuesForA.emplace_back(std::make_shared<PDBPageQueue>()); }

  std::vector<PDBPageQueuePtr> pageQueuesForC;
  pageQueuesForC.reserve(numNodes);
  for (int i = 0; i < numNodes; ++i) { pageQueuesForC.emplace_back(std::make_shared<PDBPageQueue>()); }

  std::vector<std::vector<PDBPageHandle>> setAPageVectors;
  std::vector<std::vector<PDBPageHandle>> setCPageVectors;

  std::shared_ptr<MockPageSetWriter> partitionedAPageSet = std::make_shared<MockPageSetWriter>();
  ON_CALL(*partitionedAPageSet, getNewPage).WillByDefault(testing::Invoke([&]() {
    return myMgr->getPage();
  }));
  EXPECT_CALL(*partitionedAPageSet, getNewPage).Times(testing::AtLeast(1));

  ON_CALL(*partitionedAPageSet, getNextPage(testing::An<size_t>())).WillByDefault(testing::Invoke(
      [&](size_t workerID) {
        // wait to get the page
        PDBPageHandle page;
        pageQueuesForA[curNode]->wait_dequeue(page);
        if (page == nullptr) {
          return (PDBPageHandle) nullptr;
        }
        // repin the page
        page->repin();
        // return it
        return page;
      }));

  EXPECT_CALL(*partitionedAPageSet, getNextPage).Times(testing::AtLeast(1));

  std::shared_ptr<MockPageSetWriter> partitionedCPageSet = std::make_shared<MockPageSetWriter>();
  ON_CALL(*partitionedCPageSet, getNewPage).WillByDefault(testing::Invoke([&]() {
    return myMgr->getPage();
  }));
  EXPECT_CALL(*partitionedCPageSet, getNewPage).Times(testing::AtLeast(1));

  ON_CALL(*partitionedCPageSet, getNextPage(testing::An<size_t>())).WillByDefault(testing::Invoke(
      [&](size_t workerID) {
        // wait to get the page
        PDBPageHandle page;
        pageQueuesForC[curNode]->wait_dequeue(page);
        if (page == nullptr) {
          return (PDBPageHandle) nullptr;
        }
        // repin the page
        page->repin();
        // return it
        return page;
      }));
  EXPECT_CALL(*partitionedCPageSet, getNextPage).Times(testing::AtLeast(1));

  std::queue<PDBPageHandle> BroadcastedAPageSetQueue;
  std::shared_ptr<MockPageSetWriter> BroadcastedAPageSet = std::make_shared<MockPageSetWriter>();
  ON_CALL(*BroadcastedAPageSet, getNewPage).WillByDefault(testing::Invoke([&]() {
    PDBPageHandle myPage = myMgr->getPage();
    BroadcastedAPageSetQueue.push(myPage);
    return myPage;
  }));

  EXPECT_CALL(*BroadcastedAPageSet, getNewPage).Times(testing::AtLeast(1));

  ON_CALL(*BroadcastedAPageSet, getNextPage(testing::An<size_t>())).WillByDefault(testing::Invoke(
      [&](size_t workerID) {
        PDBPageHandle myPage = BroadcastedAPageSetQueue.front();
        BroadcastedAPageSetQueue.pop();
        return myPage;
      }));
  EXPECT_CALL(*BroadcastedAPageSet, getNextPage).Times(testing::AtLeast(0));

  std::queue<PDBPageHandle> BroadcastedCPageSetQueue;
  std::shared_ptr<MockPageSetWriter> BroadcastedCPageSet = std::make_shared<MockPageSetWriter>();

  ON_CALL(*BroadcastedCPageSet, getNewPage).WillByDefault(testing::Invoke([&]() {
    PDBPageHandle myPage = myMgr->getPage();
    BroadcastedCPageSetQueue.push(myPage);
    return myPage;
  }));

  EXPECT_CALL(*BroadcastedCPageSet, getNewPage).Times(testing::AtLeast(1));

  ON_CALL(*BroadcastedCPageSet, getNextPage(testing::An<size_t>())).WillByDefault(testing::Invoke(
      [&](size_t workerID) {
        PDBPageHandle myPage = BroadcastedCPageSetQueue.front();
        BroadcastedCPageSetQueue.pop();
        return myPage;
      }));

  EXPECT_CALL(*BroadcastedCPageSet, getNextPage).Times(testing::AtLeast(0));

  std::shared_ptr<MockPageSetWriter> pageWriter = std::make_shared<MockPageSetWriter>();

  std::unordered_map<uint64_t, PDBPageHandle> writePages;

  ON_CALL(*pageWriter, getNewPage).WillByDefault(testing::Invoke([&]() {
    auto mypage = myMgr->getPage();
    writePages[mypage->whichPage()] = mypage;
    return mypage;
  }));

  EXPECT_CALL(*pageWriter, getNewPage).Times(testing::AtLeast(1));

  ON_CALL(*pageWriter, removePage(testing::An<PDBPageHandle>())).WillByDefault(testing::Invoke(
      [&](PDBPageHandle pageHandle) {
        writePages.erase(pageHandle->whichPage());
      }));

  // it should call send object exactly six times
  EXPECT_CALL(*pageWriter, removePage).Times(testing::Exactly(0));


  /// 3. Create the computations and the TCAP

  // this is the object allocation block where all of this stuff will reside
  makeObjectAllocatorBlock(1024 * 1024, true);

  // here is the list of computations
  Vector<Handle<Computation>> myComputations;

  // create all of the computation objects
  Handle<Computation> readA = makeObject<ReadInt>();
  Handle<Computation> readB = makeObject<ReadStringIntPair>();
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
  String myTCAP = "inputDataForSetScanner_0(in0) <= SCAN ('myData', 'mySetA', 'SetScanner_0')\n"
                  "inputDataForSetScanner_1(in1) <= SCAN ('myData', 'mySetB', 'SetScanner_1')\n"
                  "inputDataForSetScanner_2(in2) <= SCAN ('myData', 'mySetC', 'SetScanner_2')\n"
                  "OutFor_self_2JoinComp3(in0,OutFor_self_2_3) <= APPLY (inputDataForSetScanner_0(in0), inputDataForSetScanner_0(in0), 'JoinComp_3', 'self_2', [('lambdaType', 'self')])\n"
                  "OutFor_attAccess_3JoinComp3(in1,OutFor_attAccess_3_3) <= APPLY (inputDataForSetScanner_1(in1), inputDataForSetScanner_1(in1), 'JoinComp_3', 'attAccess_3', [('attName', 'myInt'), ('attTypeName', 'int'), ('inputTypeName', 'pdb::StringIntPair'), ('lambdaType', 'attAccess')])\n"
                  "OutFor_self_2JoinComp3_hashed(in0,OutFor_self_2_3_hash) <= HASHLEFT (OutFor_self_2JoinComp3(OutFor_self_2_3), OutFor_self_2JoinComp3(in0), 'JoinComp_3', '==_1', [])\n"
                  "OutFor_attAccess_3JoinComp3_hashed(in1,OutFor_attAccess_3_3_hash) <= HASHRIGHT (OutFor_attAccess_3JoinComp3(OutFor_attAccess_3_3), OutFor_attAccess_3JoinComp3(in1), 'JoinComp_3', '==_1', [])\n"
                  "OutForJoinedFor_equals_1JoinComp3(in0,in1) <= JOIN (OutFor_self_2JoinComp3_hashed(OutFor_self_2_3_hash), OutFor_self_2JoinComp3_hashed(in0), OutFor_attAccess_3JoinComp3_hashed(OutFor_attAccess_3_3_hash), OutFor_attAccess_3JoinComp3_hashed(in1), 'JoinComp_3')\n"
                  "LExtractedFor1_self_2JoinComp3(in0,in1,LExtractedFor1_self_2_3) <= APPLY (OutForJoinedFor_equals_1JoinComp3(in0), OutForJoinedFor_equals_1JoinComp3(in0,in1), 'JoinComp_3', 'self_2', [('lambdaType', 'self')])\n"
                  "RExtractedFor1_attAccess_3JoinComp3(in0,in1,LExtractedFor1_self_2_3,RExtractedFor1_attAccess_3_3) <= APPLY (LExtractedFor1_self_2JoinComp3(in1), LExtractedFor1_self_2JoinComp3(in0,in1,LExtractedFor1_self_2_3), 'JoinComp_3', 'attAccess_3', [('attName', 'myInt'), ('attTypeName', 'int'), ('inputTypeName', 'pdb::StringIntPair'), ('lambdaType', 'attAccess')])\n"
                  "OutFor_OutForJoinedFor_equals_1JoinComp3_BOOL(in0,in1,bool_1_3) <= APPLY (RExtractedFor1_attAccess_3JoinComp3(LExtractedFor1_self_2_3,RExtractedFor1_attAccess_3_3), RExtractedFor1_attAccess_3JoinComp3(in0,in1), 'JoinComp_3', '==_1', [('lambdaType', '==')])\n"
                  "OutFor_OutForJoinedFor_equals_1JoinComp3_FILTERED(in0,in1) <= FILTER (OutFor_OutForJoinedFor_equals_1JoinComp3_BOOL(bool_1_3), OutFor_OutForJoinedFor_equals_1JoinComp3_BOOL(in0,in1), 'JoinComp_3')\n"
                  "OutFor_attAccess_5JoinComp3(in0,in1,OutFor_attAccess_5_3) <= APPLY (OutFor_OutForJoinedFor_equals_1JoinComp3_FILTERED(in1), OutFor_OutForJoinedFor_equals_1JoinComp3_FILTERED(in0,in1), 'JoinComp_3', 'attAccess_5', [('attName', 'myString'), ('attTypeName', 'pdb::Handle&lt;pdb::String&gt;'), ('inputTypeName', 'pdb::StringIntPair'), ('lambdaType', 'attAccess')])\n"
                  "OutFor_self_6JoinComp3(in2,OutFor_self_6_3) <= APPLY (inputDataForSetScanner_2(in2), inputDataForSetScanner_2(in2), 'JoinComp_3', 'self_6', [('lambdaType', 'self')])\n"
                  "OutFor_attAccess_5JoinComp3_hashed(in0,in1,OutFor_attAccess_5_3_hash) <= HASHLEFT (OutFor_attAccess_5JoinComp3(OutFor_attAccess_5_3), OutFor_attAccess_5JoinComp3(in0,in1), 'JoinComp_3', '==_4', [])\n"
                  "OutFor_self_6JoinComp3_hashed(in2,OutFor_self_6_3_hash) <= HASHRIGHT (OutFor_self_6JoinComp3(OutFor_self_6_3), OutFor_self_6JoinComp3(in2), 'JoinComp_3', '==_4', [])\n"
                  "OutForJoinedFor_equals_4JoinComp3(in0,in1,in2) <= JOIN (OutFor_attAccess_5JoinComp3_hashed(OutFor_attAccess_5_3_hash), OutFor_attAccess_5JoinComp3_hashed(in0,in1), OutFor_self_6JoinComp3_hashed(OutFor_self_6_3_hash), OutFor_self_6JoinComp3_hashed(in2), 'JoinComp_3')\n"
                  "LExtractedFor4_attAccess_5JoinComp3(in0,in1,in2,LExtractedFor4_attAccess_5_3) <= APPLY (OutForJoinedFor_equals_4JoinComp3(in1), OutForJoinedFor_equals_4JoinComp3(in0,in1,in2), 'JoinComp_3', 'attAccess_5', [('attName', 'myString'), ('attTypeName', 'pdb::Handle&lt;pdb::String&gt;'), ('inputTypeName', 'pdb::StringIntPair'), ('lambdaType', 'attAccess')])\n"
                  "RExtractedFor4_self_6JoinComp3(in0,in1,in2,LExtractedFor4_attAccess_5_3,RExtractedFor4_self_6_3) <= APPLY (LExtractedFor4_attAccess_5JoinComp3(in2), LExtractedFor4_attAccess_5JoinComp3(in0,in1,in2,LExtractedFor4_attAccess_5_3), 'JoinComp_3', 'self_6', [('lambdaType', 'self')])\n"
                  "OutFor_OutForJoinedFor_equals_4JoinComp3_BOOL(in0,in1,in2,bool_4_3) <= APPLY (RExtractedFor4_self_6JoinComp3(LExtractedFor4_attAccess_5_3,RExtractedFor4_self_6_3), RExtractedFor4_self_6JoinComp3(in0,in1,in2), 'JoinComp_3', '==_4', [('lambdaType', '==')])\n"
                  "OutFor_OutForJoinedFor_equals_4JoinComp3_FILTERED(in0,in1,in2) <= FILTER (OutFor_OutForJoinedFor_equals_4JoinComp3_BOOL(bool_4_3), OutFor_OutForJoinedFor_equals_4JoinComp3_BOOL(in0,in1,in2), 'JoinComp_3')\n"
                  "OutFor_native_lambda_7JoinComp3(OutFor_native_lambda_7_3) <= APPLY (OutFor_OutForJoinedFor_equals_4JoinComp3_FILTERED(in0,in1,in2), OutFor_OutForJoinedFor_equals_4JoinComp3_FILTERED(), 'JoinComp_3', 'native_lambda_7', [('lambdaType', 'native_lambda')])\n"
                  "OutFor_native_lambda_7JoinComp3_out( ) <= OUTPUT ( OutFor_native_lambda_7JoinComp3 ( OutFor_native_lambda_7_3 ), 'outSet', 'myDB', 'SetWriter_4')";


  std::cout << myTCAP << std::endl;

  // and create a query object that contains all of this stuff
  ComputePlan myPlan(std::make_shared<LogicalPlan>(myTCAP, myComputations));

  /// 4. Run the pipeline to process the A<int> set. Basically this splits set A into a numNodes * threadsPerNode JoinMaps.
  /// Each page being put into the pageQueue will have numNodes * threadsPerNode  number of JoinMaps. Each join map has records
  /// with the same hash % (numNodes * threadsPerNode). The join map records will be of type JoinTuple<int, char[0]>

  std::map<ComputeInfoType, ComputeInfoPtr> params = {{ComputeInfoType::PAGE_PROCESSOR, std::make_shared<BroadcastJoinProcessor>(numNodes,threadsPerNode, pageQueuesForA, myMgr)},
                                                      {ComputeInfoType::SOURCE_SET_INFO, std::make_shared<pdb::SourceSetArg>(std::make_shared<PDBCatalogSet>("myData", "mySetA","", 0, PDB_CATALOG_SET_VECTOR_CONTAINER))}};
  PipelinePtr myPipeline = myPlan.buildPipeline(std::string("inputDataForSetScanner_0"), /* this is the TupleSet the pipeline starts with */
                                                std::string("OutFor_self_2JoinComp3_hashed"),     /* this is the TupleSet the pipeline ends with */
                                                setAReader,
                                                partitionedAPageSet,
                                                params,
                                                numNodes,
                                                threadsPerNode,
                                                20,
                                                curThread);

  std::cout << "\nRUNNING PIPELINE\n";
  myPipeline->run();
  std::cout << "\nDONE RUNNING PIPELINE\n";
  myPipeline = nullptr;

  for (int i = 0; i < numNodes; ++i) {
    pageQueuesForA[i]->enqueue(nullptr);
    PDBPageHandle myPage;
    std::vector<PDBPageHandle> pageVector;
    do {
      pageQueuesForA[i]->wait_dequeue(myPage);
      pageVector.emplace_back(myPage);
    } while (myPage != nullptr);
    setAPageVectors.emplace_back(std::move(pageVector));
  }

  /// 5. Process the pages in Set A for every worker in each node
  for (curThread = 0; curThread < threadsPerNode; ++curThread) {

    for_each(setAPageVectors[curNode].begin(),
             setAPageVectors[curNode].end(),
             [&](PDBPageHandle &page) { pageQueuesForA[curNode]->enqueue(page); });

    myPipeline = myPlan.buildBroadcastJoinPipeline("OutFor_self_2JoinComp3_hashed",
                                                   partitionedAPageSet,
                                                   BroadcastedAPageSet,
                                                   threadsPerNode,
                                                   numNodes,
                                                   curThread);
    std::cout << "\nRUNNING BROADCAST JOIN PIPELINE FOR SET A\n";
    myPipeline->run();
    std::cout << "\nDONE RUNNING BROADCAST JOIN PIPELINE FOR SET A\n";
    myPipeline = nullptr;
  }

  /// 6. Run the pipeline to process the C<String> set. Basically this splits set C into a numNodes * threadsPerNode JoinMaps.
  /// Each page being put into the pageQueue will have numNodes * threadsPerNode number of JoinMaps. Each join map has records
  /// with the same hash % (numNodes * threadsPerNode). The join map records will be of type JoinTuple<String, char[0]>
  params = {{ComputeInfoType::PAGE_PROCESSOR,std::make_shared<BroadcastJoinProcessor>(numNodes, threadsPerNode, pageQueuesForC, myMgr)},
            {ComputeInfoType::SOURCE_SET_INFO, std::make_shared<pdb::SourceSetArg>(std::make_shared<PDBCatalogSet>("myData","mySetC","",0, PDB_CATALOG_SET_VECTOR_CONTAINER))}};
  myPipeline = myPlan.buildPipeline(std::string("inputDataForSetScanner_2"), /* this is the TupleSet the pipeline starts with */
                                    std::string("OutFor_self_6JoinComp3_hashed"),     /* this is the TupleSet the pipeline ends with */
                                    setCReader,
                                    partitionedCPageSet,
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

  for (int i = 0; i < numNodes; ++i) {
    pageQueuesForC[i]->enqueue(nullptr);
    PDBPageHandle myPage;
    std::vector<PDBPageHandle> pageVector;
    do {
      pageQueuesForC[i]->wait_dequeue(myPage);
      pageVector.emplace_back(myPage);
    } while (myPage != nullptr);
    setCPageVectors.emplace_back(std::move(pageVector));
  }

  /// 7. Process the pages in Set C for every worker in each node
  for (curThread = 0; curThread < threadsPerNode; ++curThread) {
    for_each(setCPageVectors[curNode].begin(),
             setCPageVectors[curNode].end(),
             [&](PDBPageHandle &page) { pageQueuesForC[curNode]->enqueue(page); });

    myPipeline = myPlan.buildBroadcastJoinPipeline("OutFor_self_6JoinComp3_hashed",
                                                   partitionedCPageSet,
                                                   BroadcastedCPageSet,
                                                   threadsPerNode,
                                                   numNodes,
                                                   curThread);
    std::cout << "\nRUNNING BROADCAST JOIN PIPELINE FOR SET C\n";
    myPipeline->run();
    std::cout << "\nDONE RUNNING BROADCAST JOIN PIPELINE FOR SET C\n";
    myPipeline = nullptr;
  }


  BroadcastedAPageSetQueue.push(nullptr);
  BroadcastedCPageSetQueue.push(nullptr);

  /// 8. Process the right side of the Join. Build the pipeline with the Join Argument from pages in Set A and Set C
  unordered_map<string, JoinArgPtr> hashTables = {{"OutFor_self_2JoinComp3_hashed", std::make_shared<JoinArg>(BroadcastedAPageSet)},
                                                  {"OutFor_self_6JoinComp3_hashed", std::make_shared<JoinArg>(BroadcastedCPageSet)}};
  // set the parameters
  params = {{ComputeInfoType::PAGE_PROCESSOR, std::make_shared<NullProcessor>()},
            {ComputeInfoType::JOIN_ARGS, std::make_shared<JoinArguments>(hashTables)},
            {ComputeInfoType::SOURCE_SET_INFO, std::make_shared<pdb::SourceSetArg>(std::make_shared<PDBCatalogSet>("myData","mySetB","",0, PDB_CATALOG_SET_VECTOR_CONTAINER))}};
  myPipeline = myPlan.buildPipeline(std::string("inputDataForSetScanner_1"), /* this is the TupleSet the pipeline starts with */
                                    std::string("OutFor_native_lambda_7JoinComp3_out"),     /* this is the TupleSet the pipeline ends with */
                                    setBReader,
                                    pageWriter,
                                    params,
                                    numNodes,
                                    threadsPerNode,
                                    20,
                                    curThread);

  // and now, simply run the pipeline and then destroy it!!!
  std::cout << "\nRUNNING JOIN PIPELINE\n";
  myPipeline->run();
  std::cout << "\nDONE RUNNING JOIN PIPELINE\n";
  myPipeline = nullptr;

  for (auto &page : writePages) {

    page.second->repin();
    Handle<Vector<Handle<String>>>
        myVec = ((Record<Vector<Handle<String>>> *) page.second->getBytes())->getRootObject();

    std::cout << "Found that this has " << myVec->size() << " strings in it.\n";

    for (int i = 0; i < myVec->size(); ++i) {
      if (i % 1000 == 0) {
        std::cout << *(*myVec)[i] << std::endl;
      }
    }

    page.second->unpin();
  }

}