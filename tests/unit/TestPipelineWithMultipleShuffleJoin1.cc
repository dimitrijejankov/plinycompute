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
#include <processors/ShuffleJoinProcessor.h>
#include <processors/NullProcessor.h>

#include "SillyJoin.h"
#include "ReadInt.h"
#include "ReadStringIntPair.h"
#include "SillyReadOfC.h"
#include "SillyWrite.h"

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
        std::cout << "got to " << i << " when proucing data for ReadInt.\n";
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
      Handle <Vector <Handle <StringIntPair>>> data = makeObject <Vector <Handle <StringIntPair>>> ();
      int i = 0;
      try {
        for (; true; i++) {
          std::ostringstream oss;
          oss << "My string is " << i;
          oss.str();
          Handle <StringIntPair> myPair = makeObject <StringIntPair> (oss.str (), i);
          data->push_back (myPair);
        }
      } catch (NotEnoughSpace &e) {
        std::cout << "got to " << i << " when proucing data for ReadStringIntPair.\n";
        getRecord(data);
      }
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
      Handle <Vector <Handle <String>>> data = makeObject <Vector <Handle <String>>> ();

      int j = 0;
      try {
        for (int i = 0; true; i += 3) {
          std::ostringstream oss;
          oss << "My string is " << i;
          oss.str();
          Handle <String> myString = makeObject <String> (oss.str ());
          data->push_back (myString);
          j++;
        }
      } catch (NotEnoughSpace &e) {
        std :: cout << "got to " << j << " when proucing data for SillyReadOfC.\n";
        getRecord (data);
      }
    }
    numPages++;
  }

  return page;
}

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

TEST(PipelineTest, TestShuffleJoin) {

  // this is our configuration we are testing
  const uint64_t numNodes = 2;
  const uint64_t threadsPerNode = 2;
  const uint64_t curNode = 1;
  const uint64_t curThread = 0;

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
  std::vector<PDBPageQueuePtr> setAPageQueues;
  setAPageQueues.reserve(numNodes);
  for(int i = 0; i < numNodes; ++i) { setAPageQueues.emplace_back(std::make_shared<PDBPageQueue>()); }

  // the page set that is going to contain the partitioned preaggregation results
  std::shared_ptr<MockPageSetWriter> partitionedAPageSet = std::make_shared<MockPageSetWriter>();

  ON_CALL(*partitionedAPageSet, getNewPage).WillByDefault(testing::Invoke([&]() {
    return myMgr->getPage();
  }));

  EXPECT_CALL(*partitionedAPageSet, getNewPage).Times(testing::AtLeast(1));

  // the page set that is going to contain the partitioned preaggregation results
  ON_CALL(*partitionedAPageSet, getNextPage(testing::An<size_t>())).WillByDefault(testing::Invoke(
      [&](size_t workerID) {

        // wait to get the page
        PDBPageHandle page;
        setAPageQueues[curNode]->wait_dequeue(page);

        if(page == nullptr) {
          return (PDBPageHandle) nullptr;
        }

        // repin the page
        page->repin();

        // return it
        return page;
      }));

  // it should call send object exactly six times
  EXPECT_CALL(*partitionedAPageSet, getNextPage).Times(testing::AtLeast(0));

  // the page set that is gonna provide stuff
  std::shared_ptr<MockPageSetReader> setBReader = std::make_shared<MockPageSetReader>();

  // make the function return pages with Employee objects
  ON_CALL(*setBReader, getNextPage(testing::An<size_t>())).WillByDefault(testing::Invoke([&](size_t workerID) {
    return getSetBPageWithData(myMgr);
  }));

  EXPECT_CALL(*setBReader, getNextPage(testing::An<size_t>())).Times(7);

  // the page set that is gonna provide stuff
  std::shared_ptr<MockPageSetReader> setCReader = std::make_shared<MockPageSetReader>();

  // make the function return pages with Employee objects
  ON_CALL(*setCReader, getNextPage(testing::An<size_t>())).WillByDefault(testing::Invoke([&](size_t workerID) {
    return getSetCPageWithData(myMgr);
  }));

  EXPECT_CALL(*setCReader, getNextPage(testing::An<size_t>())).Times(7);

  // make the function return pages with the Vector<JoinMap<JoinRecord>>
  std::vector<PDBPageQueuePtr> setBPageQueues;
  setBPageQueues.reserve(numNodes);
  for(int i = 0; i < numNodes; ++i) { setBPageQueues.emplace_back(std::make_shared<PDBPageQueue>()); }

  // the page set that is going to contain the partitioned preaggregation results
  std::shared_ptr<MockPageSetWriter> partitionedBPageSet = std::make_shared<MockPageSetWriter>();

  ON_CALL(*partitionedBPageSet, getNewPage).WillByDefault(testing::Invoke([&]() {
    return myMgr->getPage();
  }));

  EXPECT_CALL(*partitionedBPageSet, getNewPage).Times(testing::AtLeast(1));

  // the page set that is going to contain the partitioned preaggregation results
  ON_CALL(*partitionedBPageSet, getNextPage(testing::An<size_t>())).WillByDefault(testing::Invoke(
      [&](size_t workerID) {

        // wait to get the page
        PDBPageHandle page;
        setBPageQueues[curNode]->wait_dequeue(page);

        if(page == nullptr) {
          return (PDBPageHandle) nullptr;
        }

        // repin the page
        page->repin();

        // return it
        return page;
      }));

  // it should call send object exactly six times
  EXPECT_CALL(*partitionedBPageSet, getNextPage).Times(testing::AtLeast(0));

  // make the function return pages with the Vector<JoinMap<JoinRecord>>
  std::vector<PDBPageQueuePtr> setCPageQueues;
  setCPageQueues.reserve(numNodes);
  for(int i = 0; i < numNodes; ++i) { setCPageQueues.emplace_back(std::make_shared<PDBPageQueue>()); }

  // the page set that is going to contain the partitioned preaggregation results
  std::shared_ptr<MockPageSetWriter> partitionedCPageSet = std::make_shared<MockPageSetWriter>();

  ON_CALL(*partitionedCPageSet, getNewPage).WillByDefault(testing::Invoke([&]() {
    return myMgr->getPage();
  }));

  EXPECT_CALL(*partitionedCPageSet, getNewPage).Times(testing::AtLeast(1));

  // the page set that is going to contain the partitioned preaggregation results
  ON_CALL(*partitionedCPageSet, getNextPage(testing::An<size_t>())).WillByDefault(testing::Invoke(
      [&](size_t workerID) {

        // wait to get the page
        PDBPageHandle page;
        setCPageQueues[curNode]->wait_dequeue(page);

        if(page == nullptr) {
          return (PDBPageHandle) nullptr;
        }

        // repin the page
        page->repin();

        // return it
        return page;
      }));

  // it should call send object exactly six times
  EXPECT_CALL(*partitionedCPageSet, getNextPage).Times(testing::AtLeast(0));

  // make the function
  std::vector<PDBPageQueuePtr> setAndBPageQueues;
  setAndBPageQueues.reserve(numNodes);
  for(int i = 0; i < numNodes; ++i) { setAndBPageQueues.emplace_back(std::make_shared<PDBPageQueue>()); }

  // the page set that is going to contain the partitioned preaggregation results
  std::shared_ptr<MockPageSetWriter> AndBJoinedPageSet = std::make_shared<MockPageSetWriter>();

  ON_CALL(*AndBJoinedPageSet, getNewPage).WillByDefault(testing::Invoke([&]() {
    return myMgr->getPage();
  }));

  EXPECT_CALL(*AndBJoinedPageSet, getNewPage).Times(testing::AtLeast(0));

  // the page set that is going to contain the partitioned preaggregation results
  ON_CALL(*AndBJoinedPageSet, getNextPage(testing::An<size_t>())).WillByDefault(testing::Invoke(
      [&](size_t workerID) {

        // wait to get the page
        PDBPageHandle page;
        setAndBPageQueues[curNode]->wait_dequeue(page);

        if(page == nullptr) {
          return (PDBPageHandle) nullptr;
        }

        // repin the page
        page->repin();

        // return it
        return page;
      }));

  // it should call send object exactly six times
  EXPECT_CALL(*AndBJoinedPageSet, getNextPage).Times(testing::AtLeast(0));

  // the page set where we will write the final result
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

  // it should call send object exactly six times
  EXPECT_CALL(*pageWriter, removePage).Times(testing::Exactly(0));

  /// 3. Create the computations and the TCAP

  // this is the object allocation block where all of this stuff will reside
  makeObjectAllocatorBlock (1024 * 1024, true);

  // here is the list of computations
  Vector <Handle <Computation>> myComputations;

  // create all of the computation objects
  Handle <Computation> readA = makeObject <ReadInt> ();
  Handle <Computation> readB = makeObject <ReadStringIntPair> ();
  Handle <Computation> readC = makeObject <SillyReadOfC> ();
  Handle <Computation> myJoin = makeObject <SillyJoin> ();
  Handle <Computation> myWriter = makeObject <SillyWrite> ();

  // put them in the list of computations
  myComputations.push_back (readA);
  myComputations.push_back (readB);
  myComputations.push_back (readC);
  myComputations.push_back (myJoin);
  myComputations.push_back (myWriter);

  // now we create the TCAP string
  String myTCAP = "A(in0) <= SCAN ('myData', 'mySetA', 'SetScanner_0')\n"
                  "B(in1) <= SCAN ('myData', 'mySetB', 'SetScanner_1')\n"
                  "C(in2) <= SCAN ('myData', 'mySetC', 'SetScanner_2')\n"
                  "AWithAExtracted(in0,OutFor_self_2_3) <= APPLY (A(in0), A(in0), 'JoinComp_3', 'self_2', [('lambdaType', 'self')])\n"
                  "AHashed(in0,OutFor_self_2_3_hash) <= HASHLEFT (AWithAExtracted(OutFor_self_2_3), AWithAExtracted(in0), 'JoinComp_3', '==_1', [])\n"
                  "BWithAExtracted(in1,OutFor_attAccess_3_3) <= APPLY (B(in1), B(in1), 'JoinComp_3', 'attAccess_3', [('attName', 'myInt'), ('attTypeName', 'int'), ('inputTypeName', 'pdb::StringIntPair'), ('lambdaType', 'attAccess')])\n"
                  "BHashedOnA(in1,OutFor_attAccess_3_3_hash) <= HASHRIGHT (BWithAExtracted(OutFor_attAccess_3_3), BWithAExtracted(in1), 'JoinComp_3', '==_1', [])\n"
                  "AandBJoined(in0,in1) <= JOIN (AHashed(OutFor_self_2_3_hash), AHashed(in0), BHashedOnA(OutFor_attAccess_3_3_hash), BHashedOnA(in1), 'JoinComp_3')\n"
                  "AandBJoinedWithAExtracted(in0,in1,LExtractedFor1_self_2_3) <= APPLY (AandBJoined(in0), AandBJoined(in0,in1), 'JoinComp_3', 'self_2', [('lambdaType', 'self')])\n"
                  "AandBJoinedWithBothExtracted(in0,in1,LExtractedFor1_self_2_3,RExtractedFor1_attAccess_3_3) <= APPLY (AandBJoinedWithAExtracted(in1), AandBJoinedWithAExtracted(in0,in1,LExtractedFor1_self_2_3), 'JoinComp_3', 'attAccess_3', [('attName', 'myInt'), ('attTypeName', 'int'), ('inputTypeName', 'pdb::StringIntPair'), ('lambdaType', 'attAccess')])\n"
                  "AandBJoinedWithBool(in0,in1,bool_1_3) <= APPLY (AandBJoinedWithBothExtracted(LExtractedFor1_self_2_3,RExtractedFor1_attAccess_3_3), AandBJoinedWithBothExtracted(in0,in1), 'JoinComp_3', '==_1', [('lambdaType', '==')])\n"
                  "AandBJoinedFiltered(in0,in1) <= FILTER (AandBJoinedWithBool(bool_1_3), AandBJoinedWithBool(in0,in1), 'JoinComp_3')\n"
                  "AandBJoinedFilteredWithC(in0,in1,OutFor_attAccess_5_3) <= APPLY (AandBJoinedFiltered(in1), AandBJoinedFiltered(in0,in1), 'JoinComp_3', 'attAccess_5', [('attName', 'myString'), ('attTypeName', 'pdb::Handle&lt;pdb::String&gt;'), ('inputTypeName', 'pdb::StringIntPair'), ('lambdaType', 'attAccess')])\n"
                  "BHashedOnC(in0,in1,OutFor_attAccess_5_3_hash) <= HASHLEFT (AandBJoinedFilteredWithC(OutFor_attAccess_5_3), AandBJoinedFilteredWithC(in0,in1), 'JoinComp_3', '==_4', [])\n"
                  "CwithCExtracted(in2,OutFor_self_6_3) <= APPLY (C(in2), C(in2), 'JoinComp_3', 'self_6', [('lambdaType', 'self')])\n"
                  "CHashedOnC(in2,OutFor_self_6_3_hash) <= HASHRIGHT (CwithCExtracted(OutFor_self_6_3), CwithCExtracted(in2), 'JoinComp_3', '==_4', [])\n"
                  "BandCJoined(in0,in1,in2) <= JOIN (BHashedOnC(OutFor_attAccess_5_3_hash), BHashedOnC(in0,in1), CHashedOnC(OutFor_self_6_3_hash), CHashedOnC(in2), 'JoinComp_3')\n"
                  "BandCJoinedWithCExtracted(in0,in1,in2,LExtractedFor4_attAccess_5_3) <= APPLY (BandCJoined(in1), BandCJoined(in0,in1,in2), 'JoinComp_3', 'attAccess_5', [('attName', 'myString'), ('attTypeName', 'pdb::Handle&lt;pdb::String&gt;'), ('inputTypeName', 'pdb::StringIntPair'), ('lambdaType', 'attAccess')])\n"
                  "BandCJoinedWithBoth(in0,in1,in2,LExtractedFor4_attAccess_5_3,RExtractedFor4_self_6_3) <= APPLY (BandCJoinedWithCExtracted(in2), BandCJoinedWithCExtracted(in0,in1,in2,LExtractedFor4_attAccess_5_3), 'JoinComp_3', 'self_6', [('lambdaType', 'self')])\n"
                  "BandCJoinedWithBool(in0,in1,in2,bool_4_3) <= APPLY (BandCJoinedWithBoth(LExtractedFor4_attAccess_5_3,RExtractedFor4_self_6_3), BandCJoinedWithBoth(in0,in1,in2), 'JoinComp_3', '==_4', [('lambdaType', '==')])\n"
                  "last(in0,in1,in2) <= FILTER (BandCJoinedWithBool(bool_4_3), BandCJoinedWithBool(in0,in1,in2), 'JoinComp_3')\n"
                  "almostFinal(OutFor_native_lambda_7_3) <= APPLY (last(in0,in1,in2), last(), 'JoinComp_3', 'native_lambda_7', [('lambdaType', 'native_lambda')])\n"
                  "nothing( ) <= OUTPUT ( almostFinal ( OutFor_native_lambda_7_3 ), 'outSet', 'myDB', 'SetWriter_4')";

  // and create a query object that contains all of this stuff
  ComputePlan myPlan(std::make_shared<LogicalPlan>(myTCAP, myComputations));

  /// 4. Process the left side of the join (set A)

  // set the parameters
  std::map<ComputeInfoType, ComputeInfoPtr> params = { { ComputeInfoType::PAGE_PROCESSOR,  myPlan.getProcessorForJoin("AHashed", numNodes, threadsPerNode, setAPageQueues, myMgr) },
                                                       { ComputeInfoType::SOURCE_SET_INFO, std::make_shared<pdb::SourceSetArg>(std::make_shared<PDBCatalogSet>("myData", "mySetA", "", 0, PDB_CATALOG_SET_VECTOR_CONTAINER)) } };

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
  std :: cout << "\nRUNNING PIPELINE\n";
  myPipeline->run ();
  std :: cout << "\nDONE RUNNING PIPELINE\n";
  myPipeline = nullptr;

  // put nulls in the queues
  for(int i = 0; i < numNodes; ++i) { setAPageQueues[i]->enqueue(nullptr); }

  /// 5. Process the right side of the join (set B)

  params = { { ComputeInfoType::PAGE_PROCESSOR,  myPlan.getProcessorForJoin("BHashedOnA", numNodes, threadsPerNode, setBPageQueues, myMgr) },
             { ComputeInfoType::SOURCE_SET_INFO, std::make_shared<pdb::SourceSetArg>(std::make_shared<PDBCatalogSet>("myData", "mySetB", "", 0, PDB_CATALOG_SET_VECTOR_CONTAINER)) } };
  myPipeline = myPlan.buildPipeline(std::string("B"), /* this is the TupleSet the pipeline starts with */
                                    std::string("BHashedOnA"),     /* this is the TupleSet the pipeline ends with */
                                    setBReader,
                                    partitionedBPageSet,
                                    params,
                                    numNodes,
                                    threadsPerNode,
                                    20,
                                    curThread);

  // and now, simply run the pipeline and then destroy it!!!
  std :: cout << "\nRUNNING PIPELINE\n";
  myPipeline->run ();
  std :: cout << "\nDONE RUNNING PIPELINE\n";
  myPipeline = nullptr;

  // put nulls in the queues
  for(int i = 0; i < numNodes; ++i) { setBPageQueues[i]->enqueue(nullptr); }

  /// 6. Build the join pipeline (This joins A and B and prepares the right side of the next join)

  params = { { ComputeInfoType::PAGE_PROCESSOR,  myPlan.getProcessorForJoin("BHashedOnC", numNodes, threadsPerNode, setAndBPageQueues, myMgr) },
             { ComputeInfoType::JOIN_ARGS,  std::make_shared<JoinArguments>(JoinArgumentsInit {{"BHashedOnA", std::make_shared<JoinArg>(partitionedBPageSet)}}) },
             { ComputeInfoType::SHUFFLE_JOIN_ARG, std::make_shared<ShuffleJoinArg>(false) } };
  myPipeline = myPlan.buildPipeline(std::string("AandBJoined"), /* this is the TupleSet the pipeline starts with */
                                    std::string("BHashedOnC"),     /* this is the TupleSet the pipeline ends with */
                                    partitionedAPageSet,
                                    AndBJoinedPageSet,
                                    params,
                                    numNodes,
                                    threadsPerNode,
                                    20,
                                    curThread);
  myPipeline->run();

  // put nulls in the queues
  for(int i = 0; i < numNodes; ++i) { setAndBPageQueues[i]->enqueue(nullptr); }

  /// 7. Process the set C (this becomes the left side of the join)

  params = { { ComputeInfoType::PAGE_PROCESSOR,  myPlan.getProcessorForJoin("CHashedOnC", numNodes, threadsPerNode, setCPageQueues, myMgr) },
             { ComputeInfoType::SOURCE_SET_INFO, std::make_shared<pdb::SourceSetArg>(std::make_shared<PDBCatalogSet>("myData", "mySetC", "", 0, PDB_CATALOG_SET_VECTOR_CONTAINER)) } };
  myPipeline = myPlan.buildPipeline(std::string("C"), /* this is the TupleSet the pipeline starts with */
                                    std::string("CHashedOnC"),     /* this is the TupleSet the pipeline ends with */
                                    setCReader,
                                    partitionedCPageSet,
                                    params,
                                    numNodes,
                                    threadsPerNode,
                                    20,
                                    curThread);

  // and now, simply run the pipeline and then destroy it!!!
  std :: cout << "\nRUNNING PIPELINE\n";
  myPipeline->run ();
  std :: cout << "\nDONE RUNNING PIPELINE\n";
  myPipeline = nullptr;

  // put nulls in the queues
  for(int i = 0; i < numNodes; ++i) { setCPageQueues[i]->enqueue(nullptr); }

  /// 8. Do the joining
  params = { { ComputeInfoType::PAGE_PROCESSOR, std::make_shared<NullProcessor>() },
             { ComputeInfoType::JOIN_ARGS,  std::make_shared<JoinArguments>(JoinArgumentsInit {{"CHashedOnC", std::make_shared<JoinArg>(partitionedCPageSet)}}) },
             { ComputeInfoType::SHUFFLE_JOIN_ARG, std::make_shared<ShuffleJoinArg>(false) }};
  myPipeline = myPlan.buildPipeline(std::string("BandCJoined"), // left side of the join
                                    std::string("nothing"),     // the final writer
                                    AndBJoinedPageSet,
                                    pageWriter,
                                    params,
                                    numNodes,
                                    threadsPerNode,
                                    20,
                                    curThread);

  // and now, simply run the pipeline and then destroy it!!!
  myPipeline->run();
  myPipeline = nullptr;

  for(auto &page : writePages) {

    page.second->repin();

    Handle<Vector<Handle<String>>> myVec = ((Record<Vector<Handle<String>>> *) page.second->getBytes())->getRootObject();
    std::cout << "Found that this has " << myVec->size() << " strings in it.\n";
    if (myVec->size() > 0)
      std::cout << "First one is '" << *((*myVec)[56]) << "'\n";
  }
}