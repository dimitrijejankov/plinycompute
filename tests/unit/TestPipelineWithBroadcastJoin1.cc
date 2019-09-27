#include <gtest/gtest.h>
#include <memory>
#include <PDBBufferManagerImpl.h>
#include <Computation.h>
#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include "NullProcessor.h"
#include "BroadcastJoinProcessor.h"
#include "SillyJoinIntString.h"
#include "SillyWriteIntString.h"
#include "ReadInt.h"
#include "ReadStringIntPair.h"

namespace pdb {

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
      for (int i = 0; i < 8000; i++) {
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

TEST(PipelineTest, TestBroadcastJoinSingle) {

  ///0. Number of nodes and number of threadsPerNode
  const uint64_t numNodes = 2;
  const uint64_t threadsPerNode = 2;
  uint64_t curNode = 1;
  uint64_t curThread = 0;

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

  // the page set that is gonna provide stuff
  std::shared_ptr<MockPageSetReader> setBReader = std::make_shared<MockPageSetReader>();

  // make the function return pages with Employee objects
  ON_CALL(*setBReader, getNextPage(testing::An<size_t>())).WillByDefault(testing::Invoke([&](size_t workerID) {
    return getSetBPageWithData(myMgr);
  }));

  EXPECT_CALL(*setBReader, getNextPage(testing::An<size_t>())).Times(7);

  // each node will have a concurrent queue for storing pages
  std::vector<PDBPageQueuePtr> pageQueuesForA;
  pageQueuesForA.reserve(numNodes);
  for (int i = 0; i < numNodes; ++i) { pageQueuesForA.emplace_back(std::make_shared<PDBPageQueue>()); }

  // PageVectors are used as a SET for containing all the pages sent to different nodes
  std::vector<std::vector<PDBPageHandle>> setAPageVectors;

  // the page set that is going to contain the partitioned results for set A
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

  EXPECT_CALL(*partitionedAPageSet, getNextPage).Times(testing::AtLeast(0));

  // the queue that containing all the pages already been broadcasted
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
  makeObjectAllocatorBlock(1024 * 1024, true);

  // here is the list of computations
  Vector<Handle<Computation>> myComputations;

  // create all of the computation objects
  Handle <Computation> readA = makeObject <ReadInt>();
  Handle <Computation> readB = makeObject <ReadStringIntPair>();
  Handle<Computation> join = makeObject<SillyJoinIntString>();
  Handle<Computation> write = makeObject<SillyWriteIntString>();

  // put them in the list of computations
  myComputations.push_back(readA);
  myComputations.push_back(readB);
  myComputations.push_back(join);
  myComputations.push_back(write);

  pdb::String tcapString = "inputDataForSetScanner_0(in0) <= SCAN ('myData', 'mySetA', 'SetScanner_0')\n"
                           "inputDataForSetScanner_1(in1) <= SCAN ('myData', 'mySetB', 'SetScanner_1')\n"
                           "OutFor_self_1JoinComp2(in0,OutFor_self_1_2) <= APPLY (inputDataForSetScanner_0(in0), inputDataForSetScanner_0(in0), 'JoinComp_2', 'self_1', [('lambdaType', 'self')])\n"
                           "OutFor_attAccess_2JoinComp2(in1,OutFor_attAccess_2_2) <= APPLY (inputDataForSetScanner_1(in1), inputDataForSetScanner_1(in1), 'JoinComp_2', 'attAccess_2', [('attName', 'myInt'), ('attTypeName', 'int'), ('inputTypeName', 'pdb::StringIntPair'), ('lambdaType', 'attAccess')])\n"
                           "OutFor_self_1JoinComp2_hashed(in0,OutFor_self_1_2_hash) <= HASHLEFT (OutFor_self_1JoinComp2(OutFor_self_1_2), OutFor_self_1JoinComp2(in0), 'JoinComp_2', '==_0', [])\n"
                           "OutFor_attAccess_2JoinComp2_hashed(in1,OutFor_attAccess_2_2_hash) <= HASHRIGHT (OutFor_attAccess_2JoinComp2(OutFor_attAccess_2_2), OutFor_attAccess_2JoinComp2(in1), 'JoinComp_2', '==_0', [])\n"
                           "OutForJoinedFor_equals_0JoinComp2(in0,in1) <= JOIN (OutFor_self_1JoinComp2_hashed(OutFor_self_1_2_hash), OutFor_self_1JoinComp2_hashed(in0), OutFor_attAccess_2JoinComp2_hashed(OutFor_attAccess_2_2_hash), OutFor_attAccess_2JoinComp2_hashed(in1), 'JoinComp_2')\n"
                           "LExtractedFor0_self_1JoinComp2(in0,in1,LExtractedFor0_self_1_2) <= APPLY (OutForJoinedFor_equals_0JoinComp2(in0), OutForJoinedFor_equals_0JoinComp2(in0,in1), 'JoinComp_2', 'self_1', [('lambdaType', 'self')])\n"
                           "RExtractedFor0_attAccess_2JoinComp2(in0,in1,LExtractedFor0_self_1_2,RExtractedFor0_attAccess_2_2) <= APPLY (LExtractedFor0_self_1JoinComp2(in1), LExtractedFor0_self_1JoinComp2(in0,in1,LExtractedFor0_self_1_2), 'JoinComp_2', 'attAccess_2', [('attName', 'myInt'), ('attTypeName', 'int'), ('inputTypeName', 'pdb::StringIntPair'), ('lambdaType', 'attAccess')])\n"
                           "OutFor_OutForJoinedFor_equals_0JoinComp2_BOOL(in0,in1,bool_0_2) <= APPLY (RExtractedFor0_attAccess_2JoinComp2(LExtractedFor0_self_1_2,RExtractedFor0_attAccess_2_2), RExtractedFor0_attAccess_2JoinComp2(in0,in1), 'JoinComp_2', '==_0', [('lambdaType', '==')])\n"
                           "OutFor_OutForJoinedFor_equals_0JoinComp2_FILTERED(in0,in1) <= FILTER (OutFor_OutForJoinedFor_equals_0JoinComp2_BOOL(bool_0_2), OutFor_OutForJoinedFor_equals_0JoinComp2_BOOL(in0,in1), 'JoinComp_2')\n"
                           "OutFor_native_lambda_3JoinComp2(OutFor_native_lambda_3_2) <= APPLY (OutFor_OutForJoinedFor_equals_0JoinComp2_FILTERED(in0,in1), OutFor_OutForJoinedFor_equals_0JoinComp2_FILTERED(), 'JoinComp_2', 'native_lambda_3', [('lambdaType', 'native_lambda')])\n"
                           "OutFor_native_lambda_3JoinComp2_out( ) <= OUTPUT ( OutFor_native_lambda_3JoinComp2 ( OutFor_native_lambda_3_2 ), 'outSet', 'myData', 'SetWriter_3')";

  // and create a query object that contains all of this stuff
  ComputePlan myPlan(std::make_shared<LogicalPlan>(tcapString, myComputations));

  /// 4. Process the left side of the join (set A)
  std::map<ComputeInfoType, ComputeInfoPtr> params = {{ComputeInfoType::PAGE_PROCESSOR, std::make_shared<BroadcastJoinProcessor>(numNodes,threadsPerNode,pageQueuesForA, myMgr)},
                                                      {ComputeInfoType::SOURCE_SET_INFO, std::make_shared<pdb::SourceSetArg>(std::make_shared<PDBCatalogSet>("myData", "mySetA", "", 0, PDB_CATALOG_SET_VECTOR_CONTAINER))}};
  PipelinePtr myPipeline = myPlan.buildPipeline(std::string("inputDataForSetScanner_0"), /* this is the TupleSet the pipeline starts with */
                                                std::string("OutFor_self_1JoinComp2_hashed"),     /* this is the TupleSet the pipeline ends with */
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

  // collect the pages with different nodes into a single setAPageVectors
  for (int i = 0; i < numNodes; ++i) {
    pageQueuesForA[i]->enqueue(nullptr);
    PDBPageHandle page;
    std::vector<PDBPageHandle> tmp;
    do {
      pageQueuesForA[i]->wait_dequeue(page);
      tmp.emplace_back(page);
    } while (page != nullptr);
    setAPageVectors.emplace_back(std::move(tmp));
  }

  /// 5. Process the pages for every worker in each node
  for (curThread = 0; curThread < threadsPerNode; ++curThread) {
    for_each(setAPageVectors[curNode].begin(),
             setAPageVectors[curNode].end(),
             [&](PDBPageHandle &page) { pageQueuesForA[curNode]->enqueue(page); });

    myPipeline = myPlan.buildBroadcastJoinPipeline("OutFor_self_1JoinComp2_hashed",
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
  /// 6. Process the right side of the join (set B). Build the pipeline with the Join Argument from pages in Set A.

  BroadcastedAPageSetQueue.push(nullptr);

  unordered_map<string, JoinArgPtr> hashTables = {{"OutFor_self_1JoinComp2_hashed", std::make_shared<JoinArg>(BroadcastedAPageSet)}};
  // set the parameters
  params = {{ComputeInfoType::PAGE_PROCESSOR, std::make_shared<NullProcessor>()},
            {ComputeInfoType::JOIN_ARGS, std::make_shared<JoinArguments>(hashTables)},
            {ComputeInfoType::SOURCE_SET_INFO, std::make_shared<pdb::SourceSetArg>(std::make_shared<PDBCatalogSet>("myData", "mySetB", "", 0, PDB_CATALOG_SET_VECTOR_CONTAINER))}};

  myPipeline = myPlan.buildPipeline(std::string("inputDataForSetScanner_1"), /* this is the TupleSet the pipeline starts with */
                                    std::string("OutFor_native_lambda_3JoinComp2_out"),     /* this is the TupleSet the pipeline ends with */
                                    setBReader,
                                    pageWriter,
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

}