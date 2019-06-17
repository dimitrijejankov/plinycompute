#include <gtest/gtest.h>
#include <memory>
#include <PDBBufferManagerImpl.h>
#include <Computation.h>
#include <gtest/gtest.h>
#include <gmock/gmock.h>

#include "ReadInt.h"
#include "ReadStringIntPair.h"
#include "SillyJoinIntString.h"
#include "SillyWriteIntString.h"

namespace pdb {


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

TEST(PipelineTest, TestShuffleJoinSingleReversed) {

  // this is our configuration we are testing
  const uint64_t numNodes = 2;
  const uint64_t threadsPerNode = 2;
  uint64_t curNode = 1;
  uint64_t curThread = 0;

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

  // the page set that is gonna provide stuff
  std::shared_ptr<MockPageSetReader> setBReader = std::make_shared<MockPageSetReader>();

  // make the function return pages with Employee objects
  ON_CALL(*setBReader, getNextPage(testing::An<size_t>())).WillByDefault(testing::Invoke([&](size_t workerID) {
    return getSetBPageWithData(myMgr);
  }));

  EXPECT_CALL(*setBReader, getNextPage(testing::An<size_t>())).Times(7);

  // make the function return pages with the Vector<JoinMap<JoinRecord>>
  std::vector<PDBPageQueuePtr> setAPageQueues;
  std::vector<std::vector<PDBPageHandle>> setAPageVectors;
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

  // make the function return pages with the Vector<JoinMap<JoinRecord>>
  std::vector<PDBPageQueuePtr> setBPageQueues;
  std::vector<std::vector<PDBPageHandle>> setBPageVectors;
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
  Handle <Computation> readA = makeObject <ReadInt>();
  Handle <Computation> readB = makeObject <ReadStringIntPair>();
  Handle <Computation> join = makeObject <SillyJoinIntString>();
  Handle <Computation> write = makeObject <SillyWriteIntString>();

  // put them in the list of computations
  myComputations.push_back (readA);
  myComputations.push_back (readB);
  myComputations.push_back (join);
  myComputations.push_back (write);

  pdb::String tcapString = "A(a) <= SCAN ('myData', 'mySetA', 'SetScanner_0')\n"
                           "B(b) <= SCAN ('myData', 'mySetB', 'SetScanner_1')\n"
                           "A_extracted_value(a,self_0_2Extracted) <= APPLY (A(a), A(a), 'JoinComp_2', 'self_0', [('lambdaType', 'self')])\n"
                           "AHashed(a,a_value_for_hashed) <= HASHLEFT (A_extracted_value(self_0_2Extracted), A_extracted_value(a), 'JoinComp_2', '==_2', [])\n"
                           "B_extracted_value(b,b_value_for_hash) <= APPLY (B(b), B(b), 'JoinComp_2', 'attAccess_1', [('attName', 'myInt'), ('attTypeName', 'int'), ('inputTypeName', 'pdb::StringIntPair'), ('lambdaType', 'attAccess')])\n"
                           "BHashedOnA(b,b_value_for_hashed) <= HASHRIGHT (B_extracted_value(b_value_for_hash), B_extracted_value(b), 'JoinComp_2', '==_2', [])\n"
                           "\n"
                           "/* Join ( a ) and ( b ) */\n"
                           "AandBJoined(a, b) <= JOIN (AHashed(a_value_for_hashed), AHashed(a), BHashedOnA(b_value_for_hashed), BHashedOnA(b), 'JoinComp_2')\n"
                           "AandBJoined_WithLHSExtracted(a,b,LHSExtractedFor_2_2) <= APPLY (AandBJoined(a), AandBJoined(a,b), 'JoinComp_2', 'self_0', [('lambdaType', 'self')])\n"
                           "AandBJoined_WithBOTHExtracted(a,b,LHSExtractedFor_2_2,RHSExtractedFor_2_2) <= APPLY (AandBJoined_WithLHSExtracted(b), AandBJoined_WithLHSExtracted(a,b,LHSExtractedFor_2_2), 'JoinComp_2', 'attAccess_1', [('attName', 'myInt'), ('attTypeName', 'int'), ('inputTypeName', 'pdb::StringIntPair'), ('lambdaType', 'attAccess')])\n"
                           "AandBJoined_BOOL(a,b,bool_2_2) <= APPLY (AandBJoined_WithBOTHExtracted(LHSExtractedFor_2_2,RHSExtractedFor_2_2), AandBJoined_WithBOTHExtracted(a,b), 'JoinComp_2', '==_2', [('lambdaType', '==')])\n"
                           "AandBJoined_FILTERED(a, b) <= FILTER (AandBJoined_BOOL(bool_2_2), AandBJoined_BOOL(a, b), 'JoinComp_2')\n"
                           "\n"
                           "/* run Join projection on ( a b )*/\n"
                           "AandBJoined_Projection (nativ_3_2OutFor) <= APPLY (AandBJoined_FILTERED(a,b), AandBJoined_FILTERED(), 'JoinComp_2', 'native_lambda_3', [('lambdaType', 'native_lambda')])\n"
                           "out( ) <= OUTPUT ( AandBJoined_Projection ( nativ_3_2OutFor ), 'outSet', 'myData', 'SetWriter_3')";

  // and create a query object that contains all of this stuff
  ComputePlan myPlan(tcapString, myComputations);

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

  // put nulls in the queues and pull stuff out
  for(int i = 0; i < numNodes; ++i) {

    // add a null into the queue
    setAPageQueues[i]->enqueue(nullptr);

    // fill-up the vector
    PDBPageHandle page;
    std::vector<PDBPageHandle> tmp;
    do {

      // wait to get the page
      setAPageQueues[i]->wait_dequeue(page);
      tmp.emplace_back(page);

    } while (page != nullptr);

    // move the vector into the vector of vectors of pages for set A
    setAPageVectors.emplace_back(std::move(tmp));
  }

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
  for(int i = 0; i < numNodes; ++i) {

    // add a null into the queue
    setBPageQueues[i]->enqueue(nullptr);

    // fill-up the vector
    PDBPageHandle page;
    std::vector<PDBPageHandle> tmp;
    do {

      // wait to get the page
      setBPageQueues[i]->wait_dequeue(page);
      tmp.emplace_back(page);

    } while (page != nullptr);

    // move the vector into the vector of vectors of pages for set B
    setBPageVectors.emplace_back(std::move(tmp));
  }

  //
  for(curNode = 0; curNode < numNodes; ++curNode) {
    for(curThread = 0; curThread < threadsPerNode; ++curThread) {

      // copy the page back into the queue
      for_each (setAPageVectors[curNode].begin(), setAPageVectors[curNode].end(), [&](PDBPageHandle &page) { setAPageQueues[curNode]->enqueue(page); });
      for_each (setBPageVectors[curNode].begin(), setBPageVectors[curNode].end(), [&](PDBPageHandle &page) { setBPageQueues[curNode]->enqueue(page); });

      /// 6. Do the joining
      params = {{ComputeInfoType::PAGE_PROCESSOR, std::make_shared<NullProcessor>()},
                {ComputeInfoType::JOIN_ARGS, std::make_shared<JoinArguments>(JoinArgumentsInit{{"AHashed", std::make_shared<JoinArg>(partitionedAPageSet)}})},
                {ComputeInfoType::SHUFFLE_JOIN_ARG, std::make_shared<ShuffleJoinArg>(true)}};
      myPipeline = myPlan.buildPipeline(std::string("AandBJoined"), // left side of the join
                                        std::string("out"),     // the final writer
                                        partitionedBPageSet,
                                        pageWriter,
                                        params,
                                        numNodes,
                                        threadsPerNode,
                                        20,
                                        curThread);

      // and now, simply run the pipeline and then destroy it!!!
      myPipeline->run();
      myPipeline = nullptr;
    }
  }

  // we expect each of the join pairs for a particular number to appear 36 time and only numbers from 0 to 7999 are joined
  std::unordered_map<int, int> counts;
  for(int i = 0; i < 8000; ++i) { counts[i] = 36;}
  for(auto &page : writePages) {

    page.second->repin();

    Handle<Vector<Handle<String>>> myVec = ((Record<Vector<Handle<String>>> *) page.second->getBytes())->getRootObject();
    std::cout << "Found that this has " << myVec->size() << " strings in it.\n";
    for(int i = 0; i < myVec->size(); ++i) {

      // extract N from "Got int N and StringIntPair (N, 'My string is N')'";
      std::string tmp = (*myVec)[i]->c_str() + 8;
      std::size_t found = tmp.find(' ');
      tmp.resize(found);
      int n = std::stoi(tmp);

      // check the string
      std::string check = "Got int " + std::to_string(n) + " and StringIntPair ("  + std::to_string(n)  + ", '" + "My string is " + std::to_string(n) + "')'";
      EXPECT_TRUE(check == (*myVec)[i]->c_str());

      // every join result must have an N less than 8000 since the string int pairs go only up to 8000
      EXPECT_LT(n, 8000);

      counts[n]--;
    }

    page.second->unpin();
  }

  // make sure we had every record
  for_each (counts.begin(), counts.end(), [&](auto &count) { EXPECT_EQ(count.second, 0); });

}

}
