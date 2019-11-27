#include <gtest/gtest.h>
#include <gmock/gmock-generated-function-mockers.h>
#include <gmock/gmock-more-actions.h>
#include <QueryGraphAnalyzer.h>
#include <UseTemporaryAllocationBlock.h>
#include <Computation.h>
#include <KeyComputePlan.h>
#include <AtomicComputationClasses.h>
#include <PDBBufferManagerImpl.h>
#include <LogicalPlanTransformer.h>

#include "../../../../applications/TestMatrixMultiply/sharedLibraries/headers/MatrixBlock.h"
#include "../../../../applications/TestMatrixMultiply/sharedLibraries/headers/MatrixScanner.h"
#include "../../../../applications/TestMatrixMultiply/sharedLibraries/headers/MatrixMultiplyJoin.h"
#include "../../../../applications/TestMatrixMultiply/sharedLibraries/headers/MatrixMultiplyAggregation.h"
#include "../../../../applications/TestMatrixMultiply/sharedLibraries/headers/MatrixWriter.h"

namespace pdb {

using namespace matrix;

class MockPageSetReader : public pdb::PDBAbstractPageSet {
 public:

  MOCK_METHOD1(getNextPage, PDBPageHandle(size_t workerID));

  MOCK_METHOD0(getNewPage, PDBPageHandle());

  MOCK_METHOD0(getNumPages, size_t ());

  MOCK_METHOD0(resetPageSet, void ());

  MOCK_METHOD1(removePage, void(PDBPageHandle pageHandle));

  MOCK_METHOD0(getMaxPageSize, size_t ());
};

class MockPageSetWriter: public pdb::PDBAnonymousPageSet {
 public:

  MockPageSetWriter(const PDBBufferManagerInterfacePtr &bufferManager) : pdb::PDBAnonymousPageSet(bufferManager) {}

  MOCK_METHOD1(getNextPage, PDBPageHandle(size_t workerID));

  MOCK_METHOD0(getNewPage, PDBPageHandle());

  MOCK_METHOD1(removePage, void(PDBPageHandle pageHandle));

  MOCK_METHOD0(getNumPages, size_t ());
};

std::tuple<pdb::PipelinePtr, pdb::PipelinePtr> buildHashingPipeline(std::shared_ptr<KeyComputePlan> &computePlan,
                                                                    const pdb::PDBBufferManagerInterfacePtr &myMgr,
                                                                    const pdb::PDBAbstractPageSetPtr &lhsReader,
                                                                    const pdb::PDBAnonymousPageSetPtr &lhsWriter,
                                                                    const pdb::PDBAbstractPageSetPtr &rhsReader,
                                                                    const pdb::PDBAnonymousPageSetPtr &rhsWriter) {

  // get the atomic computations and the source atomic computations
  auto &computations = computePlan->getPlan()->getComputations();
  auto &sources = computations.getAllScanSets();

  // there should be exactly two sources
  if(sources.size() != 2) {
    return { nullptr, nullptr };
  }

  // get the left source comp
  auto lhsSource = computations.getProducingAtomicComputation("inputDataForSetScanner_0");

  // make the parameters for the first set
  std::map<ComputeInfoType, ComputeInfoPtr> params = {{ ComputeInfoType::SOURCE_SET_INFO, std::make_shared<pdb::SourceSetArg>(std::make_shared<PDBCatalogSet>(((ScanSet*)lhsSource.get())->getDBName(), ((ScanSet*)lhsSource.get())->getSetName(), "", 0, PDB_CATALOG_SET_VECTOR_CONTAINER)) }};

  // build the pipeline
  auto leftPipeline = computePlan->buildHashPipeline("inputDataForSetScanner_0", lhsReader, lhsWriter, params);

  // the get the right source comp
  auto rhsSource = computations.getProducingAtomicComputation("inputDataForSetScanner_1");

  // init the parameters for rhs
  params = {{ ComputeInfoType::SOURCE_SET_INFO, std::make_shared<pdb::SourceSetArg>(std::make_shared<PDBCatalogSet>(((ScanSet*)rhsSource.get())->getDBName(), ((ScanSet*)rhsSource.get())->getSetName(), "", 0, PDB_CATALOG_SET_VECTOR_CONTAINER)) }};

  // build the pipeline
  auto rightPipeline = computePlan->buildHashPipeline("inputDataForSetScanner_1", rhsReader, rhsWriter, params);

  // return the pipelines
  return {leftPipeline, rightPipeline};
}

pdb::PipelinePtr buildJoinAggPipeline(std::shared_ptr<KeyComputePlan> &computePlan,
                                      const pdb::PDBAbstractPageSetPtr &lhsReader,
                                      const pdb::PDBAbstractPageSetPtr &rhsReader,
                                      const pdb::PDBAnonymousPageSetPtr &writer,
                                      const PDBPageHandle &leftKeyPage,
                                      const PDBPageHandle &rightKeyPage) {

  // get the atomic computations and the source atomic computations
  auto &computations = computePlan->getPlan()->getComputations();
  auto &sources = computations.getAllScanSets();

  // try to find the join
  auto joinList = computations.findByPredicate([](AtomicComputationPtr &c){

    // check if this is a join
    return c->getAtomicComputationTypeID() == ApplyJoinTypeID;
  });

  // make sure we have only one join
  if(joinList.size() != 1){
    throw runtime_error("Could not find a join!");
  }

  auto &joinComp = *joinList.begin();

  // try to find the sink
  auto sinkList = computations.findByPredicate([&computations](AtomicComputationPtr &c){

    // check if this is a sink
    return computations.getConsumingAtomicComputations(c->getOutputName()).empty();
  });

  // make sure we have only one join
  if(sinkList.size() != 1){
    throw runtime_error("Could not find an aggregation!");
  }

  // get the join computation
  auto &sinkComp = *sinkList.begin();

  // the key aggregation processor
  auto aggComputation = ((AggregateCompBase*)(&computePlan->getPlan()->getNode(sinkComp->getComputationName()).getComputation()));

  //
  std::map<ComputeInfoType, ComputeInfoPtr> params;
  params = {{ComputeInfoType::PAGE_PROCESSOR, aggComputation->getAggregationKeyProcessor()},
            {ComputeInfoType::JOIN_ARGS, std::make_shared<JoinArguments>(JoinArgumentsInit{{joinComp->getRightInput().getSetName(), std::make_shared<JoinArg>(rhsReader)}})},
            {ComputeInfoType::KEY_JOIN_SOURCE_ARGS, std::make_shared<KeyJoinSourceArgs>(std::vector<PDBPageHandle>({leftKeyPage, rightKeyPage}))},
            {ComputeInfoType::SHUFFLE_JOIN_ARG, std::make_shared<ShuffleJoinArg>(false)}};

  PipelinePtr myPipeline = computePlan->buildJoinAggPipeline(joinComp->getOutputName(),
                                                             sinkComp->getOutputName(),     /* this is the TupleSet the pipeline ends with */
                                                             lhsReader,
                                                             writer,
                                                             params,
                                                             1,
                                                             1,
                                                             20,
                                                             0);

  // return the pipeline
  return std::move(myPipeline);
}

std::string getTCAPString(std::vector<Handle<Computation>> &queryGraph) {

  // create the graph analyzer
  pdb::QueryGraphAnalyzer queryAnalyzer(queryGraph);

  // here is the list of computations
  Handle<Vector<Handle<Computation>>> myComputations = makeObject<Vector<Handle<Computation>>>();

  // parse the tcap string
  std::string tcapString = queryAnalyzer.parseTCAPString(*myComputations);

  // return the tcap string
  return std::move(tcapString);
}

TEST(PipelineTest, TestJoinAggPipeline) {

  /// 1. Create the buffer manager that is going to provide the pages to the pipeline

  // create the buffer manager
  pdb::PDBBufferManagerInterfacePtr myMgr = std::make_shared<PDBBufferManagerImpl>();
  ((PDBBufferManagerImpl*) myMgr.get())->initialize("tempDSFSD", 64 * 1024 * 1024, 16, "metadata", ".");

  /// 0. Create a computation graph

  // for allocations
  const UseTemporaryAllocationBlock tempBlock{1024 * 1024 * 128};

  Handle <Computation> readA = makeObject <MatrixScanner>("myData", "A");
  Handle <Computation> readB = makeObject <MatrixScanner>("myData", "B");
  Handle <Computation> join = makeObject <MatrixMultiplyJoin>();
  join->setInput(0, readA);
  join->setInput(1, readB);
  Handle<Computation> myAggregation = makeObject<MatrixMultiplyAggregation>();
  myAggregation->setInput(join);
  Handle<Computation> myWriter = makeObject<MatrixWriter>("myData", "C");
  myWriter->setInput(myAggregation);

  // here is the list of computations
  Vector<Handle<Computation>> computations;

  // put them in the list of computations
  computations.push_back(readA);
  computations.push_back(readB);
  computations.push_back(join);
  computations.push_back(myAggregation);
  computations.push_back(myWriter);

  /// 1. Get the TCAP

  // we need a vector
  std::vector<Handle<Computation>> queryGraph = { myWriter };

  // get the tcap from the query graph
  std::cout << "The original full plan : \n";
  std::string tcap = getTCAPString(queryGraph);
  cout << "\033[0;32m" << tcap <<"\033[0m\n";

  /// 2. Figure out the logical plan and the compute plan

  // make a logical plan
  auto logicalPlan = std::make_shared<LogicalPlan>(tcap, computations);

  // make the transformer
  auto transformer = std::make_shared<LogicalPlanTransformer>(logicalPlan);

  // add the transformation
  transformer->addTransformation(std::make_shared<InsertKeyScanSetsTransformation>("inputDataForSetScanner_0"));
  transformer->addTransformation(std::make_shared<InsertKeyScanSetsTransformation>("inputDataForSetScanner_1"));
  transformer->addTransformation(std::make_shared<JoinKeySideTransformation>("inputDataForSetScanner_0"));
  transformer->addTransformation(std::make_shared<JoinKeySideTransformation>("inputDataForSetScanner_1"));
  transformer->addTransformation(std::make_shared<JoinKeyTransformation>("OutForJoinedFor_equals_0JoinComp2"));
  transformer->addTransformation(std::make_shared<DropDependents>("aggOutForAggregationComp3"));
  transformer->addTransformation(std::make_shared<AggKeyTransformation>("OutFor_key_2AggregationComp3"));
  transformer->addTransformation(std::make_shared<AddJoinTID>("OutForJoinedFor_equals_0JoinComp2"));

  // apply all the transformations
  logicalPlan = transformer->applyTransformations();

  // modified TCAP
  std::cout << "The key only plan : \n";
  cout << "\033[1;32m" << *logicalPlan <<"\033[0m\n";

  // make the compute plan
  auto computePlan = std::make_shared<KeyComputePlan>(logicalPlan);

  /// 3. Make the page sets

  // the page set that is gonna provide stuff
  std::shared_ptr<MockPageSetReader> lhsReader = std::make_shared<MockPageSetReader>();

  // make the function return pages with Employee objects
  ON_CALL(*lhsReader, getNextPage(testing::An<size_t>())).WillByDefault(testing::Invoke(
      [&](size_t workerID) {

        // this implementation only serves six pages
        static int numPages = 0;
        if (numPages == 1)
          return (PDBPageHandle) nullptr;

        // create a page, loading it with random data
        auto page = myMgr->getPage();
        {
          const pdb::UseTemporaryAllocationBlock tempBlock{page->getBytes(), 64 * 1024};

          // write a bunch of matrix meta
          pdb::Handle<pdb::Vector<pdb::Handle<pdb::MatrixBlockMeta>>> matrixKeys = pdb::makeObject<pdb::Vector<pdb::Handle<pdb::MatrixBlockMeta>>>();

          // store the values
          for(uint32_t i = 0; i < 2; ++i) {
            for(uint32_t j = 0; j < 2; ++j) {
              matrixKeys->push_back(makeObject<pdb::MatrixBlockMeta>(i, j));
            }
          }

          // set the root object
          getRecord (matrixKeys);
        }
        numPages++;
        return page;
      }
  ));

  // it should call send object exactly six times
  EXPECT_CALL(*lhsReader, getNextPage(testing::An<size_t>())).Times(2);

  // the page set that is gonna provide stuff
  std::shared_ptr<MockPageSetWriter> lhsWriter = std::make_shared<MockPageSetWriter>(myMgr);

  std::unordered_map<uint64_t, PDBPageHandle> lhsWritePages;
  ON_CALL(*lhsWriter, getNewPage).WillByDefault(testing::Invoke(
      [&]() {

        // store the page
        auto page = myMgr->getPage();
        lhsWritePages[page->whichPage()] = page;

        return page;
      }));

  // it should call this method many times
  EXPECT_CALL(*lhsWriter, getNewPage).Times(testing::AtLeast(1));

  ON_CALL(*lhsWriter, removePage(testing::An<PDBPageHandle>())).WillByDefault(testing::Invoke(
      [&](PDBPageHandle pageHandle) {
        lhsWritePages.erase(pageHandle->whichPage());
      }));

  // it should call send object exactly six times
  EXPECT_CALL(*lhsWriter, removePage).Times(testing::Exactly(0));

  // make the function return pages
  std::unordered_map<uint64_t, PDBPageHandle>::iterator lhsIt;
  ON_CALL(*lhsWriter, getNextPage(testing::An<size_t>())).WillByDefault(testing::Invoke(
      [&](size_t workerID) {

        // we have a page if we do return it
        if(lhsIt != lhsWritePages.end()) {
          auto page = lhsIt->second;
          lhsIt++;
          return page;
        }

        // return the null ptr
        return (PDBPageHandle) nullptr;
      }
  ));

  // it should call send object exactly six times
  EXPECT_CALL(*lhsWriter, getNextPage).Times(testing::Exactly(2));

  // the page set that is gonna provide stuff
  std::shared_ptr<MockPageSetReader> rhsReader = std::make_shared<MockPageSetReader>();

  // make the function return pages with Employee objects
  ON_CALL(*rhsReader, getNextPage(testing::An<size_t>())).WillByDefault(testing::Invoke(
      [&](size_t workerID) {

        // this implementation only serves six pages
        static int numPages = 0;
        if (numPages == 1)
          return (PDBPageHandle) nullptr;

        // create a page, loading it with random data
        auto page = myMgr->getPage();
        {
          const pdb::UseTemporaryAllocationBlock tempBlock{page->getBytes(), 64 * 1024};

          // write a bunch of matrix meta
          pdb::Handle<pdb::Vector<pdb::Handle<pdb::MatrixBlockMeta>>> matrixKeys = pdb::makeObject<pdb::Vector<pdb::Handle<pdb::MatrixBlockMeta>>>();

          // store the values
          for(uint32_t i = 0; i < 2; ++i) {
            for(uint32_t j = 0; j < 2; ++j) {
              matrixKeys->push_back(makeObject<pdb::MatrixBlockMeta>(i, j));
            }
          }

          // set the root object
          getRecord (matrixKeys);
        }
        numPages++;
        return page;
      }
  ));

  // it should call send object exactly six times
  EXPECT_CALL(*rhsReader, getNextPage(testing::An<size_t>())).Times(2);

  // the page set that is gonna provide stuff
  std::shared_ptr<MockPageSetWriter> rhsWriter = std::make_shared<MockPageSetWriter>(myMgr);

  std::unordered_map<uint64_t, PDBPageHandle> rhsWritePages;
  ON_CALL(*rhsWriter, getNewPage).WillByDefault(testing::Invoke(
      [&]() {

        // store the page
        auto page = myMgr->getPage();
        rhsWritePages[page->whichPage()] = page;

        return page;
      }));

  // it should call this method many times
  EXPECT_CALL(*rhsWriter, getNewPage).Times(testing::AtLeast(1));

  ON_CALL(*rhsWriter, removePage(testing::An<PDBPageHandle>())).WillByDefault(testing::Invoke(
      [&](PDBPageHandle pageHandle) {
        lhsWritePages.erase(pageHandle->whichPage());
      }));

  // it should call send object exactly six times
  EXPECT_CALL(*rhsWriter, removePage).Times(testing::Exactly(0));

  // make the function return pages
  std::unordered_map<uint64_t, PDBPageHandle>::iterator rhsIt;
  ON_CALL(*rhsWriter, getNextPage(testing::An<size_t>())).WillByDefault(testing::Invoke(
      [&](size_t workerID) {

        // we have a page if we do return it
        if(rhsIt != rhsWritePages.end()) {
          auto page = rhsIt->second;
          rhsIt++;
          return page;
        }

        // return the null ptr
        return (PDBPageHandle) nullptr;
      }
  ));

  // it should call send object exactly six times
  EXPECT_CALL(*rhsWriter, getNextPage).Times(testing::Exactly(2));

  /// 4. Create the hashing pipelines

  // where we put the hashed records
  PDBPageQueuePtr lhsPageQueue;
  PDBPageQueuePtr rhsPageQueue;

  auto hashingPipelines = buildHashingPipeline(computePlan, myMgr, lhsReader, lhsWriter, rhsReader, rhsWriter);

  std::get<0>(hashingPipelines)->run();
  std::get<1>(hashingPipelines)->run();

  // 5. Run the last pipeline

  // the page set that is gonna provide stuff
  std::shared_ptr<MockPageSetWriter> finalWriter = std::make_shared<MockPageSetWriter>(myMgr);

  std::unordered_map<uint64_t, PDBPageHandle> finalWritePages;
  ON_CALL(*finalWriter, getNewPage).WillByDefault(testing::Invoke(
      [&]() {

        // store the page
        auto page = myMgr->getPage();
        finalWritePages[page->whichPage()] = page;

        return page;
      }));

  // it should call this method many times
  EXPECT_CALL(*finalWriter, getNewPage).Times(testing::AtLeast(1));

  ON_CALL(*finalWriter, removePage(testing::An<PDBPageHandle>())).WillByDefault(testing::Invoke(
      [&](PDBPageHandle pageHandle) {
        finalWritePages.erase(pageHandle->whichPage());
      }));

  // it should call send object exactly six times
  EXPECT_CALL(*finalWriter, removePage).Times(testing::Exactly(0));

  // set the lhs and rhs
  lhsIt = lhsWritePages.begin();
  rhsIt = rhsWritePages.begin();

  // get the left and right key page, basically a map of pdb::Map<Key, uint32_t>
  PDBPageHandle leftKeyPage = myMgr->getPage();
  PDBPageHandle rightKeyPage = myMgr->getPage();

  // run the final pipeline
  auto finalPipeline = buildJoinAggPipeline(computePlan, lhsWriter, rhsWriter, finalWriter, leftKeyPage, rightKeyPage);
  finalPipeline->run();
}


}
