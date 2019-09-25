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
};

class MockPageSetWriter: public pdb::PDBAnonymousPageSet {
 public:

  MOCK_METHOD1(getNextPage, PDBPageHandle(size_t workerID));

  MOCK_METHOD0(getNewPage, PDBPageHandle());

  MOCK_METHOD1(removePage, void(PDBPageHandle pageHandle));

  MOCK_METHOD0(getNumPages, size_t ());
};

std::tuple<pdb::PipelinePtr, pdb::PipelinePtr> buildHashingPipeline(std::shared_ptr<KeyComputePlan> &computePlan,
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
  auto lhsSource = sources.front();

  // make the parameters for the first set
  std::map<ComputeInfoType, ComputeInfoPtr> params = {{ ComputeInfoType::SOURCE_SET_INFO, std::make_shared<pdb::SourceSetArg>(std::make_shared<PDBCatalogSet>(((ScanSet*)lhsSource.get())->getDBName(), ((ScanSet*)lhsSource.get())->getSetName(), "", 0, PDB_CATALOG_SET_VECTOR_CONTAINER)) }};

  // build the pipeline
  auto leftPipeline = computePlan->buildHashPipeline(lhsSource, lhsReader, lhsWriter, params);

  // the get the right source comp
  auto rhsSource = sources.back();

  // init the parameters for rhs
  params = {{ ComputeInfoType::SOURCE_SET_INFO, std::make_shared<pdb::SourceSetArg>(std::make_shared<PDBCatalogSet>(((ScanSet*)rhsSource.get())->getDBName(), ((ScanSet*)rhsSource.get())->getSetName(), "", 0, PDB_CATALOG_SET_VECTOR_CONTAINER)) }};

  // build the pipeline
  auto rightPipeline = computePlan->buildHashPipeline(rhsSource, rhsReader, rhsWriter, params);

  // return the pipelines
  return {leftPipeline, rightPipeline};
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
  pdb::PDBBufferManagerImpl myMgr;
  myMgr.initialize("tempDSFSD", 64 * 1024 * 1024, 16, "metadata", ".");

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
  transformer->addTransformation(std::make_shared<AggKeyTransformation>("OutFor_key_2AggregationComp3"));
  transformer->addTransformation(std::make_shared<DropDependents>("aggOutForAggregationComp3_out"));

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
        auto page = myMgr.getPage();
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
  std::shared_ptr<MockPageSetWriter> lhsWriter = std::make_shared<MockPageSetWriter>();

  std::unordered_map<uint64_t, PDBPageHandle> lhsWritePages;
  ON_CALL(*lhsWriter, getNewPage).WillByDefault(testing::Invoke(
      [&]() {

        // store the page
        auto page = myMgr.getPage();
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
        auto page = myMgr.getPage();
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
  std::shared_ptr<MockPageSetWriter> rhsWriter = std::make_shared<MockPageSetWriter>();

  std::unordered_map<uint64_t, PDBPageHandle> rhsWritePages;
  ON_CALL(*rhsWriter, getNewPage).WillByDefault(testing::Invoke(
      [&]() {

        // store the page
        auto page = myMgr.getPage();
        lhsWritePages[page->whichPage()] = page;

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

  /// 4. Create the hashing pipelines

  auto hashingPipelines = buildHashingPipeline(computePlan, lhsReader, lhsWriter, rhsReader, rhsWriter);

  std::get<0>(hashingPipelines)->run();
  std::get<1>(hashingPipelines)->run();
}


}
