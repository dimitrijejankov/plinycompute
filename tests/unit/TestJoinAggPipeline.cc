#include <gtest/gtest.h>
#include <gmock/gmock-generated-function-mockers.h>
#include <gmock/gmock-more-actions.h>
#include <UseTemporaryAllocationBlock.h>
#include <Computation.h>
#include <KeyComputePlan.h>
#include <AtomicComputationClasses.h>
#include <PDBBufferManagerImpl.h>

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
  //auto rightPipeline = computePlan->buildHashPipeline(rhsSource, rhsReader, rhsWriter, params);

  // return the pipelines
  return {leftPipeline, nullptr};
}

TEST(PipelineTest, TestJoinAggPipeline) {

  /// 1. Create the buffer manager that is going to provide the pages to the pipeline

  // create the buffer manager
  pdb::PDBBufferManagerImpl myMgr;
  myMgr.initialize("tempDSFSD", 64 * 1024 * 1024, 16, "metadata", ".");

  /// 0. Create a computation graph and the TCAP

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
  //computations.push_back(myWriter);

  std::string tcap = "inputDataForSetScanner_0(in0) <= SCAN ('myData', 'A', 'SetScanner_0')\n"
                     "inputDataForSetScanner_1(in1) <= SCAN ('myData', 'B', 'SetScanner_1')\n"
                     "OutFor_attAccess_1JoinComp2(in0,OutFor_attAccess_1_2) <= APPLY (inputDataForSetScanner_0(in0), inputDataForSetScanner_0(in0), 'JoinComp_2', 'attAccess_1', [('attName', 'colID'), ('attTypeName', 'unsignedint'), ('inputTypeName', 'pdb::matrix::MatrixBlockMeta'), ('lambdaType', 'attAccess')])\n"
                     "OutFor_attAccess_2JoinComp2(in1,OutFor_attAccess_2_2) <= APPLY (inputDataForSetScanner_1(in1), inputDataForSetScanner_1(in1), 'JoinComp_2', 'attAccess_2', [('attName', 'rowID'), ('attTypeName', 'unsignedint'), ('inputTypeName', 'pdb::matrix::MatrixBlockMeta'), ('lambdaType', 'attAccess')])\n"
                     "OutFor_attAccess_1JoinComp2_hashed(in0,OutFor_attAccess_1_2_hash) <= HASHLEFT (OutFor_attAccess_1JoinComp2(OutFor_attAccess_1_2), OutFor_attAccess_1JoinComp2(in0), 'JoinComp_2', '==_0', [])\n"
                     "OutFor_attAccess_2JoinComp2_hashed(in1,OutFor_attAccess_2_2_hash) <= HASHRIGHT (OutFor_attAccess_2JoinComp2(OutFor_attAccess_2_2), OutFor_attAccess_2JoinComp2(in1), 'JoinComp_2', '==_0', [])\n"
                     "OutForJoinedFor_equals_0JoinComp2(in0,in1) <= JOIN (OutFor_attAccess_1JoinComp2_hashed(OutFor_attAccess_1_2_hash), OutFor_attAccess_1JoinComp2_hashed(in0), OutFor_attAccess_2JoinComp2_hashed(OutFor_attAccess_2_2_hash), OutFor_attAccess_2JoinComp2_hashed(in1), 'JoinComp_2')\n"
                     "LExtractedFor0_attAccess_1JoinComp2(in0,in1,LExtractedFor0_attAccess_1_2) <= APPLY (OutForJoinedFor_equals_0JoinComp2(in0), OutForJoinedFor_equals_0JoinComp2(in0,in1), 'JoinComp_2', 'attAccess_1', [('attName', 'colID'), ('attTypeName', 'unsignedint'), ('inputTypeName', 'pdb::matrix::MatrixBlockMeta'), ('lambdaType', 'attAccess')])\n"
                     "RExtractedFor0_attAccess_2JoinComp2(in0,in1,LExtractedFor0_attAccess_1_2,RExtractedFor0_attAccess_2_2) <= APPLY (LExtractedFor0_attAccess_1JoinComp2(in1), LExtractedFor0_attAccess_1JoinComp2(in0,in1,LExtractedFor0_attAccess_1_2), 'JoinComp_2', 'attAccess_2', [('attName', 'rowID'), ('attTypeName', 'unsignedint'), ('inputTypeName', 'pdb::matrix::MatrixBlockMeta'), ('lambdaType', 'attAccess')])\n"
                     "OutFor_OutForJoinedFor_equals_0JoinComp2_BOOL(in0,in1,bool_0_2) <= APPLY (RExtractedFor0_attAccess_2JoinComp2(LExtractedFor0_attAccess_1_2,RExtractedFor0_attAccess_2_2), RExtractedFor0_attAccess_2JoinComp2(in0,in1), 'JoinComp_2', '==_0', [('lambdaType', '==')])\n"
                     "OutFor_OutForJoinedFor_equals_0JoinComp2_FILTERED(in0,in1) <= FILTER (OutFor_OutForJoinedFor_equals_0JoinComp2_BOOL(bool_0_2), OutFor_OutForJoinedFor_equals_0JoinComp2_BOOL(in0,in1), 'JoinComp_2')\n"
                     "OutFor_native_lambda_3JoinComp2(OutFor_native_lambda_3_2) <= APPLY (OutFor_OutForJoinedFor_equals_0JoinComp2_FILTERED(in0,in1), OutFor_OutForJoinedFor_equals_0JoinComp2_FILTERED(), 'JoinComp_2', 'native_lambda_3', [('lambdaType', 'native_lambda')])\n"
                     "OutFor_self_1AggregationComp3(OutFor_native_lambda_3_2,OutFor_self_1_3) <= APPLY (OutFor_native_lambda_3JoinComp2(OutFor_native_lambda_3_2), OutFor_native_lambda_3JoinComp2(OutFor_native_lambda_3_2), 'AggregationComp_3', 'self_1', [('lambdaType', 'self')])\n"
                     "OutFor_deref_0AggregationComp3(OutFor_deref_0_3) <= APPLY (OutFor_self_1AggregationComp3(OutFor_self_1_3), OutFor_self_1AggregationComp3(), 'AggregationComp_3', 'deref_0', [('lambdaType', 'deref')])";

  /// 1. Create the compute plan

  //
  auto logicalPlan = std::make_shared<LogicalPlan>(tcap, computations, true);

  //
  auto computePlan = std::make_shared<KeyComputePlan>(logicalPlan);

  /// 2.

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

  std::shared_ptr<MockPageSetReader> rhsReader = std::make_shared<MockPageSetReader>();
  std::shared_ptr<MockPageSetWriter> rhsWriter = std::make_shared<MockPageSetWriter>();

  /// 3. Create the hashing pipelines

  auto hashingPipelines = buildHashingPipeline(computePlan, lhsReader, lhsWriter, rhsReader, rhsWriter);

  std::get<0>(hashingPipelines)->run();

}


}
