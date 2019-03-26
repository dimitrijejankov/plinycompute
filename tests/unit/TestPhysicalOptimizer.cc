#include <PDBPhysicalOptimizer.h>
#include <PDBAggregationPipeAlgorithm.h>
#include <PDBStraightPipeAlgorithm.h>

#include <iostream>
#include <boost/filesystem/operations.hpp>
#include <gtest/gtest.h>
#include <gmock/gmock.h>

namespace pdb {

class MockCatalog {
public:

  MOCK_METHOD3(getSet, pdb::PDBCatalogSetPtr(const std::string &, const std::string &, std::string &));
};

TEST(TestPhysicalOptimizer, TestSelection) {

  // 1MB for algorithm and stuff
  const pdb::UseTemporaryAllocationBlock tempBlock{1024 * 1024};

  // setup the input parameters
  uint64_t compID = 98;
  String myTCAPString =
      "inputDataForScanSet_0(in0) <= SCAN ('input_set', 'by8_db', 'SetScanner_0') \n"\
      "nativ_0OutForSelectionComp1(in0,nativ_0_1OutFor) <= APPLY (inputDataForScanSet_0(in0), inputDataForScanSet_0(in0), 'SelectionComp_1', 'native_lambda_0', [('lambdaType', 'native_lambda')]) \n"\
      "filteredInputForSelectionComp1(in0) <= FILTER (nativ_0OutForSelectionComp1(nativ_0_1OutFor), nativ_0OutForSelectionComp1(in0), 'SelectionComp_1') \n"\
      "nativ_1OutForSelectionComp1 (nativ_1_1OutFor) <= APPLY (filteredInputForSelectionComp1(in0), filteredInputForSelectionComp1(), 'SelectionComp_1', 'native_lambda_1', [('lambdaType', 'native_lambda')]) \n"\
      "nativ_1OutForSelectionComp1_out() <= OUTPUT ( nativ_1OutForSelectionComp1 ( nativ_1_1OutFor ), 'output_set', 'by8_db', 'SetWriter_2') \n";

  // make a logger
  auto logger = make_shared<pdb::PDBLogger>("log.out");

  // make the mock client
  auto catalogClient = std::make_shared<MockCatalog>();
  ON_CALL(*catalogClient, getSet(testing::An<const std::string &>(), testing::An<const std::string &>(), testing::An<std::string &>())).WillByDefault(testing::Invoke(
      [&](const std::string &, const std::string &, std::string &errMsg) {
        return std::make_shared<pdb::PDBCatalogSet>("input_set", "by8_db", "Nothing");
      }));

  EXPECT_CALL(*catalogClient, getSet).Times(testing::Exactly(1));

  // init the optimizer
  pdb::PDBPhysicalOptimizer optimizer(compID, myTCAPString, catalogClient, logger);

  // we should have one source so we should be able to generate an algorithm
  EXPECT_TRUE(optimizer.hasAlgorithmToRun());

  // the algorithm should be a PDBAggregationPipeAlgorithm
  auto algorithm = optimizer.getNextAlgorithm();

  // cast the algorithm
  Handle<pdb::PDBStraightPipeAlgorithm> strAlgorithm = unsafeCast<pdb::PDBStraightPipeAlgorithm>(algorithm);

  // check the source
  EXPECT_EQ(strAlgorithm->source->sourceType, SetScanSource);
  EXPECT_EQ((std::string) strAlgorithm->firstTupleSet, std::string("inputDataForScanSet_0"));
  EXPECT_EQ((std::string) strAlgorithm->source->pageSetIdentifier.second, std::string("inputDataForScanSet_0"));
  EXPECT_EQ(strAlgorithm->source->pageSetIdentifier.first, compID);

  // check the sink
  EXPECT_EQ(strAlgorithm->sink->sinkType, SetSink);
  EXPECT_EQ((std::string) strAlgorithm->finalTupleSet, "nativ_1OutForSelectionComp1_out");
  EXPECT_EQ((std::string) strAlgorithm->sink->pageSetIdentifier.second, "nativ_1OutForSelectionComp1_out");
  EXPECT_EQ(strAlgorithm->sink->pageSetIdentifier.first, compID);

  // we should be done here
  EXPECT_FALSE(optimizer.hasAlgorithmToRun());
}


TEST(TestPhysicalOptimizer, TestAggregation) {

  // 1MB for algorithm and stuff
  const pdb::UseTemporaryAllocationBlock tempBlock{1024 * 1024};

  // setup the input parameters
  uint64_t compID = 99;
  pdb::String tcapString =
  "inputData (in) <= SCAN ('input_set', 'by8_db', 'SetScanner_0', []) \n"
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
  "nothing () <= OUTPUT (final (result), 'outSet', 'myDB', 'SetWriter_4', [])";


  // make a logger
  auto logger = make_shared<pdb::PDBLogger>("log.out");

  // make the mock client
  auto catalogClient = std::make_shared<MockCatalog>();
  ON_CALL(*catalogClient, getSet(testing::An<const std::string &>(), testing::An<const std::string &>(), testing::An<std::string &>())).WillByDefault(testing::Invoke(
    [&](const std::string &, const std::string &, std::string &errMsg) {
      return std::make_shared<pdb::PDBCatalogSet>("input_set", "by8_db", "Nothing");
    }));

  EXPECT_CALL(*catalogClient, getSet).Times(testing::Exactly(1));

  // init the optimizer
  pdb::PDBPhysicalOptimizer optimizer(compID, tcapString, catalogClient, logger);

  // we should have one source so we should be able to generate an algorithm
  EXPECT_TRUE(optimizer.hasAlgorithmToRun());

  // the first algorithm should be a PDBAggregationPipeAlgorithm
  auto algorithm1 = optimizer.getNextAlgorithm();

  // cast the algorithm
  Handle<pdb::PDBAggregationPipeAlgorithm> aggAlgorithm = unsafeCast<pdb::PDBAggregationPipeAlgorithm>(algorithm1);

  // check the source
  EXPECT_EQ(aggAlgorithm->source->sourceType, SetScanSource);
  EXPECT_EQ((std::string) aggAlgorithm->firstTupleSet, std::string("inputData"));
  EXPECT_EQ((std::string) aggAlgorithm->source->pageSetIdentifier.second, std::string("inputData"));
  EXPECT_EQ(aggAlgorithm->source->pageSetIdentifier.first, compID);

  // check the sink that we are
  EXPECT_EQ(aggAlgorithm->hashedToSend->sinkType, AggShuffleSink);
  EXPECT_EQ((std::string) aggAlgorithm->hashedToSend->pageSetIdentifier.second, "aggWithValue_hashed_to_send");
  EXPECT_EQ(aggAlgorithm->hashedToSend->pageSetIdentifier.first, compID);

  // check the source
  EXPECT_EQ(aggAlgorithm->hashedToRecv->sourceType, ShuffledAggregatesSource);
  EXPECT_EQ((std::string) aggAlgorithm->hashedToRecv->pageSetIdentifier.second, std::string("aggWithValue_hashed_to_recv"));
  EXPECT_EQ(aggAlgorithm->hashedToRecv->pageSetIdentifier.first, compID);

  // check the sink
  EXPECT_EQ(aggAlgorithm->sink->sinkType, AggregationSink);
  EXPECT_EQ((std::string) aggAlgorithm->finalTupleSet, "aggWithValue");
  EXPECT_EQ((std::string) aggAlgorithm->sink->pageSetIdentifier.second, "aggWithValue");
  EXPECT_EQ(aggAlgorithm->sink->pageSetIdentifier.first, compID);

  // we should have another source that reads the aggregation so we can generate another algorithm
  EXPECT_TRUE(optimizer.hasAlgorithmToRun());

  // the second algorithm should be a PDBStraightPipeAlgorithm
  auto algorithm2 = optimizer.getNextAlgorithm();

  // cast the algorithm
  Handle<pdb::PDBStraightPipeAlgorithm> strAlgorithm = unsafeCast<pdb::PDBStraightPipeAlgorithm>(algorithm2);

  // check the source
  EXPECT_EQ(strAlgorithm->source->sourceType, AggregationSource);
  EXPECT_EQ((std::string) strAlgorithm->firstTupleSet, std::string("agg"));
  EXPECT_EQ((std::string) strAlgorithm->source->pageSetIdentifier.second, std::string("aggWithValue"));
  EXPECT_EQ(strAlgorithm->source->pageSetIdentifier.first, compID);

  // check the sink
  EXPECT_EQ(strAlgorithm->sink->sinkType, SetSink);
  EXPECT_EQ((std::string) strAlgorithm->finalTupleSet, "nothing");
  EXPECT_EQ((std::string) strAlgorithm->sink->pageSetIdentifier.second, "nothing");
  EXPECT_EQ(strAlgorithm->sink->pageSetIdentifier.first, compID);

  // we should be done
  EXPECT_FALSE(optimizer.hasAlgorithmToRun());
}

}
