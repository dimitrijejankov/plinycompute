#include <PDBPhysicalOptimizer.h>
#include <PDBAggregationPipeAlgorithm.h>
#include <PDBStraightPipeAlgorithm.h>
#include <PDBBroadcastForJoinAlgorithm.h>
#include <PDBShuffleForJoinAlgorithm.h>

#include <iostream>
#include <boost/filesystem/operations.hpp>
#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <physicalAlgorithms/PDBBroadcastForJoinAlgorithm.h>

namespace pdb {

class MockCatalog {
 public:

  MOCK_METHOD3(getSet, pdb::PDBCatalogSetPtr(
      const std::string &, const std::string &, std::string &));
};

auto getPageSetsToRemove(pdb::PDBPhysicalOptimizer &optimizer) {
  auto pageSetsToRemove = std::set<PDBPageSetIdentifier, PageSetIdentifierComparator>();
  auto tmp = optimizer.getPageSetsToRemove();
  std::copy(tmp.begin(), tmp.end(), std::inserter(pageSetsToRemove, pageSetsToRemove.end()));
  return std::move(pageSetsToRemove);
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
  ON_CALL(*catalogClient,
          getSet(testing::An<const std::string &>(),
                 testing::An<const std::string &>(),
                 testing::An<std::string &>())).WillByDefault(testing::Invoke(
      [&](const std::string &, const std::string &, std::string &errMsg) {
        return std::make_shared<pdb::PDBCatalogSet>("input_set", "by8_db", "Nothing", 10);
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

  // check the sourceEXPECT_EQ(aggAlgorithm->source->sourceType, SetScanSource);
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

  // get the page sets we want to remove
  auto pageSetsToRemove = getPageSetsToRemove(optimizer);
  EXPECT_TRUE(pageSetsToRemove.find(std::make_pair(compID, "aggWithValue_hashed_to_send")) != pageSetsToRemove.end());
  EXPECT_TRUE(pageSetsToRemove.find(std::make_pair(compID, "aggWithValue_hashed_to_recv")) != pageSetsToRemove.end());
  EXPECT_EQ(pageSetsToRemove.size(), 2);

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

  // get the page sets we want to remove
  pageSetsToRemove = getPageSetsToRemove(optimizer);
  EXPECT_TRUE(pageSetsToRemove.find(std::make_pair(compID, "aggWithValue")) != pageSetsToRemove.end());
  EXPECT_TRUE(pageSetsToRemove.find(std::make_pair(compID, "nothing")) != pageSetsToRemove.end());
  EXPECT_EQ(pageSetsToRemove.size(), 2);

  // we should be done
  EXPECT_FALSE(optimizer.hasAlgorithmToRun());
}

TEST(TestPhysicalOptimizer, TestJoin1) {

  // 1MB for algorithm and stuff
  const pdb::UseTemporaryAllocationBlock tempBlock{1024 * 1024};

  // setup the input parameters
  uint64_t compID = 99;
  pdb::String tcapString =
      "A(a) <= SCAN ('myData', 'mySetA', 'SetScanner_0')\n"
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

  // make a logger
  auto logger = make_shared<pdb::PDBLogger>("log.out");

  // make the mock client
  auto catalogClient = std::make_shared<MockCatalog>();
  ON_CALL(*catalogClient,
          getSet(testing::An<const std::string &>(),
                 testing::An<const std::string &>(),
                 testing::An<std::string &>())).WillByDefault(testing::Invoke(
      [&](const std::string &dbName, const std::string &setName, std::string &errMsg) {
        if (setName == "mySetA") {
          auto tmp = std::make_shared<pdb::PDBCatalogSet>("mySetA", "myData", "Nothing", 1000);
          return tmp;
        } else {
          auto tmp = std::make_shared<pdb::PDBCatalogSet>("mySetB", "myData", "Nothing", 2000);
          return tmp;
        }
      }));

  EXPECT_CALL(*catalogClient, getSet).Times(testing::Exactly(2));

  // init the optimizer
  pdb::PDBPhysicalOptimizer optimizer(compID, tcapString, catalogClient, logger);

  // we should have two sources so we should be able to generate an algorithm
  EXPECT_TRUE(optimizer.hasAlgorithmToRun());

  // the first algorithm should be a PDBBroadcastForJoinAlgorithm
  Handle<pdb::PDBBroadcastForJoinAlgorithm>
      algorithmBroadcastA = unsafeCast<pdb::PDBBroadcastForJoinAlgorithm>(optimizer.getNextAlgorithm());

  // check the source
  EXPECT_EQ(algorithmBroadcastA->source->sourceType, SetScanSource);
  EXPECT_EQ((std::string) algorithmBroadcastA->firstTupleSet, std::string("A"));
  EXPECT_EQ((std::string) algorithmBroadcastA->source->pageSetIdentifier.second, std::string("A"));
  EXPECT_EQ(algorithmBroadcastA->source->pageSetIdentifier.first, compID);

  // check the intermediate set
  EXPECT_EQ(algorithmBroadcastA->intermediate->sinkType, BroadcastIntermediateJoinSink);
  EXPECT_EQ((std::string) algorithmBroadcastA->finalTupleSet, "AHashed");
  EXPECT_EQ((std::string) algorithmBroadcastA->intermediate->pageSetIdentifier.second, "AHashed_to_broadcast");
  EXPECT_EQ(algorithmBroadcastA->intermediate->pageSetIdentifier.first, compID);

  // check the sink
  EXPECT_EQ(algorithmBroadcastA->sink->sinkType, BroadcastJoinSink);
  EXPECT_EQ((std::string) algorithmBroadcastA->finalTupleSet, "AHashed");
  EXPECT_EQ((std::string) algorithmBroadcastA->sink->pageSetIdentifier.second, "AHashed");
  EXPECT_EQ(algorithmBroadcastA->sink->pageSetIdentifier.first, compID);

  // we should have one source so we should be able to generate an algorithm
  EXPECT_TRUE(optimizer.hasAlgorithmToRun());

  // get the next algorithm
  auto algorithmPipelineThroughB = optimizer.getNextAlgorithm();

  // check the source
  EXPECT_EQ(algorithmPipelineThroughB->source->sourceType, SetScanSource);
  EXPECT_EQ((std::string) algorithmPipelineThroughB->firstTupleSet, std::string("B"));
  EXPECT_EQ((std::string) algorithmPipelineThroughB->source->pageSetIdentifier.second, std::string("B"));
  EXPECT_EQ(algorithmPipelineThroughB->source->pageSetIdentifier.first, compID);

  // check the sink
  EXPECT_EQ(algorithmPipelineThroughB->sink->sinkType, SetSink);
  EXPECT_EQ((std::string) algorithmPipelineThroughB->finalTupleSet, "out");
  EXPECT_EQ((std::string) algorithmPipelineThroughB->sink->pageSetIdentifier.second, "out");
  EXPECT_EQ(algorithmPipelineThroughB->sink->pageSetIdentifier.first, compID);

  // check how many secondary sources we have
  pdb::Vector<pdb::Handle<PDBSourcePageSetSpec>> &additionalSources = *algorithmPipelineThroughB->secondarySources;
  EXPECT_EQ(additionalSources.size(), 1);

  EXPECT_EQ(additionalSources[0]->sourceType, BroadcastJoinSource);
  EXPECT_EQ(additionalSources[0]->pageSetIdentifier.first, compID);
  EXPECT_EQ((std::string) additionalSources[0]->pageSetIdentifier.second, "AHashed");

  // we should be done
  EXPECT_FALSE(optimizer.hasAlgorithmToRun());
}

TEST(TestPhysicalOptimizer, TestJoin2) {

  // 1MB for algorithm and stuff
  const pdb::UseTemporaryAllocationBlock tempBlock{1024 * 1024};

  // setup the input parameters
  uint64_t compID = 99;
  pdb::String tcapString =
      "A(a) <= SCAN ('myData', 'mySetA', 'SetScanner_0')\n"
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

  // make a logger
  auto logger = make_shared<pdb::PDBLogger>("log.out");

  // make the mock client
  auto catalogClient = std::make_shared<MockCatalog>();
  ON_CALL(*catalogClient,
          getSet(testing::An<const std::string &>(),
                 testing::An<const std::string &>(),
                 testing::An<std::string &>())).WillByDefault(testing::Invoke(
      [&](const std::string &dbName, const std::string &setName, std::string &errMsg) {
        if (setName == "mySetA") {
          auto tmp =
              std::make_shared<pdb::PDBCatalogSet>("mySetA", "myData", "Nothing", std::numeric_limits<size_t>::max());
          return tmp;
        } else {
          auto tmp = std::make_shared<pdb::PDBCatalogSet>("mySetB",
                                                          "myData",
                                                          "Nothing",
                                                          std::numeric_limits<size_t>::max() - 1);
          return tmp;
        }
      }));

  EXPECT_CALL(*catalogClient, getSet).Times(testing::Exactly(2));

  // init the optimizer
  pdb::PDBPhysicalOptimizer optimizer(compID, tcapString, catalogClient, logger);

  EXPECT_TRUE(optimizer.hasAlgorithmToRun());

  Handle<pdb::PDBShuffleForJoinAlgorithm> shuffleB = unsafeCast<pdb::PDBShuffleForJoinAlgorithm>(optimizer.getNextAlgorithm());

  // check the source
  EXPECT_EQ(shuffleB->source->sourceType, SetScanSource);
  EXPECT_EQ((std::string) shuffleB->firstTupleSet, std::string("B"));
  EXPECT_EQ((std::string) shuffleB->source->pageSetIdentifier.second, std::string("B"));
  EXPECT_EQ(shuffleB->source->pageSetIdentifier.first, compID);

  // check the intermediate set
  EXPECT_EQ(shuffleB->intermediate->sinkType, JoinShuffleIntermediateSink);
  EXPECT_EQ((std::string) shuffleB->intermediate->pageSetIdentifier.second, "BHashedOnA_to_shuffle");
  EXPECT_EQ(shuffleB->intermediate->pageSetIdentifier.first, compID);

  // check the sink
  EXPECT_EQ(shuffleB->sink->sinkType, JoinShuffleSink);
  EXPECT_EQ((std::string) shuffleB->finalTupleSet, "BHashedOnA");
  EXPECT_EQ((std::string) shuffleB->sink->pageSetIdentifier.second, "BHashedOnA");
  EXPECT_EQ(shuffleB->sink->pageSetIdentifier.first, compID);

  // get the page sets we want to remove
  auto pageSetsToRemove = optimizer.getPageSetsToRemove();
  EXPECT_EQ(pageSetsToRemove[0].first, compID);
  EXPECT_EQ(pageSetsToRemove[0].second, "BHashedOnA_to_shuffle");
  EXPECT_EQ(pageSetsToRemove.size(), 1);

  // we should have another algorithm now for side A
  EXPECT_TRUE(optimizer.hasAlgorithmToRun());

  Handle<pdb::PDBShuffleForJoinAlgorithm> shuffleA = unsafeCast<pdb::PDBShuffleForJoinAlgorithm>(optimizer.getNextAlgorithm());

  // check the source
  EXPECT_EQ(shuffleA->source->sourceType, SetScanSource);
  EXPECT_EQ((std::string) shuffleA->firstTupleSet, "A");
  EXPECT_EQ((std::string) shuffleA->source->pageSetIdentifier.second, "A");
  EXPECT_EQ(shuffleA->source->pageSetIdentifier.first, compID);

  // check the intermediate set
  EXPECT_EQ(shuffleA->intermediate->sinkType, JoinShuffleIntermediateSink);
  EXPECT_EQ((std::string) shuffleA->intermediate->pageSetIdentifier.second, "AHashed_to_shuffle");
  EXPECT_EQ(shuffleA->intermediate->pageSetIdentifier.first, compID);

  // check the sink
  EXPECT_EQ(shuffleB->sink->sinkType, JoinShuffleSink);
  EXPECT_EQ((std::string) shuffleA->finalTupleSet, "AHashed");
  EXPECT_EQ((std::string) shuffleA->sink->pageSetIdentifier.second, "AHashed");
  EXPECT_EQ(shuffleA->sink->pageSetIdentifier.first, compID);

  // get the page sets we want to remove
  pageSetsToRemove = optimizer.getPageSetsToRemove();
  EXPECT_EQ(pageSetsToRemove[0].first, compID);
  EXPECT_EQ(pageSetsToRemove[0].second, "AHashed_to_shuffle");
  EXPECT_EQ(pageSetsToRemove.size(), 1);

  EXPECT_TRUE(optimizer.hasAlgorithmToRun());

  Handle<pdb::PDBStraightPipeAlgorithm> doJoin = unsafeCast<pdb::PDBStraightPipeAlgorithm>(optimizer.getNextAlgorithm());

  // check the source
  EXPECT_EQ(doJoin->source->sourceType, ShuffledJoinTuplesSource);
  EXPECT_EQ((std::string) doJoin->firstTupleSet, "AandBJoined");
  EXPECT_EQ((std::string) doJoin->source->pageSetIdentifier.second, "BHashedOnA");
  EXPECT_EQ(doJoin->source->pageSetIdentifier.first, compID);

  // check the sink
  EXPECT_EQ(doJoin->sink->sinkType, SetSink);
  EXPECT_EQ((std::string) doJoin->finalTupleSet, "out");
  EXPECT_EQ((std::string) doJoin->sink->pageSetIdentifier.second, "out");
  EXPECT_EQ(doJoin->sink->pageSetIdentifier.first, compID);

  // get the page sets we want to remove
  pageSetsToRemove = optimizer.getPageSetsToRemove();

  // left side of the join should be removed
  EXPECT_EQ(pageSetsToRemove[0].first, compID);
  EXPECT_EQ(pageSetsToRemove[0].second, "AHashed");

  // right side of the join should be removed
  EXPECT_EQ(pageSetsToRemove[1].first, compID);
  EXPECT_EQ(pageSetsToRemove[1].second, "BHashedOnA");

  // remove the output since it is materialized
  EXPECT_EQ(pageSetsToRemove[2].first, compID);
  EXPECT_EQ(pageSetsToRemove[2].second, "out");

  // there should be exactly two sets to remove
  EXPECT_EQ(pageSetsToRemove.size(), 3);

  // check how many secondary sources we have
  pdb::Vector<pdb::Handle<PDBSourcePageSetSpec>> &additionalSources = *doJoin->secondarySources;

  // we should have only one
  EXPECT_EQ(additionalSources.size(), 1);

  // check it
  EXPECT_EQ(additionalSources[0]->sourceType, ShuffledJoinTuplesSource);
  EXPECT_EQ(additionalSources[0]->pageSetIdentifier.first, compID);
  EXPECT_EQ((std::string) additionalSources[0]->pageSetIdentifier.second, "AHashed");

  EXPECT_FALSE(optimizer.hasAlgorithmToRun());
}

TEST(TestPhysicalOptimizer, TestJoin3) {

  // 1MB for algorithm and stuff
  const pdb::UseTemporaryAllocationBlock tempBlock{1024 * 1024};

  // setup the input parameters
  uint64_t compID = 89;
  pdb::String tcapString =
      "/* scan the three inputs */ \n"
      "A (a) <= SCAN ('myData', 'mySetA', 'SetScanner_0', []) \n"
      "B (aAndC) <= SCAN ('myData', 'mySetB', 'SetScanner_1', []) \n"
      "C (c) <= SCAN ('myData', 'mySetC', 'SetScanner_2', []) \n"
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

  // make a logger
  auto logger = make_shared<pdb::PDBLogger>("log.out");

  // make the mock client
  auto catalogClient = std::make_shared<MockCatalog>();
  ON_CALL(*catalogClient,
          getSet(testing::An<const std::string &>(),
                 testing::An<const std::string &>(),
                 testing::An<std::string &>())).WillByDefault(testing::Invoke(
      [&](const std::string &dbName, const std::string &setName, std::string &errMsg) {
        if (setName == "mySetA") {
          auto tmp = std::make_shared<pdb::PDBCatalogSet>("mySetA",
                                                          "myData",
                                                          "Nothing",
                                                          std::numeric_limits<size_t>::max() - 1);
          return tmp;
        } else if (setName == "mySetC") {
          auto tmp = std::make_shared<pdb::PDBCatalogSet>("mySetC", "myData", "Nothing", 0);
          return tmp;
        } else {
          auto tmp = std::make_shared<pdb::PDBCatalogSet>("mySetB", "myData", "Nothing", std::numeric_limits<size_t>::max());
          return tmp;
        }
      }));

  EXPECT_CALL(*catalogClient, getSet).Times(testing::Exactly(3));

  // init the optimizer
  pdb::PDBPhysicalOptimizer optimizer(compID, tcapString, catalogClient, logger);

  EXPECT_TRUE(optimizer.hasAlgorithmToRun());

  // the first algorithm should be a PDBBroadcastForJoinAlgorithm
  Handle<pdb::PDBBroadcastForJoinAlgorithm> algorithmBroadcastC = unsafeCast<pdb::PDBBroadcastForJoinAlgorithm>(optimizer.getNextAlgorithm());

  // check the source
  EXPECT_EQ(algorithmBroadcastC->source->sourceType, SetScanSource);
  EXPECT_EQ((std::string) algorithmBroadcastC->firstTupleSet, std::string("C"));
  EXPECT_EQ((std::string) algorithmBroadcastC->source->pageSetIdentifier.second, std::string("C"));
  EXPECT_EQ(algorithmBroadcastC->source->pageSetIdentifier.first, compID);

  // check the intermediate set
  EXPECT_EQ(algorithmBroadcastC->intermediate->sinkType, BroadcastIntermediateJoinSink);
  EXPECT_EQ((std::string) algorithmBroadcastC->finalTupleSet, "CHashedOnC");
  EXPECT_EQ((std::string) algorithmBroadcastC->intermediate->pageSetIdentifier.second, "CHashedOnC_to_broadcast");
  EXPECT_EQ(algorithmBroadcastC->intermediate->pageSetIdentifier.first, compID);

  // check the sink
  EXPECT_EQ(algorithmBroadcastC->sink->sinkType, BroadcastJoinSink);
  EXPECT_EQ((std::string) algorithmBroadcastC->finalTupleSet, "CHashedOnC");
  EXPECT_EQ((std::string) algorithmBroadcastC->sink->pageSetIdentifier.second, "CHashedOnC");
  EXPECT_EQ(algorithmBroadcastC->sink->pageSetIdentifier.first, compID);

  EXPECT_TRUE(optimizer.hasAlgorithmToRun());

  Handle<pdb::PDBShuffleForJoinAlgorithm>
      shuffleA = unsafeCast<pdb::PDBShuffleForJoinAlgorithm>(optimizer.getNextAlgorithm());

  // check the source
  EXPECT_EQ(shuffleA->source->sourceType, SetScanSource);
  EXPECT_EQ((std::string) shuffleA->firstTupleSet, "A");
  EXPECT_EQ((std::string) shuffleA->source->pageSetIdentifier.second, "A");
  EXPECT_EQ(shuffleA->source->pageSetIdentifier.first, compID);

  // check the intermediate set
  EXPECT_EQ(shuffleA->intermediate->sinkType, JoinShuffleIntermediateSink);
  EXPECT_EQ((std::string) shuffleA->intermediate->pageSetIdentifier.second, "AHashed_to_shuffle");
  EXPECT_EQ(shuffleA->intermediate->pageSetIdentifier.first, compID);

  // check the sink
  EXPECT_EQ(shuffleA->sink->sinkType, JoinShuffleSink);
  EXPECT_EQ((std::string) shuffleA->finalTupleSet, "AHashed");
  EXPECT_EQ((std::string) shuffleA->sink->pageSetIdentifier.second, "AHashed");
  EXPECT_EQ(shuffleA->sink->pageSetIdentifier.first, compID);

  EXPECT_TRUE(optimizer.hasAlgorithmToRun());

  Handle<pdb::PDBShuffleForJoinAlgorithm> shuffleB = unsafeCast<pdb::PDBShuffleForJoinAlgorithm>(optimizer.getNextAlgorithm());

  // check the source
  EXPECT_EQ(shuffleB->source->sourceType, SetScanSource);
  EXPECT_EQ((std::string) shuffleB->firstTupleSet, std::string("B"));
  EXPECT_EQ((std::string) shuffleB->source->pageSetIdentifier.second, std::string("B"));
  EXPECT_EQ(shuffleB->source->pageSetIdentifier.first, compID);

  // check the intermediate set
  EXPECT_EQ(shuffleB->intermediate->sinkType, JoinShuffleIntermediateSink);
  EXPECT_EQ((std::string) shuffleB->intermediate->pageSetIdentifier.second, "BHashedOnA_to_shuffle");
  EXPECT_EQ(shuffleB->intermediate->pageSetIdentifier.first, compID);

  // check the sink
  EXPECT_EQ(shuffleB->sink->sinkType, JoinShuffleSink);
  EXPECT_EQ((std::string) shuffleB->finalTupleSet, "BHashedOnA");
  EXPECT_EQ((std::string) shuffleB->sink->pageSetIdentifier.second, "BHashedOnA");
  EXPECT_EQ(shuffleB->sink->pageSetIdentifier.first, compID);

  EXPECT_TRUE(optimizer.hasAlgorithmToRun());

  Handle<pdb::PDBStraightPipeAlgorithm> doJoin = unsafeCast<pdb::PDBStraightPipeAlgorithm>(optimizer.getNextAlgorithm());

  // check the source
  EXPECT_EQ(doJoin->source->sourceType, ShuffledJoinTuplesSource);
  EXPECT_EQ((std::string) doJoin->firstTupleSet, "AandBJoined");
  EXPECT_EQ((std::string) doJoin->source->pageSetIdentifier.second, "AHashed");
  EXPECT_EQ(doJoin->source->pageSetIdentifier.first, compID);

  // check the sink
  EXPECT_EQ(doJoin->sink->sinkType, SetSink);
  EXPECT_EQ((std::string) doJoin->finalTupleSet, "nothing");
  EXPECT_EQ((std::string) doJoin->sink->pageSetIdentifier.second, "nothing");
  EXPECT_EQ(doJoin->sink->pageSetIdentifier.first, compID);

  size_t cnt = 0;

  // check how many secondary sources we have
  pdb::Vector<pdb::Handle<PDBSourcePageSetSpec>> &additionalSources = *doJoin->secondarySources;
  for (int i = 0; i < 2; ++i) {

    // grab a the source
    pdb::Handle<PDBSourcePageSetSpec> &src = additionalSources[i];
    if (src->pageSetIdentifier.second == "CHashedOnC") {

      // check it
      EXPECT_EQ(additionalSources[i]->sourceType, BroadcastJoinSource);
      EXPECT_EQ(additionalSources[i]->pageSetIdentifier.first, compID);
      EXPECT_EQ((std::string) additionalSources[i]->pageSetIdentifier.second, "CHashedOnC");

      cnt++;
    } else if (src->pageSetIdentifier.second == "BHashedOnA") {

      // check it
      EXPECT_EQ(additionalSources[i]->sourceType, ShuffledJoinTuplesSource);
      EXPECT_EQ(additionalSources[i]->pageSetIdentifier.first, compID);
      EXPECT_EQ((std::string) additionalSources[i]->pageSetIdentifier.second, "BHashedOnA");

      cnt++;
    }
  }

  // we should have two additional sources
  EXPECT_EQ(additionalSources.size(), 2);
  EXPECT_EQ(cnt, 2);

  EXPECT_FALSE(optimizer.hasAlgorithmToRun());
}

TEST(TestPhysicalOptimizer, TestJoin4) {

  // 1MB for algorithm and stuff
  const pdb::UseTemporaryAllocationBlock tempBlock{1024 * 1024};

  // setup the input parameters
  uint64_t compID = 89;
  pdb::String tcapString =
      "/* scan the three inputs */ \n"
      "A (a) <= SCAN ('myData', 'mySetA', 'SetScanner_0', []) \n"
      "B (aAndC) <= SCAN ('myData', 'mySetB', 'SetScanner_1', []) \n"
      "C (c) <= SCAN ('myData', 'mySetC', 'SetScanner_2', []) \n"
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

  // make a logger
  auto logger = make_shared<pdb::PDBLogger>("log.out");

  // make the mock client
  auto catalogClient = std::make_shared<MockCatalog>();
  ON_CALL(*catalogClient,
          getSet(testing::An<const std::string &>(),
                 testing::An<const std::string &>(),
                 testing::An<std::string &>())).WillByDefault(testing::Invoke(
      [&](const std::string &dbName, const std::string &setName, std::string &errMsg) {
        if (setName == "mySetA") {
          auto tmp = std::make_shared<pdb::PDBCatalogSet>("mySetA",
                                                          "myData",
                                                          "Nothing",
                                                          std::numeric_limits<size_t>::max() - 1);
          return tmp;
        } else if (setName == "mySetC") {
          auto tmp = std::make_shared<pdb::PDBCatalogSet>("mySetC",
                                                          "myData",
                                                          "Nothing",
                                                          std::numeric_limits<size_t>::max() - 2);
          return tmp;
        } else {
          auto tmp = std::make_shared<pdb::PDBCatalogSet>("mySetB", "myData", "Nothing", std::numeric_limits<size_t>::max());
          return tmp;
        }
      }));

  EXPECT_CALL(*catalogClient, getSet).Times(testing::Exactly(3));

  // init the optimizer
  pdb::PDBPhysicalOptimizer optimizer(compID, tcapString, catalogClient, logger);


  /// TODO finish this
}

TEST(TestPhysicalOptimizer, TestMultiSink) {

  // 1MB for algorithm and stuff
  const pdb::UseTemporaryAllocationBlock tempBlock{1024 * 1024};

  // setup the parameters
  uint64_t compID = 55;
  pdb::String tcapString = "inputDataForSetScanner_0(in0) <= SCAN ('myData', 'mySetA', 'SetScanner_0')\n"
                           "methodCall_0OutFor_SelectionComp1(in0,methodCall_0OutFor__getSteve) <= APPLY (inputDataForSetScanner_0(in0), inputDataForSetScanner_0(in0), 'SelectionComp_1', 'methodCall_0', [('inputTypeName', 'pdb::Supervisor'), ('lambdaType', 'methodCall'), ('methodName', 'getSteve'), ('returnTypeName', 'pdb::Supervisor')])\n"
                           "attAccess_1OutForSelectionComp1(in0,methodCall_0OutFor__getSteve,att_1OutFor_me) <= APPLY (methodCall_0OutFor_SelectionComp1(in0), methodCall_0OutFor_SelectionComp1(in0,methodCall_0OutFor__getSteve), 'SelectionComp_1', 'attAccess_1', [('attName', 'me'), ('attTypeName', 'pdb::Handle&lt;pdb::Employee&gt;'), ('inputTypeName', 'pdb::Supervisor'), ('lambdaType', 'attAccess')])\n"
                           "equals_2OutForSelectionComp1(in0,methodCall_0OutFor__getSteve,att_1OutFor_me,bool_2_1) <= APPLY (attAccess_1OutForSelectionComp1(methodCall_0OutFor__getSteve,att_1OutFor_me), attAccess_1OutForSelectionComp1(in0,methodCall_0OutFor__getSteve,att_1OutFor_me), 'SelectionComp_1', '==_2', [('lambdaType', '==')])\n"
                           "filteredInputForSelectionComp1(in0) <= FILTER (equals_2OutForSelectionComp1(bool_2_1), equals_2OutForSelectionComp1(in0), 'SelectionComp_1')\n"
                           "methodCall_3OutFor_SelectionComp1(in0,methodCall_3OutFor__getMe) <= APPLY (filteredInputForSelectionComp1(in0), filteredInputForSelectionComp1(in0), 'SelectionComp_1', 'methodCall_3', [('inputTypeName', 'pdb::Supervisor'), ('lambdaType', 'methodCall'), ('methodName', 'getMe'), ('returnTypeName', 'pdb::Supervisor')])\n"
                           "deref_4OutForSelectionComp1 (methodCall_3OutFor__getMe) <= APPLY (methodCall_3OutFor_SelectionComp1(methodCall_3OutFor__getMe), methodCall_3OutFor_SelectionComp1(), 'SelectionComp_1', 'deref_4')\n"
                           "attAccess_0OutForAggregationComp2(methodCall_3OutFor__getMe,att_0OutFor_department) <= APPLY (deref_4OutForSelectionComp1(methodCall_3OutFor__getMe), deref_4OutForSelectionComp1(methodCall_3OutFor__getMe), 'AggregationComp_2', 'attAccess_0', [('attName', 'department'), ('attTypeName', 'pdb::String'), ('inputTypeName', 'pdb::Employee'), ('lambdaType', 'attAccess')])\n"
                           "deref_1OutForAggregationComp2(methodCall_3OutFor__getMe, att_0OutFor_department) <= APPLY (attAccess_0OutForAggregationComp2(att_0OutFor_department), attAccess_0OutForAggregationComp2(methodCall_3OutFor__getMe), 'AggregationComp_2', 'deref_1')\n"
                           "methodCall_2OutFor_AggregationComp2(att_0OutFor_department,methodCall_2OutFor__getSalary) <= APPLY (deref_1OutForAggregationComp2(methodCall_3OutFor__getMe), deref_1OutForAggregationComp2(att_0OutFor_department), 'AggregationComp_2', 'methodCall_2', [('inputTypeName', 'pdb::Employee'), ('lambdaType', 'methodCall'), ('methodName', 'getSalary'), ('returnTypeName', 'pdb::Employee')])\n"
                           "aggOutForAggregationComp2 (aggOutFor2)<= AGGREGATE (methodCall_2OutFor_AggregationComp2(att_0OutFor_department, methodCall_2OutFor__getSalary),'AggregationComp_2')\n"
                           "methodCall_0OutFor_SelectionComp3(aggOutFor2,methodCall_0OutFor__checkSales) <= APPLY (aggOutForAggregationComp2(aggOutFor2), aggOutForAggregationComp2(aggOutFor2), 'SelectionComp_3', 'methodCall_0', [('inputTypeName', 'pdb::DepartmentTotal'), ('lambdaType', 'methodCall'), ('methodName', 'checkSales'), ('returnTypeName', 'pdb::DepartmentTotal')])\n"
                           "filteredInputForSelectionComp3(aggOutFor2) <= FILTER (methodCall_0OutFor_SelectionComp3(methodCall_0OutFor__checkSales), methodCall_0OutFor_SelectionComp3(aggOutFor2), 'SelectionComp_3')\n"
                           "methodCall_1OutFor_SelectionComp3 (methodCall_1OutFor__getTotSales) <= APPLY (filteredInputForSelectionComp3(aggOutFor2), filteredInputForSelectionComp3(), 'SelectionComp_3', 'methodCall_1', [('inputTypeName', 'pdb::DepartmentTotal'), ('lambdaType', 'methodCall'), ('methodName', 'getTotSales'), ('returnTypeName', 'pdb::DepartmentTotal')])\n"
                           "methodCall_1OutFor_SelectionComp3_out( ) <= OUTPUT ( methodCall_1OutFor_SelectionComp3 ( methodCall_1OutFor__getTotSales ), 'outSet1', 'myDB', 'SetWriter_4')\n"
                           "methodCall_0OutFor_SelectionComp5(aggOutFor2,methodCall_0OutFor__checkSales) <= APPLY (aggOutForAggregationComp2(aggOutFor2), aggOutForAggregationComp2(aggOutFor2), 'SelectionComp_5', 'methodCall_0', [('inputTypeName', 'pdb::DepartmentTotal'), ('lambdaType', 'methodCall'), ('methodName', 'checkSales'), ('returnTypeName', 'pdb::DepartmentTotal')])\n"
                           "filteredInputForSelectionComp5(aggOutFor2) <= FILTER (methodCall_0OutFor_SelectionComp5(methodCall_0OutFor__checkSales), methodCall_0OutFor_SelectionComp5(aggOutFor2), 'SelectionComp_5')\n"
                           "methodCall_1OutFor_SelectionComp5 (methodCall_1OutFor__getTotSales) <= APPLY (filteredInputForSelectionComp5(aggOutFor2), filteredInputForSelectionComp5(), 'SelectionComp_5', 'methodCall_1', [('inputTypeName', 'pdb::DepartmentTotal'), ('lambdaType', 'methodCall'), ('methodName', 'getTotSales'), ('returnTypeName', 'pdb::DepartmentTotal')])\n"
                           "methodCall_1OutFor_SelectionComp5_out( ) <= OUTPUT ( methodCall_1OutFor_SelectionComp5 ( methodCall_1OutFor__getTotSales ), 'outSet2', 'myDB', 'SetWriter_6')\n";

  // make a logger
  auto logger = make_shared<pdb::PDBLogger>("log.out");

  // make the mock client
  auto catalogClient = std::make_shared<MockCatalog>();
  ON_CALL(*catalogClient,
          getSet(testing::An<const std::string &>(), testing::An<const std::string &>(), testing::An<std::string &>())).WillByDefault(testing::Invoke(
          [&](const std::string &dbName, const std::string &setName, std::string &errMsg) {
            return std::make_shared<pdb::PDBCatalogSet>("mySetA", "myData", "Nothing", std::numeric_limits<size_t>::max());
          }));

  EXPECT_CALL(*catalogClient, getSet).Times(testing::Exactly(1));

  // init the optimizer
  pdb::PDBPhysicalOptimizer optimizer(compID, tcapString, catalogClient, logger);

  //



}

}
