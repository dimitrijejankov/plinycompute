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

#include <ReadInt.h>
#include <ReadStringIntPair.h>
#include <StringSelectionOfStringIntPair.h>
#include <IntSimpleJoin.h>
#include <WriteSumResult.h>
#include <IntAggregation.h>
#include <physicalAlgorithms/PDBPhysicalAlgorithm.h>
#include <physicalOptimizer/PDBJoinPhysicalNode.h>

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
      "inputData (in) <= SCAN ('by8_db', 'input_set', 'SetScanner_0', []) \n"
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
        return std::make_shared<pdb::PDBCatalogSet>("input_set", "by8_db", "Nothing", 10, PDB_CATALOG_SET_NO_CONTAINER);
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
  auto &source1 = aggAlgorithm->sources[0];
  EXPECT_EQ(source1.pageSet->sourceType, SetScanSource);
  EXPECT_EQ((std::string) source1.firstTupleSet, std::string("inputData"));
  EXPECT_EQ((std::string) source1.pageSet->pageSetIdentifier.second, std::string("inputData"));
  EXPECT_EQ(source1.pageSet->pageSetIdentifier.first, compID);
  EXPECT_EQ((std::string)source1.sourceSet->database, "by8_db");
  EXPECT_EQ((std::string)source1.sourceSet->set, "input_set");

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
  auto &source2 = strAlgorithm->sources[0];
  EXPECT_EQ(source2.pageSet->sourceType, AggregationSource);
  EXPECT_EQ((std::string) source2.firstTupleSet, std::string("agg"));
  EXPECT_EQ((std::string) source2.pageSet->pageSetIdentifier.second, std::string("aggWithValue"));
  EXPECT_EQ(source2.pageSet->pageSetIdentifier.first, compID);

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

TEST(TestPhysicalOptimizer, TestMultiSink) {

  // 1MB for algorithm and stuff
  const pdb::UseTemporaryAllocationBlock tempBlock{1024 * 1024};

  // setup the parameters
  uint64_t compID = 55;
  pdb::String tcapString = "inputData(in0) <= SCAN ('myData', 'mySetA', 'SetScanner_0')\n"
                           "methodCall_0OutFor_SelectionComp1(in0,methodCall_0OutFor__getSteve) <= APPLY (inputData(in0), inputData(in0), 'SelectionComp_1', 'methodCall_0', [('inputTypeName', 'pdb::Supervisor'), ('lambdaType', 'methodCall'), ('methodName', 'getSteve'), ('returnTypeName', 'pdb::Supervisor')])\n"
                           "attAccess_1OutForSelectionComp1(in0,methodCall_0OutFor__getSteve,att_1OutFor_me) <= APPLY (methodCall_0OutFor_SelectionComp1(in0), methodCall_0OutFor_SelectionComp1(in0,methodCall_0OutFor__getSteve), 'SelectionComp_1', 'attAccess_1', [('attName', 'me'), ('attTypeName', 'pdb::Handle&lt;pdb::Employee&gt;'), ('inputTypeName', 'pdb::Supervisor'), ('lambdaType', 'attAccess')])\n"
                           "equals_2OutForSelectionComp1(in0,methodCall_0OutFor__getSteve,att_1OutFor_me,bool_2_1) <= APPLY (attAccess_1OutForSelectionComp1(methodCall_0OutFor__getSteve,att_1OutFor_me), attAccess_1OutForSelectionComp1(in0,methodCall_0OutFor__getSteve,att_1OutFor_me), 'SelectionComp_1', '==_2', [('lambdaType', '==')])\n"
                           "filteredInputForSelectionComp1(in0) <= FILTER (equals_2OutForSelectionComp1(bool_2_1), equals_2OutForSelectionComp1(in0), 'SelectionComp_1')\n"
                           "methodCall_3OutFor_SelectionComp1(in0,methodCall_3OutFor__getMe) <= APPLY (filteredInputForSelectionComp1(in0), filteredInputForSelectionComp1(in0), 'SelectionComp_1', 'methodCall_3', [('inputTypeName', 'pdb::Supervisor'), ('lambdaType', 'methodCall'), ('methodName', 'getMe'), ('returnTypeName', 'pdb::Supervisor')])\n"
                           "deref_4OutForSelectionComp1 (methodCall_3OutFor__getMe) <= APPLY (methodCall_3OutFor_SelectionComp1(methodCall_3OutFor__getMe), methodCall_3OutFor_SelectionComp1(), 'SelectionComp_1', 'deref_4')\n"
                           "attAccess_0OutForAggregationComp2(methodCall_3OutFor__getMe,att_0OutFor_department) <= APPLY (deref_4OutForSelectionComp1(methodCall_3OutFor__getMe), deref_4OutForSelectionComp1(methodCall_3OutFor__getMe), 'AggregationComp_2', 'attAccess_0', [('attName', 'department'), ('attTypeName', 'pdb::String'), ('inputTypeName', 'pdb::Employee'), ('lambdaType', 'attAccess')])\n"
                           "deref_1OutForAggregationComp2(methodCall_3OutFor__getMe, att_0OutFor_department) <= APPLY (attAccess_0OutForAggregationComp2(att_0OutFor_department), attAccess_0OutForAggregationComp2(methodCall_3OutFor__getMe), 'AggregationComp_2', 'deref_1')\n"
                           "aggWithValue(att_0OutFor_department,methodCall_2OutFor__getSalary) <= APPLY (deref_1OutForAggregationComp2(methodCall_3OutFor__getMe), deref_1OutForAggregationComp2(att_0OutFor_department), 'AggregationComp_2', 'methodCall_2', [('inputTypeName', 'pdb::Employee'), ('lambdaType', 'methodCall'), ('methodName', 'getSalary'), ('returnTypeName', 'pdb::Employee')])\n"
                           "agg (aggOutFor2)<= AGGREGATE (aggWithValue(att_0OutFor_department, methodCall_2OutFor__getSalary),'AggregationComp_2')\n"
                           "selectionOne(aggOutFor2,methodCall_0OutFor__checkSales) <= APPLY (agg(aggOutFor2), agg(aggOutFor2), 'SelectionComp_3', 'methodCall_0', [('inputTypeName', 'pdb::DepartmentTotal'), ('lambdaType', 'methodCall'), ('methodName', 'checkSales'), ('returnTypeName', 'pdb::DepartmentTotal')])\n"
                           "selectionOneFilter(aggOutFor2) <= FILTER (selectionOne(methodCall_0OutFor__checkSales), selectionOne(aggOutFor2), 'SelectionComp_3')\n"
                           "selectionOneFilterRemoved (methodCall_1OutFor__getTotSales) <= APPLY (selectionOneFilter(aggOutFor2), selectionOneFilter(), 'SelectionComp_3', 'methodCall_1', [('inputTypeName', 'pdb::DepartmentTotal'), ('lambdaType', 'methodCall'), ('methodName', 'getTotSales'), ('returnTypeName', 'pdb::DepartmentTotal')])\n"
                           "selectionOneFilterRemoved_out( ) <= OUTPUT ( selectionOneFilterRemoved ( methodCall_1OutFor__getTotSales ), 'outSet1', 'myDB', 'SetWriter_4')\n"
                           "selectionTwo(aggOutFor2,methodCall_0OutFor__checkSales) <= APPLY (agg(aggOutFor2), agg(aggOutFor2), 'SelectionComp_5', 'methodCall_0', [('inputTypeName', 'pdb::DepartmentTotal'), ('lambdaType', 'methodCall'), ('methodName', 'checkSales'), ('returnTypeName', 'pdb::DepartmentTotal')])\n"
                           "selectionTwoFilter(aggOutFor2) <= FILTER (selectionTwo(methodCall_0OutFor__checkSales), selectionTwo(aggOutFor2), 'SelectionComp_5')\n"
                           "selectionTwoFilterRemoved (methodCall_1OutFor__getTotSales) <= APPLY (selectionTwoFilter(aggOutFor2), selectionTwoFilter(), 'SelectionComp_5', 'methodCall_1', [('inputTypeName', 'pdb::DepartmentTotal'), ('lambdaType', 'methodCall'), ('methodName', 'getTotSales'), ('returnTypeName', 'pdb::DepartmentTotal')])\n"
                           "selectionTwoFilterRemoved_out( ) <= OUTPUT ( selectionTwoFilterRemoved ( methodCall_1OutFor__getTotSales ), 'outSet2', 'myDB', 'SetWriter_6')\n";

  // make a logger
  auto logger = make_shared<pdb::PDBLogger>("log.out");

  // make the mock client
  auto catalogClient = std::make_shared<MockCatalog>();
  ON_CALL(*catalogClient,
          getSet(testing::An<const std::string &>(), testing::An<const std::string &>(), testing::An<std::string &>())).WillByDefault(testing::Invoke(
      [&](const std::string &dbName, const std::string &setName, std::string &errMsg) {
        return std::make_shared<pdb::PDBCatalogSet>("mySetA", "myData", "Nothing", std::numeric_limits<size_t>::max(), PDB_CATALOG_SET_NO_CONTAINER);
      }));

  EXPECT_CALL(*catalogClient, getSet).Times(testing::Exactly(1));

  // init the optimizer
  pdb::PDBPhysicalOptimizer optimizer(compID, tcapString, catalogClient, logger);

  /// 1. Get the first algorithm, it should be an aggregation

  // we should have one source so we should be able to generate an algorithm
  EXPECT_TRUE(optimizer.hasAlgorithmToRun());

  // the first algorithm should be a PDBAggregationPipeAlgorithm
  auto algorithm1 = optimizer.getNextAlgorithm();

  // cast the algorithm
  Handle<pdb::PDBAggregationPipeAlgorithm> aggAlgorithm = unsafeCast<pdb::PDBAggregationPipeAlgorithm>(algorithm1);

  // check the source
  auto &source1 = aggAlgorithm->sources[0];
  EXPECT_EQ(source1.pageSet->sourceType, SetScanSource);
  EXPECT_EQ((std::string) source1.firstTupleSet, std::string("inputData"));
  EXPECT_EQ((std::string) source1.pageSet->pageSetIdentifier.second, std::string("inputData"));
  EXPECT_EQ(source1.pageSet->pageSetIdentifier.first, compID);
  EXPECT_EQ((std::string) source1.sourceSet->database, "myData");
  EXPECT_EQ((std::string) source1.sourceSet->set, "mySetA");

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

  /// 2. Get the second algorithm, it should be an straight pipeline

  // check if the optimizer has another algorithm
  EXPECT_TRUE(optimizer.hasAlgorithmToRun());

  // the third algorithm should be a PDBStraightPipeAlgorithm
  auto algorithm2 = optimizer.getNextAlgorithm();

  // cast the algorithm
  Handle<pdb::PDBStraightPipeAlgorithm> strAlgorithm1 = unsafeCast<pdb::PDBStraightPipeAlgorithm>(algorithm2);

  // check the source
  auto &source2 = strAlgorithm1->sources[0];
  EXPECT_EQ(source2.pageSet->sourceType, AggregationSource);
  EXPECT_EQ((std::string) source2.firstTupleSet, std::string("agg"));
  EXPECT_EQ((std::string) source2.pageSet->pageSetIdentifier.second, std::string("aggWithValue"));
  EXPECT_EQ(source2.pageSet->pageSetIdentifier.first, compID);

  // check the sink
  EXPECT_EQ(strAlgorithm1->sink->sinkType, SetSink);
  EXPECT_EQ((std::string) strAlgorithm1->finalTupleSet, "selectionTwoFilterRemoved_out");
  EXPECT_EQ((std::string) strAlgorithm1->sink->pageSetIdentifier.second, "selectionTwoFilterRemoved_out");
  EXPECT_EQ(strAlgorithm1->sink->pageSetIdentifier.first, compID);

  // get the page sets we want to remove
  pageSetsToRemove = getPageSetsToRemove(optimizer);
  EXPECT_TRUE(pageSetsToRemove.find(std::make_pair(compID, "selectionTwoFilterRemoved_out")) != pageSetsToRemove.end());
  EXPECT_EQ(pageSetsToRemove.size(), 1);

  /// 3. Get the third algorithm, it should be an straight pipeline

  // check if the optimizer has another algorithm
  EXPECT_TRUE(optimizer.hasAlgorithmToRun());

  // the second algorithm should be a PDBStraightPipeAlgorithm
  auto algorithm3 = optimizer.getNextAlgorithm();

  // cast the algorithm
  Handle<pdb::PDBStraightPipeAlgorithm> strAlgorithm2 = unsafeCast<pdb::PDBStraightPipeAlgorithm>(algorithm3);

  // check the source
  auto &source3 = strAlgorithm2->sources[0];
  EXPECT_EQ(source3.pageSet->sourceType, AggregationSource);
  EXPECT_EQ((std::string) source3.firstTupleSet, std::string("agg"));
  EXPECT_EQ((std::string) source3.pageSet->pageSetIdentifier.second, std::string("aggWithValue"));
  EXPECT_EQ(source3.pageSet->pageSetIdentifier.first, compID);

  // check the sink
  EXPECT_EQ(strAlgorithm2->sink->sinkType, SetSink);
  EXPECT_EQ((std::string) strAlgorithm2->finalTupleSet, "selectionOneFilterRemoved_out");
  EXPECT_EQ((std::string) strAlgorithm2->sink->pageSetIdentifier.second, "selectionOneFilterRemoved_out");
  EXPECT_EQ(strAlgorithm2->sink->pageSetIdentifier.first, compID);

  // get the page sets we want to remove
  pageSetsToRemove = getPageSetsToRemove(optimizer);
  EXPECT_TRUE(pageSetsToRemove.find(std::make_pair(compID, "selectionOneFilterRemoved_out")) != pageSetsToRemove.end());
  EXPECT_TRUE(pageSetsToRemove.find(std::make_pair(compID, "aggWithValue")) != pageSetsToRemove.end());
  EXPECT_EQ(pageSetsToRemove.size(), 2);

  // we should not have anything anymore
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

  // set the join thrashold so we force a broadcast join
  PDBJoinPhysicalNode::SHUFFLE_JOIN_THRASHOLD = numeric_limits<uint64_t>::max();

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
          auto tmp = std::make_shared<pdb::PDBCatalogSet>("mySetA", "myData", "Nothing", 1000, PDB_CATALOG_SET_NO_CONTAINER);
          return tmp;
        } else {
          auto tmp = std::make_shared<pdb::PDBCatalogSet>("mySetB", "myData", "Nothing", 2000, PDB_CATALOG_SET_NO_CONTAINER);
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
  auto &source1 = algorithmBroadcastA->sources[0];
  EXPECT_EQ(source1.pageSet->sourceType, SetScanSource);
  EXPECT_EQ((std::string) source1.firstTupleSet, std::string("A"));
  EXPECT_EQ((std::string) source1.pageSet->pageSetIdentifier.second, std::string("A"));
  EXPECT_EQ(source1.pageSet->pageSetIdentifier.first, compID);
  EXPECT_EQ((std::string) source1.sourceSet->database, "myData");
  EXPECT_EQ((std::string) source1.sourceSet->set, "mySetA");

  // check the sink
  EXPECT_EQ(algorithmBroadcastA->sink->sinkType, BroadcastJoinSink);
  EXPECT_EQ((std::string) algorithmBroadcastA->finalTupleSet, "AHashed");
  EXPECT_EQ((std::string) algorithmBroadcastA->sink->pageSetIdentifier.second, "AHashed");
  EXPECT_EQ(algorithmBroadcastA->sink->pageSetIdentifier.first, compID);

  // get the page sets we want to remove
  auto pageSetsToRemove = getPageSetsToRemove(optimizer);
  EXPECT_TRUE(pageSetsToRemove.find(std::make_pair(compID, "AHashed_hashed_to_send")) != pageSetsToRemove.end());
  EXPECT_TRUE(pageSetsToRemove.find(std::make_pair(compID, "AHashed_hashed_to_recv")) != pageSetsToRemove.end());
  EXPECT_EQ(pageSetsToRemove.size(), 2);

  // we should have one source so we should be able to generate an algorithm
  EXPECT_TRUE(optimizer.hasAlgorithmToRun());

  // get the next algorithm
  auto algorithmPipelineThroughB = optimizer.getNextAlgorithm();

  // check the source
  auto &source2 = algorithmPipelineThroughB->sources[0];
  EXPECT_EQ(source2.pageSet->sourceType, SetScanSource);
  EXPECT_EQ((std::string) source2.firstTupleSet, std::string("B"));
  EXPECT_EQ((std::string) source2.pageSet->pageSetIdentifier.second, std::string("B"));
  EXPECT_EQ(source2.pageSet->pageSetIdentifier.first, compID);
  EXPECT_EQ((std::string) source2.sourceSet->database, "myData");
  EXPECT_EQ((std::string) source2.sourceSet->set, "mySetB");

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

  // get the page sets we want to remove
  pageSetsToRemove = getPageSetsToRemove(optimizer);
  EXPECT_TRUE(pageSetsToRemove.find(std::make_pair(compID, "AHashed")) != pageSetsToRemove.end());
  EXPECT_TRUE(pageSetsToRemove.find(std::make_pair(compID, "out")) != pageSetsToRemove.end());
  EXPECT_EQ(pageSetsToRemove.size(), 2);

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

  // set the join threshold so we force a shuffle join
  PDBJoinPhysicalNode::SHUFFLE_JOIN_THRASHOLD = 0;

  // make the mock client
  auto catalogClient = std::make_shared<MockCatalog>();
  ON_CALL(*catalogClient,
          getSet(testing::An<const std::string &>(),
                 testing::An<const std::string &>(),
                 testing::An<std::string &>())).WillByDefault(testing::Invoke(
      [&](const std::string &dbName, const std::string &setName, std::string &errMsg) {
        if (setName == "mySetA") {
          return std::make_shared<pdb::PDBCatalogSet>("mySetA", "myData", "Nothing", std::numeric_limits<size_t>::max(), PDB_CATALOG_SET_NO_CONTAINER);
        } else {
          return std::make_shared<pdb::PDBCatalogSet>("mySetB",
                                                      "myData",
                                                      "Nothing",
                                                      std::numeric_limits<size_t>::max() - 1, PDB_CATALOG_SET_NO_CONTAINER);
        }
      }));

  EXPECT_CALL(*catalogClient, getSet).Times(testing::Exactly(2));

  // init the optimizer
  pdb::PDBPhysicalOptimizer optimizer(compID, tcapString, catalogClient, logger);

  EXPECT_TRUE(optimizer.hasAlgorithmToRun());

  Handle<pdb::PDBShuffleForJoinAlgorithm> shuffleB = unsafeCast<pdb::PDBShuffleForJoinAlgorithm>(optimizer.getNextAlgorithm());

  // check the source
  auto &source1 = shuffleB->sources[0];
  EXPECT_EQ(source1.pageSet->sourceType, SetScanSource);
  EXPECT_EQ((std::string) source1.firstTupleSet, std::string("B"));
  EXPECT_EQ((std::string) source1.pageSet->pageSetIdentifier.second, std::string("B"));
  EXPECT_EQ(source1.pageSet->pageSetIdentifier.first, compID);
  EXPECT_EQ((std::string) source1.sourceSet->database, "myData");
  EXPECT_EQ((std::string) source1.sourceSet->set, "mySetB");

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
  auto pageSetsToRemove = getPageSetsToRemove(optimizer);
  EXPECT_TRUE(pageSetsToRemove.find(std::make_pair(compID, "BHashedOnA_to_shuffle")) != pageSetsToRemove.end());
  EXPECT_EQ(pageSetsToRemove.size(), 1);

  // we should have another algorithm now for side A
  EXPECT_TRUE(optimizer.hasAlgorithmToRun());

  Handle<pdb::PDBShuffleForJoinAlgorithm> shuffleA = unsafeCast<pdb::PDBShuffleForJoinAlgorithm>(optimizer.getNextAlgorithm());

  // check the source
  auto &source2 = shuffleA->sources[0];
  EXPECT_EQ(source2.pageSet->sourceType, SetScanSource);
  EXPECT_EQ((std::string) source2.firstTupleSet, "A");
  EXPECT_EQ((std::string) source2.pageSet->pageSetIdentifier.second, "A");
  EXPECT_EQ(source2.pageSet->pageSetIdentifier.first, compID);
  EXPECT_EQ((std::string) source2.sourceSet->database, "myData");
  EXPECT_EQ((std::string) source2.sourceSet->set, "mySetA");

  // check the intermediate set
  EXPECT_EQ(shuffleA->intermediate->sinkType, JoinShuffleIntermediateSink);
  EXPECT_EQ((std::string) shuffleA->intermediate->pageSetIdentifier.second, "AHashed_to_shuffle");
  EXPECT_EQ(shuffleA->intermediate->pageSetIdentifier.first, compID);

  // check the sink
  EXPECT_EQ(shuffleA->sink->sinkType, JoinShuffleSink);
  EXPECT_EQ((std::string) shuffleA->finalTupleSet, "AHashed");
  EXPECT_EQ((std::string) shuffleA->sink->pageSetIdentifier.second, "AHashed");
  EXPECT_EQ(shuffleA->sink->pageSetIdentifier.first, compID);

  // get the page sets we want to remove
  pageSetsToRemove = getPageSetsToRemove(optimizer);
  EXPECT_TRUE(pageSetsToRemove.find(std::make_pair(compID, "AHashed_to_shuffle")) != pageSetsToRemove.end());
  EXPECT_EQ(pageSetsToRemove.size(), 1);

  EXPECT_TRUE(optimizer.hasAlgorithmToRun());

  // update the stats on the page sets
  optimizer.updatePageSet(std::make_pair(compID, "BHashedOnA"), 10);
  optimizer.updatePageSet(std::make_pair(compID, "AHashed"), 11);

  Handle<pdb::PDBStraightPipeAlgorithm> doJoin = unsafeCast<pdb::PDBStraightPipeAlgorithm>(optimizer.getNextAlgorithm());

  // check the source
  auto &source3 = doJoin->sources[0];
  EXPECT_EQ(source3.pageSet->sourceType, ShuffledJoinTuplesSource);
  EXPECT_EQ((std::string) source3.firstTupleSet, "AandBJoined");
  EXPECT_EQ((std::string) source3.pageSet->pageSetIdentifier.second, "BHashedOnA");
  EXPECT_EQ(source3.pageSet->pageSetIdentifier.first, compID);

  // check the sink
  EXPECT_EQ(doJoin->sink->sinkType, SetSink);
  EXPECT_EQ((std::string) doJoin->finalTupleSet, "out");
  EXPECT_EQ((std::string) doJoin->sink->pageSetIdentifier.second, "out");
  EXPECT_EQ(doJoin->sink->pageSetIdentifier.first, compID);

  // get the page sets we want to remove
  pageSetsToRemove = getPageSetsToRemove(optimizer);
  EXPECT_TRUE(pageSetsToRemove.find(std::make_pair(compID, "AHashed")) != pageSetsToRemove.end());
  EXPECT_TRUE(pageSetsToRemove.find(std::make_pair(compID, "BHashedOnA")) != pageSetsToRemove.end());
  EXPECT_TRUE(pageSetsToRemove.find(std::make_pair(compID, "out")) != pageSetsToRemove.end());
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

  // set the join threshold so we force a shuffle join
  PDBJoinPhysicalNode::SHUFFLE_JOIN_THRASHOLD = std::numeric_limits<uint64_t>::max();

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
                                                          std::numeric_limits<size_t>::max() - 1,
                                                          PDB_CATALOG_SET_NO_CONTAINER);
          return tmp;
        } else if (setName == "mySetC") {
          auto tmp = std::make_shared<pdb::PDBCatalogSet>("mySetC", "myData", "Nothing", 0, PDB_CATALOG_SET_NO_CONTAINER);
          return tmp;
        } else {
          auto tmp = std::make_shared<pdb::PDBCatalogSet>("mySetB", "myData", "Nothing", std::numeric_limits<size_t>::max(), PDB_CATALOG_SET_NO_CONTAINER);
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
  auto &source1 = algorithmBroadcastC->sources[0];
  EXPECT_EQ(source1.pageSet->sourceType, SetScanSource);
  EXPECT_EQ((std::string) source1.firstTupleSet, std::string("C"));
  EXPECT_EQ((std::string) source1.pageSet->pageSetIdentifier.second, std::string("C"));
  EXPECT_EQ(source1.pageSet->pageSetIdentifier.first, compID);
  EXPECT_EQ((std::string) source1.sourceSet->database, "myData");
  EXPECT_EQ((std::string) source1.sourceSet->set, "mySetC");

  // check the sink
  EXPECT_EQ(algorithmBroadcastC->sink->sinkType, BroadcastJoinSink);
  EXPECT_EQ((std::string) algorithmBroadcastC->finalTupleSet, "CHashedOnC");
  EXPECT_EQ((std::string) algorithmBroadcastC->sink->pageSetIdentifier.second, "CHashedOnC");
  EXPECT_EQ(algorithmBroadcastC->sink->pageSetIdentifier.first, compID);

  // get the page sets we want to remove
  auto pageSetsToRemove = getPageSetsToRemove(optimizer);
  EXPECT_TRUE(pageSetsToRemove.find(std::make_pair(compID, "CHashedOnC_hashed_to_send")) != pageSetsToRemove.end());
  EXPECT_TRUE(pageSetsToRemove.find(std::make_pair(compID, "CHashedOnC_hashed_to_recv")) != pageSetsToRemove.end());
  EXPECT_EQ(pageSetsToRemove.size(), 2);

  // force a shuffle algorithm
  PDBJoinPhysicalNode::SHUFFLE_JOIN_THRASHOLD = 0;

  EXPECT_TRUE(optimizer.hasAlgorithmToRun());

  Handle<pdb::PDBShuffleForJoinAlgorithm> shuffleA = unsafeCast<pdb::PDBShuffleForJoinAlgorithm>(optimizer.getNextAlgorithm());

  // check the source
  auto &source2 = shuffleA->sources[0];
  EXPECT_EQ(source2.pageSet->sourceType, SetScanSource);
  EXPECT_EQ((std::string) source2.firstTupleSet, "A");
  EXPECT_EQ((std::string) source2.pageSet->pageSetIdentifier.second, "A");
  EXPECT_EQ(source2.pageSet->pageSetIdentifier.first, compID);
  EXPECT_EQ((std::string) source2.sourceSet->database, "myData");
  EXPECT_EQ((std::string) source2.sourceSet->set, "mySetA");

  // check the intermediate set
  EXPECT_EQ(shuffleA->intermediate->sinkType, JoinShuffleIntermediateSink);
  EXPECT_EQ((std::string) shuffleA->intermediate->pageSetIdentifier.second, "AHashed_to_shuffle");
  EXPECT_EQ(shuffleA->intermediate->pageSetIdentifier.first, compID);

  // check the sink
  EXPECT_EQ(shuffleA->sink->sinkType, JoinShuffleSink);
  EXPECT_EQ((std::string) shuffleA->finalTupleSet, "AHashed");
  EXPECT_EQ((std::string) shuffleA->sink->pageSetIdentifier.second, "AHashed");
  EXPECT_EQ(shuffleA->sink->pageSetIdentifier.first, compID);

  // get the page sets we want to remove
  pageSetsToRemove = getPageSetsToRemove(optimizer);
  EXPECT_TRUE(pageSetsToRemove.find(std::make_pair(compID, "AHashed_to_shuffle")) != pageSetsToRemove.end());
  EXPECT_EQ(pageSetsToRemove.size(), 1);


  EXPECT_TRUE(optimizer.hasAlgorithmToRun());

  Handle<pdb::PDBShuffleForJoinAlgorithm> shuffleB = unsafeCast<pdb::PDBShuffleForJoinAlgorithm>(optimizer.getNextAlgorithm());

  // check the source
  auto &source3 = shuffleB->sources[0];
  EXPECT_EQ(source3.pageSet->sourceType, SetScanSource);
  EXPECT_EQ((std::string) source3.firstTupleSet, std::string("B"));
  EXPECT_EQ((std::string) source3.pageSet->pageSetIdentifier.second, std::string("B"));
  EXPECT_EQ(source3.pageSet->pageSetIdentifier.first, compID);
  EXPECT_EQ((std::string)source3.sourceSet->database, "myData");
  EXPECT_EQ((std::string)source3.sourceSet->set, "mySetB");

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
  pageSetsToRemove = getPageSetsToRemove(optimizer);
  EXPECT_TRUE(pageSetsToRemove.find(std::make_pair(compID, "BHashedOnA_to_shuffle")) != pageSetsToRemove.end());
  EXPECT_EQ(pageSetsToRemove.size(), 1);

  EXPECT_TRUE(optimizer.hasAlgorithmToRun());

  // update the stats on the page sets
  optimizer.updatePageSet(std::make_pair(compID, "BHashedOnA"), 10);
  optimizer.updatePageSet(std::make_pair(compID, "AHashed"), 9);


  Handle<pdb::PDBStraightPipeAlgorithm> doJoin = unsafeCast<pdb::PDBStraightPipeAlgorithm>(optimizer.getNextAlgorithm());

  // check the source
  auto &source4 = doJoin->sources[0];
  EXPECT_EQ(source4.pageSet->sourceType, ShuffledJoinTuplesSource);
  EXPECT_EQ((std::string) source4.firstTupleSet, "AandBJoined");
  EXPECT_EQ((std::string) source4.pageSet->pageSetIdentifier.second, "AHashed");
  EXPECT_EQ(source4.pageSet->pageSetIdentifier.first, compID);

  // check the sink
  EXPECT_EQ(doJoin->sink->sinkType, SetSink);
  EXPECT_EQ((std::string) doJoin->finalTupleSet, "nothing");
  EXPECT_EQ((std::string) doJoin->sink->pageSetIdentifier.second, "nothing");
  EXPECT_EQ(doJoin->sink->pageSetIdentifier.first, compID);

  // get the page sets we want to remove
  pageSetsToRemove = getPageSetsToRemove(optimizer);
  EXPECT_TRUE(pageSetsToRemove.find(std::make_pair(compID, "CHashedOnC")) != pageSetsToRemove.end());
  EXPECT_TRUE(pageSetsToRemove.find(std::make_pair(compID, "AHashed")) != pageSetsToRemove.end());
  EXPECT_TRUE(pageSetsToRemove.find(std::make_pair(compID, "BHashedOnA")) != pageSetsToRemove.end());
  EXPECT_TRUE(pageSetsToRemove.find(std::make_pair(compID, "nothing")) != pageSetsToRemove.end());
  EXPECT_EQ(pageSetsToRemove.size(), 4);

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

TEST(TestPhysicalOptimizer, TestAggregationAfterTwoWayJoin) {

  const size_t compID = 76;

  // 1MB for algorithm and stuff
  const pdb::UseTemporaryAllocationBlock tempBlock{1024 * 1024};

  // the TCAP we are about to run
  String tcap = "inputDataForSetScanner_0(in0) <= SCAN ('test78_db', 'test78_set1', 'SetScanner_0')\n"
                "inputDataForSetScanner_1(in1) <= SCAN ('test78_db', 'test78_set2', 'SetScanner_1')\n"
                "\n"
                "/* Apply selection filtering */\n"
                "nativ_0OutForSelectionComp2(in1,nativ_0_2OutFor) <= APPLY (inputDataForSetScanner_1(in1), inputDataForSetScanner_1(in1), 'SelectionComp_2', 'native_lambda_0', [('lambdaType', 'native_lambda')])\n"
                "filteredInputForSelectionComp2(in1) <= FILTER (nativ_0OutForSelectionComp2(nativ_0_2OutFor), nativ_0OutForSelectionComp2(in1), 'SelectionComp_2')\n"
                "\n"
                "/* Apply selection projection */\n"
                "attAccess_1OutForSelectionComp2(in1,att_1OutFor_myString) <= APPLY (filteredInputForSelectionComp2(in1), filteredInputForSelectionComp2(in1), 'SelectionComp_2', 'attAccess_1', [('attName', 'myString'), ('attTypeName', 'pdb::Handle&lt;pdb::String&gt;'), ('inputTypeName', 'pdb::StringIntPair'), ('lambdaType', 'attAccess')])\n"
                "deref_2OutForSelectionComp2 (att_1OutFor_myString) <= APPLY (attAccess_1OutForSelectionComp2(att_1OutFor_myString), attAccess_1OutForSelectionComp2(), 'SelectionComp_2', 'deref_2')\n"
                "self_0ExtractedJoinComp3(in0,self_0_3Extracted) <= APPLY (inputDataForSetScanner_0(in0), inputDataForSetScanner_0(in0), 'JoinComp_3', 'self_0', [('lambdaType', 'self')])\n"
                "self_0ExtractedJoinComp3_hashed(in0,self_0_3Extracted_hash) <= HASHLEFT (self_0ExtractedJoinComp3(self_0_3Extracted), self_0ExtractedJoinComp3(in0), 'JoinComp_3', '==_2', [])\n"
                "attAccess_1ExtractedForJoinComp3(in1,att_1ExtractedFor_myInt) <= APPLY (inputDataForSetScanner_1(in1), inputDataForSetScanner_1(in1), 'JoinComp_3', 'attAccess_1', [('attName', 'myInt'), ('attTypeName', 'int'), ('inputTypeName', 'pdb::StringIntPair'), ('lambdaType', 'attAccess')])\n"
                "attAccess_1ExtractedForJoinComp3_hashed(in1,att_1ExtractedFor_myInt_hash) <= HASHRIGHT (attAccess_1ExtractedForJoinComp3(att_1ExtractedFor_myInt), attAccess_1ExtractedForJoinComp3(in1), 'JoinComp_3', '==_2', [])\n"
                "\n"
                "/* Join ( in0 ) and ( in1 ) */\n"
                "JoinedFor_equals2JoinComp3(in0, in1) <= JOIN (self_0ExtractedJoinComp3_hashed(self_0_3Extracted_hash), self_0ExtractedJoinComp3_hashed(in0), attAccess_1ExtractedForJoinComp3_hashed(att_1ExtractedFor_myInt_hash), attAccess_1ExtractedForJoinComp3_hashed(in1), 'JoinComp_3')\n"
                "JoinedFor_equals2JoinComp3_WithLHSExtracted(in0,in1,LHSExtractedFor_2_3) <= APPLY (JoinedFor_equals2JoinComp3(in0), JoinedFor_equals2JoinComp3(in0,in1), 'JoinComp_3', 'self_0', [('lambdaType', 'self')])\n"
                "JoinedFor_equals2JoinComp3_WithBOTHExtracted(in0,in1,LHSExtractedFor_2_3,RHSExtractedFor_2_3) <= APPLY (JoinedFor_equals2JoinComp3_WithLHSExtracted(in1), JoinedFor_equals2JoinComp3_WithLHSExtracted(in0,in1,LHSExtractedFor_2_3), 'JoinComp_3', 'attAccess_1', [('attName', 'myInt'), ('attTypeName', 'int'), ('inputTypeName', 'pdb::StringIntPair'), ('lambdaType', 'attAccess')])\n"
                "JoinedFor_equals2JoinComp3_BOOL(in0,in1,bool_2_3) <= APPLY (JoinedFor_equals2JoinComp3_WithBOTHExtracted(LHSExtractedFor_2_3,RHSExtractedFor_2_3), JoinedFor_equals2JoinComp3_WithBOTHExtracted(in0,in1), 'JoinComp_3', '==_2', [('lambdaType', '==')])\n"
                "JoinedFor_equals2JoinComp3_FILTERED(in0, in1) <= FILTER (JoinedFor_equals2JoinComp3_BOOL(bool_2_3), JoinedFor_equals2JoinComp3_BOOL(in0, in1), 'JoinComp_3')\n"
                "attAccess_3ExtractedForJoinComp3(in0,in1,att_3ExtractedFor_myString) <= APPLY (JoinedFor_equals2JoinComp3_FILTERED(in1), JoinedFor_equals2JoinComp3_FILTERED(in0,in1), 'JoinComp_3', 'attAccess_3', [('attName', 'myString'), ('attTypeName', 'pdb::Handle&lt;pdb::String&gt;'), ('inputTypeName', 'pdb::StringIntPair'), ('lambdaType', 'attAccess')])\n"
                "attAccess_3ExtractedForJoinComp3_hashed(in0,in1,att_3ExtractedFor_myString_hash) <= HASHLEFT (attAccess_3ExtractedForJoinComp3(att_3ExtractedFor_myString), attAccess_3ExtractedForJoinComp3(in0,in1), 'JoinComp_3', '==_5', [])\n"
                "self_4ExtractedJoinComp3(att_1OutFor_myString,self_4_3Extracted) <= APPLY (deref_2OutForSelectionComp2(att_1OutFor_myString), deref_2OutForSelectionComp2(att_1OutFor_myString), 'JoinComp_3', 'self_4', [('lambdaType', 'self')])\n"
                "self_4ExtractedJoinComp3_hashed(att_1OutFor_myString,self_4_3Extracted_hash) <= HASHRIGHT (self_4ExtractedJoinComp3(self_4_3Extracted), self_4ExtractedJoinComp3(att_1OutFor_myString), 'JoinComp_3', '==_5', [])\n"
                "\n"
                "/* Join ( in0 in1 ) and ( att_1OutFor_myString ) */\n"
                "JoinedFor_equals5JoinComp3(in0, in1, att_1OutFor_myString) <= JOIN (attAccess_3ExtractedForJoinComp3_hashed(att_3ExtractedFor_myString_hash), attAccess_3ExtractedForJoinComp3_hashed(in0, in1), self_4ExtractedJoinComp3_hashed(self_4_3Extracted_hash), self_4ExtractedJoinComp3_hashed(att_1OutFor_myString), 'JoinComp_3')\n"
                "JoinedFor_equals5JoinComp3_WithLHSExtracted(in0,in1,att_1OutFor_myString,LHSExtractedFor_5_3) <= APPLY (JoinedFor_equals5JoinComp3(in1), JoinedFor_equals5JoinComp3(in0,in1,att_1OutFor_myString), 'JoinComp_3', 'attAccess_3', [('attName', 'myString'), ('attTypeName', 'pdb::Handle&lt;pdb::String&gt;'), ('inputTypeName', 'pdb::StringIntPair'), ('lambdaType', 'attAccess')])\n"
                "JoinedFor_equals5JoinComp3_WithBOTHExtracted(in0,in1,att_1OutFor_myString,LHSExtractedFor_5_3,RHSExtractedFor_5_3) <= APPLY (JoinedFor_equals5JoinComp3_WithLHSExtracted(att_1OutFor_myString), JoinedFor_equals5JoinComp3_WithLHSExtracted(in0,in1,att_1OutFor_myString,LHSExtractedFor_5_3), 'JoinComp_3', 'self_4', [('lambdaType', 'self')])\n"
                "JoinedFor_equals5JoinComp3_BOOL(in0,in1,att_1OutFor_myString,bool_5_3) <= APPLY (JoinedFor_equals5JoinComp3_WithBOTHExtracted(LHSExtractedFor_5_3,RHSExtractedFor_5_3), JoinedFor_equals5JoinComp3_WithBOTHExtracted(in0,in1,att_1OutFor_myString), 'JoinComp_3', '==_5', [('lambdaType', '==')])\n"
                "JoinedFor_equals5JoinComp3_FILTERED(in0, in1, att_1OutFor_myString) <= FILTER (JoinedFor_equals5JoinComp3_BOOL(bool_5_3), JoinedFor_equals5JoinComp3_BOOL(in0, in1, att_1OutFor_myString), 'JoinComp_3')\n"
                "\n"
                "/* run Join projection on ( in0 )*/\n"
                "nativ_7OutForJoinComp3 (nativ_7_3OutFor) <= APPLY (JoinedFor_equals5JoinComp3_FILTERED(in0), JoinedFor_equals5JoinComp3_FILTERED(), 'JoinComp_3', 'native_lambda_7', [('lambdaType', 'native_lambda')])\n"
                "\n"
                "/* Extract key for aggregation */\n"
                "nativ_0OutForAggregationComp4(nativ_7_3OutFor,nativ_0_4OutFor) <= APPLY (nativ_7OutForJoinComp3(nativ_7_3OutFor), nativ_7OutForJoinComp3(nativ_7_3OutFor), 'AggregationComp_4', 'native_lambda_0', [('lambdaType', 'native_lambda')])\n"
                "\n"
                "/* Extract value for aggregation */\n"
                "nativ_1OutForAggregationComp4(nativ_0_4OutFor,nativ_1_4OutFor) <= APPLY (nativ_0OutForAggregationComp4(nativ_7_3OutFor), nativ_0OutForAggregationComp4(nativ_0_4OutFor), 'AggregationComp_4', 'native_lambda_1', [('lambdaType', 'native_lambda')])\n"
                "\n"
                "/* Apply aggregation */\n"
                "aggOutForAggregationComp4 (aggOutFor4)<= AGGREGATE (nativ_1OutForAggregationComp4(nativ_0_4OutFor, nativ_1_4OutFor),'AggregationComp_4')\n"
                "aggOutForAggregationComp4_out( ) <= OUTPUT ( aggOutForAggregationComp4 ( aggOutFor4 ), 'test78_db', 'output_set1', 'SetWriter_5')";


  // make a logger
  auto logger = make_shared<pdb::PDBLogger>("log.out");

  // make the mock client
  auto catalogClient = std::make_shared<MockCatalog>();
  ON_CALL(*catalogClient,
          getSet(testing::An<const std::string &>(),
                 testing::An<const std::string &>(),
                 testing::An<std::string &>())).WillByDefault(testing::Invoke(
      [&](const std::string &dbName, const std::string &setName, std::string &errMsg) {

        if (setName == "test78_set1") {
          auto tmp = std::make_shared<pdb::PDBCatalogSet>("test78_db",
                                                          "test78_set1",
                                                          "Nothing",
                                                          std::numeric_limits<size_t>::max() - 1,
                                                          PDB_CATALOG_SET_NO_CONTAINER);
          return tmp;
        } else {
          auto tmp = std::make_shared<pdb::PDBCatalogSet>("test78_db",
                                                          "test78_set2",
                                                          "Nothing",
                                                          std::numeric_limits<size_t>::max() - 2,
                                                          PDB_CATALOG_SET_NO_CONTAINER);
          return tmp;
        }
      }));

  EXPECT_CALL(*catalogClient, getSet).Times(testing::Exactly(3));

  // change the threshold so that we do a shuffle
  PDBJoinPhysicalNode::SHUFFLE_JOIN_THRASHOLD = 0;

  // init the optimizer
  pdb::PDBPhysicalOptimizer optimizer(compID, tcap, catalogClient, logger);

  /// 1. First algorithm

  EXPECT_TRUE(optimizer.hasAlgorithmToRun());

  Handle<pdb::PDBShuffleForJoinAlgorithm> shuffleSet2FirstJoin = unsafeCast<pdb::PDBShuffleForJoinAlgorithm>(optimizer.getNextAlgorithm());

  // check the source
  auto &source1 = shuffleSet2FirstJoin->sources[0];
  EXPECT_EQ(source1.pageSet->sourceType, SetScanSource);
  EXPECT_EQ((std::string) source1.firstTupleSet, std::string("inputDataForSetScanner_1"));
  EXPECT_EQ((std::string) source1.pageSet->pageSetIdentifier.second, std::string("inputDataForSetScanner_1"));
  EXPECT_EQ(source1.pageSet->pageSetIdentifier.first, compID);
  EXPECT_EQ((std::string) source1.sourceSet->database, "test78_db");
  EXPECT_EQ((std::string) source1.sourceSet->set, "test78_set2");

  // check the intermediate set
  EXPECT_EQ(shuffleSet2FirstJoin->intermediate->sinkType, JoinShuffleIntermediateSink);
  EXPECT_EQ((std::string) shuffleSet2FirstJoin->intermediate->pageSetIdentifier.second, "attAccess_1ExtractedForJoinComp3_hashed_to_shuffle");
  EXPECT_EQ(shuffleSet2FirstJoin->intermediate->pageSetIdentifier.first, compID);

  // check the sink
  EXPECT_EQ(shuffleSet2FirstJoin->sink->sinkType, JoinShuffleSink);
  EXPECT_EQ((std::string) shuffleSet2FirstJoin->finalTupleSet, "attAccess_1ExtractedForJoinComp3_hashed");
  EXPECT_EQ((std::string) shuffleSet2FirstJoin->sink->pageSetIdentifier.second, "attAccess_1ExtractedForJoinComp3_hashed");
  EXPECT_EQ(shuffleSet2FirstJoin->sink->pageSetIdentifier.first, compID);

  // get the page sets we want to remove
  auto pageSetsToRemove = getPageSetsToRemove(optimizer);
  EXPECT_TRUE(pageSetsToRemove.find(std::make_pair(compID, "attAccess_1ExtractedForJoinComp3_hashed_to_shuffle")) != pageSetsToRemove.end());
  EXPECT_EQ(pageSetsToRemove.size(), 1);

  /// 2. Second algorithm

  EXPECT_TRUE(optimizer.hasAlgorithmToRun());

  Handle<pdb::PDBShuffleForJoinAlgorithm> shuffleSet2SecondJoin = unsafeCast<pdb::PDBShuffleForJoinAlgorithm>(optimizer.getNextAlgorithm());

  // check the source
  auto &source2 = shuffleSet2SecondJoin->sources[0];
  EXPECT_EQ(source2.pageSet->sourceType, SetScanSource);
  EXPECT_EQ((std::string) source2.firstTupleSet, std::string("inputDataForSetScanner_1"));
  EXPECT_EQ((std::string) source2.pageSet->pageSetIdentifier.second, std::string("inputDataForSetScanner_1"));
  EXPECT_EQ(source2.pageSet->pageSetIdentifier.first, compID);
  EXPECT_EQ((std::string)source2.sourceSet->database, "test78_db");
  EXPECT_EQ((std::string)source2.sourceSet->set, "test78_set2");

  // check the intermediate set
  EXPECT_EQ(shuffleSet2SecondJoin->intermediate->sinkType, JoinShuffleIntermediateSink);
  EXPECT_EQ((std::string) shuffleSet2SecondJoin->intermediate->pageSetIdentifier.second, "self_4ExtractedJoinComp3_hashed_to_shuffle");
  EXPECT_EQ(shuffleSet2SecondJoin->intermediate->pageSetIdentifier.first, compID);

  // check the sink
  EXPECT_EQ(shuffleSet2SecondJoin->sink->sinkType, JoinShuffleSink);
  EXPECT_EQ((std::string) shuffleSet2SecondJoin->finalTupleSet, "self_4ExtractedJoinComp3_hashed");
  EXPECT_EQ((std::string) shuffleSet2SecondJoin->sink->pageSetIdentifier.second, "self_4ExtractedJoinComp3_hashed");
  EXPECT_EQ(shuffleSet2SecondJoin->sink->pageSetIdentifier.first, compID);

  // get the page sets we want to remove
  pageSetsToRemove = getPageSetsToRemove(optimizer);
  EXPECT_TRUE(pageSetsToRemove.find(std::make_pair(compID, "self_4ExtractedJoinComp3_hashed_to_shuffle")) != pageSetsToRemove.end());
  EXPECT_EQ(pageSetsToRemove.size(), 1);

  /// 3. Third algorithm

  EXPECT_TRUE(optimizer.hasAlgorithmToRun());

  Handle<pdb::PDBShuffleForJoinAlgorithm> shuffleSet1 = unsafeCast<pdb::PDBShuffleForJoinAlgorithm>(optimizer.getNextAlgorithm());

  // check the source
  auto &source3 = shuffleSet1->sources[0];
  EXPECT_EQ(source3.pageSet->sourceType, SetScanSource);
  EXPECT_EQ((std::string) source3.firstTupleSet, std::string("inputDataForSetScanner_0"));
  EXPECT_EQ((std::string) source3.pageSet->pageSetIdentifier.second, std::string("inputDataForSetScanner_0"));
  EXPECT_EQ(source3.pageSet->pageSetIdentifier.first, compID);
  EXPECT_EQ((std::string) source3.sourceSet->database, "test78_db");
  EXPECT_EQ((std::string) source3.sourceSet->set, "test78_set1");

  // check the intermediate set
  EXPECT_EQ(shuffleSet1->intermediate->sinkType, JoinShuffleIntermediateSink);
  EXPECT_EQ((std::string) shuffleSet1->intermediate->pageSetIdentifier.second, "self_0ExtractedJoinComp3_hashed_to_shuffle");
  EXPECT_EQ(shuffleSet1->intermediate->pageSetIdentifier.first, compID);

  // check the sink
  EXPECT_EQ(shuffleSet1->sink->sinkType, JoinShuffleSink);
  EXPECT_EQ((std::string) shuffleSet1->finalTupleSet, "self_0ExtractedJoinComp3_hashed");
  EXPECT_EQ((std::string) shuffleSet1->sink->pageSetIdentifier.second, "self_0ExtractedJoinComp3_hashed");
  EXPECT_EQ(shuffleSet1->sink->pageSetIdentifier.first, compID);

  // get the page sets we want to remove
  pageSetsToRemove = getPageSetsToRemove(optimizer);
  EXPECT_TRUE(pageSetsToRemove.find(std::make_pair(compID, "self_0ExtractedJoinComp3_hashed_to_shuffle")) != pageSetsToRemove.end());
  EXPECT_EQ(pageSetsToRemove.size(), 1);

  /// 4. Fourth algorithm

  // update the stats on the page sets
  optimizer.updatePageSet(std::make_pair(compID, "attAccess_1ExtractedForJoinComp3_hashed"), 10);
  optimizer.updatePageSet(std::make_pair(compID, "self_0ExtractedJoinComp3_hashed"), 11);

  EXPECT_TRUE(optimizer.hasAlgorithmToRun());

  // get the algorithm
  Handle<pdb::PDBShuffleForJoinAlgorithm> doJoin = unsafeCast<pdb::PDBShuffleForJoinAlgorithm>(optimizer.getNextAlgorithm());

  // check the source
  auto &source4 = doJoin->sources[0];
  EXPECT_EQ(source4.pageSet->sourceType, ShuffledJoinTuplesSource);
  EXPECT_EQ((std::string) source4.firstTupleSet, "JoinedFor_equals2JoinComp3");
  EXPECT_EQ((std::string) source4.pageSet->pageSetIdentifier.second, "attAccess_1ExtractedForJoinComp3_hashed");
  EXPECT_EQ(source4.pageSet->pageSetIdentifier.first, compID);

  // check the intermediate set
  EXPECT_EQ(doJoin->intermediate->sinkType, JoinShuffleIntermediateSink);
  EXPECT_EQ((std::string) doJoin->intermediate->pageSetIdentifier.second, "attAccess_3ExtractedForJoinComp3_hashed_to_shuffle");
  EXPECT_EQ(doJoin->intermediate->pageSetIdentifier.first, compID);

  // check the sink
  EXPECT_EQ(doJoin->sink->sinkType, JoinShuffleSink);
  EXPECT_EQ((std::string) doJoin->finalTupleSet, "attAccess_3ExtractedForJoinComp3_hashed");
  EXPECT_EQ((std::string) doJoin->sink->pageSetIdentifier.second, "attAccess_3ExtractedForJoinComp3_hashed");
  EXPECT_EQ(doJoin->sink->pageSetIdentifier.first, compID);

  // get the page sets we want to remove
  pageSetsToRemove = getPageSetsToRemove(optimizer);
  EXPECT_TRUE(pageSetsToRemove.find(std::make_pair(compID, "attAccess_1ExtractedForJoinComp3_hashed")) != pageSetsToRemove.end());
  EXPECT_TRUE(pageSetsToRemove.find(std::make_pair(compID, "self_0ExtractedJoinComp3_hashed")) != pageSetsToRemove.end());
  EXPECT_TRUE(pageSetsToRemove.find(std::make_pair(compID, "attAccess_1ExtractedForJoinComp3_hashed")) != pageSetsToRemove.end());
  EXPECT_EQ(pageSetsToRemove.size(), 3);

  // check how many secondary sources we have
  pdb::Vector<pdb::Handle<PDBSourcePageSetSpec>> &additionalSources = *doJoin->secondarySources;

  // we should have only one
  EXPECT_EQ(additionalSources.size(), 1);

  // check it
  EXPECT_EQ(additionalSources[0]->sourceType, ShuffledJoinTuplesSource);
  EXPECT_EQ(additionalSources[0]->pageSetIdentifier.first, compID);
  EXPECT_EQ((std::string) additionalSources[0]->pageSetIdentifier.second, "self_0ExtractedJoinComp3_hashed");

  /// 5. Fourth algorithm

  // update the stats on the page sets
  optimizer.updatePageSet(std::make_pair(compID, "self_4ExtractedJoinComp3_hashed"), 10);
  optimizer.updatePageSet(std::make_pair(compID, "attAccess_3ExtractedForJoinComp3_hashed"), 11);

  EXPECT_TRUE(optimizer.hasAlgorithmToRun());

  // cast the algorithm
  Handle<pdb::PDBAggregationPipeAlgorithm> aggAlgorithm = unsafeCast<pdb::PDBAggregationPipeAlgorithm>(optimizer.getNextAlgorithm());

  // check the source
  auto &source5 = aggAlgorithm->sources[0];
  EXPECT_EQ(source5.pageSet->sourceType, ShuffledJoinTuplesSource);
  EXPECT_EQ((std::string) source5.firstTupleSet, std::string("JoinedFor_equals5JoinComp3"));
  EXPECT_EQ((std::string) source5.pageSet->pageSetIdentifier.second, std::string("self_4ExtractedJoinComp3_hashed"));
  EXPECT_EQ(source5.pageSet->pageSetIdentifier.first, compID);

  // check the sink that we are
  EXPECT_EQ(aggAlgorithm->hashedToSend->sinkType, AggShuffleSink);
  EXPECT_EQ((std::string) aggAlgorithm->hashedToSend->pageSetIdentifier.second, "nativ_1OutForAggregationComp4_hashed_to_send");
  EXPECT_EQ(aggAlgorithm->hashedToSend->pageSetIdentifier.first, compID);

  // check the source
  EXPECT_EQ(aggAlgorithm->hashedToRecv->sourceType, ShuffledAggregatesSource);
  EXPECT_EQ((std::string) aggAlgorithm->hashedToRecv->pageSetIdentifier.second, std::string("nativ_1OutForAggregationComp4_hashed_to_recv"));
  EXPECT_EQ(aggAlgorithm->hashedToRecv->pageSetIdentifier.first, compID);

  // check the sink
  EXPECT_EQ(aggAlgorithm->sink->sinkType, AggregationSink);
  EXPECT_EQ((std::string) aggAlgorithm->finalTupleSet, "nativ_1OutForAggregationComp4");
  EXPECT_EQ((std::string) aggAlgorithm->sink->pageSetIdentifier.second, "nativ_1OutForAggregationComp4");
  EXPECT_EQ(aggAlgorithm->sink->pageSetIdentifier.first, compID);

  // check the sets we need to materialize
  EXPECT_EQ(aggAlgorithm->setsToMaterialize->size(), 1);
  EXPECT_EQ((std::string)(*aggAlgorithm->setsToMaterialize)[0].database, "test78_db");
  EXPECT_EQ((std::string)(*aggAlgorithm->setsToMaterialize)[0].set, "output_set1");

  // get the page sets we want to remove
  pageSetsToRemove = getPageSetsToRemove(optimizer);
  EXPECT_TRUE(pageSetsToRemove.find(std::make_pair(compID, "nativ_1OutForAggregationComp4_hashed_to_send")) != pageSetsToRemove.end());
  EXPECT_TRUE(pageSetsToRemove.find(std::make_pair(compID, "nativ_1OutForAggregationComp4_hashed_to_recv")) != pageSetsToRemove.end());
  EXPECT_TRUE(pageSetsToRemove.find(std::make_pair(compID, "self_4ExtractedJoinComp3_hashed")) != pageSetsToRemove.end());
  EXPECT_TRUE(pageSetsToRemove.find(std::make_pair(compID, "attAccess_3ExtractedForJoinComp3_hashed")) != pageSetsToRemove.end());
  EXPECT_TRUE(pageSetsToRemove.find(std::make_pair(compID, "nativ_1OutForAggregationComp4")) != pageSetsToRemove.end());
  EXPECT_EQ(pageSetsToRemove.size(), 5);

  // we should have another source that reads the aggregation so we can generate another algorithm
  EXPECT_FALSE(optimizer.hasAlgorithmToRun());
}

TEST(TestPhysicalOptimizer, TestUnion1) {

  std::string tcap = "inputDataForSetScanner_0(in0) <= SCAN ('db', 'input_set1', 'SetScanner_0')\n"
                     "inputDataForSetScanner_1(in1) <= SCAN ('db', 'input_set2', 'SetScanner_1')\n"
                     "unionOutUnionComp2 (unionOutFor2 )<= UNION (inputDataForSetScanner_0(in0), inputDataForSetScanner_1(in1),'UnionComp_2')\n"
                     "unionOutUnionComp2_out( ) <= OUTPUT ( unionOutUnionComp2 ( unionOutFor2 ), 'chris_db', 'outputSet', 'SetWriter_3')";
  // get the string to compile
  tcap.push_back('\0');

  // 1MB for algorithm and stuff
  const pdb::UseTemporaryAllocationBlock tempBlock{1024 * 1024};

  // make a logger
  auto logger = make_shared<pdb::PDBLogger>("log.out");

  // make the mock client
  auto catalogClient = std::make_shared<MockCatalog>();
  ON_CALL(*catalogClient,
          getSet(testing::An<const std::string &>(),
                 testing::An<const std::string &>(),
                 testing::An<std::string &>())).WillByDefault(testing::Invoke(
      [&](const std::string &dbName, const std::string &setName, std::string &errMsg) {

        if (setName == "test78_set1") {
          auto tmp = std::make_shared<pdb::PDBCatalogSet>("db",
                                                          "input_set1",
                                                          "Nothing",
                                                          std::numeric_limits<size_t>::max() - 1,
                                                          PDB_CATALOG_SET_NO_CONTAINER);
          return tmp;
        } else {
          auto tmp = std::make_shared<pdb::PDBCatalogSet>("db",
                                                          "input_set2",
                                                          "Nothing",
                                                          std::numeric_limits<size_t>::max() - 2,
                                                          PDB_CATALOG_SET_NO_CONTAINER);
          return tmp;
        }
      }));

  EXPECT_CALL(*catalogClient, getSet).Times(testing::Exactly(2));

  // set the computation id
  const size_t compID = 76;

  // init the optimizer
  pdb::PDBPhysicalOptimizer optimizer(compID, tcap, catalogClient, logger);

  /// 1. There should only be one algorithm

  EXPECT_TRUE(optimizer.hasAlgorithmToRun());

  Handle<pdb::PDBStraightPipeAlgorithm> unionAlgorithm = unsafeCast<pdb::PDBStraightPipeAlgorithm>(optimizer.getNextAlgorithm());

  // we should have two sources
  EXPECT_EQ(unionAlgorithm->sources.size(), 2);

  // these are the sources we expect to have
  std::set<string> expectedSources = {"inputDataForSetScanner_0", "inputDataForSetScanner_1"};

  // check if the first source is right
  auto &source1 = unionAlgorithm->sources[0];
  auto it = expectedSources.find(source1.firstTupleSet);
  EXPECT_TRUE(it != expectedSources.end());
  EXPECT_EQ((std::string) source1.pageSet->pageSetIdentifier.second, *it);
  EXPECT_EQ(source1.pageSet->pageSetIdentifier.first, compID);
  EXPECT_EQ(source1.pageSet->sourceType, SetScanSource);

  // check if the second source is right
  auto &source2 = unionAlgorithm->sources[1];
  auto jt = expectedSources.find(source2.firstTupleSet);
  EXPECT_TRUE(jt != expectedSources.end());
  EXPECT_TRUE(it != jt);
  EXPECT_EQ((std::string) source2.pageSet->pageSetIdentifier.second, *jt);
  EXPECT_EQ(source2.pageSet->pageSetIdentifier.first, compID);
  EXPECT_EQ(source2.pageSet->sourceType, SetScanSource);

  // check the sink
  EXPECT_EQ(unionAlgorithm->sink->sinkType, SetSink);
  EXPECT_EQ((std::string) unionAlgorithm->finalTupleSet, "unionOutUnionComp2_out");
  EXPECT_EQ((std::string) unionAlgorithm->sink->pageSetIdentifier.second, "unionOutUnionComp2_out");
  EXPECT_EQ(unionAlgorithm->sink->pageSetIdentifier.first, compID);

  // get the page sets we want to remove
  auto pageSetsToRemove = getPageSetsToRemove(optimizer);
  EXPECT_TRUE(pageSetsToRemove.find(std::make_pair(compID, "unionOutUnionComp2_out")) != pageSetsToRemove.end());
  EXPECT_EQ(pageSetsToRemove.size(), 1);
}

TEST(TestPhysicalOptimizer, TestUnion2) {

  std::string tcap = "inputDataForSetScanner_0(in0) <= SCAN ('chris_db', 'input_set1', 'SetScanner_0')\n"
                     "/* Apply selection filtering */\n"
                     "OutFor_native_lambda_0SelectionComp1(in0,OutFor_native_lambda_0_1) <= APPLY (inputDataForSetScanner_0(in0), inputDataForSetScanner_0(in0), 'SelectionComp_1', 'native_lambda_0', [('lambdaType', 'native_lambda')])\n"
                     "filteredInputForSelectionComp1(in0) <= FILTER (OutFor_native_lambda_0SelectionComp1(OutFor_native_lambda_0_1), OutFor_native_lambda_0SelectionComp1(in0), 'SelectionComp_1')\n"
                     "/* Apply selection projection */\n"
                     "OutFor_native_lambda_1SelectionComp1(OutFor_native_lambda_1_1) <= APPLY (filteredInputForSelectionComp1(in0), filteredInputForSelectionComp1(), 'SelectionComp_1', 'native_lambda_1', [('lambdaType', 'native_lambda')])\n"
                     "inputDataForSetScanner_2(in2) <= SCAN ('chris_db', 'input_set2', 'SetScanner_2')\n"
                     "/* Apply selection filtering */\n"
                     "OutFor_native_lambda_0SelectionComp3(in2,OutFor_native_lambda_0_3) <= APPLY (inputDataForSetScanner_2(in2), inputDataForSetScanner_2(in2), 'SelectionComp_3', 'native_lambda_0', [('lambdaType', 'native_lambda')])\n"
                     "filteredInputForSelectionComp3(in2) <= FILTER (OutFor_native_lambda_0SelectionComp3(OutFor_native_lambda_0_3), OutFor_native_lambda_0SelectionComp3(in2), 'SelectionComp_3')\n"
                     "/* Apply selection projection */\n"
                     "OutFor_native_lambda_1SelectionComp3(OutFor_native_lambda_1_3) <= APPLY (filteredInputForSelectionComp3(in2), filteredInputForSelectionComp3(), 'SelectionComp_3', 'native_lambda_1', [('lambdaType', 'native_lambda')])\n"
                     "unionOutUnionComp4 (unionOutFor4)<= UNION (OutFor_native_lambda_1SelectionComp1(OutFor_native_lambda_1_1),OutFor_native_lambda_1SelectionComp3(OutFor_native_lambda_1_3),'UnionComp_4')\n"
                     "inputDataForSetScanner_5(in5) <= SCAN ('chris_db', 'input_set3', 'SetScanner_5')\n"
                     "/* Apply selection filtering */\n"
                     "OutFor_native_lambda_0SelectionComp6(in5,OutFor_native_lambda_0_6) <= APPLY (inputDataForSetScanner_5(in5), inputDataForSetScanner_5(in5), 'SelectionComp_6', 'native_lambda_0', [('lambdaType', 'native_lambda')])\n"
                     "filteredInputForSelectionComp6(in5) <= FILTER (OutFor_native_lambda_0SelectionComp6(OutFor_native_lambda_0_6), OutFor_native_lambda_0SelectionComp6(in5), 'SelectionComp_6')\n"
                     "/* Apply selection projection */\n"
                     "OutFor_native_lambda_1SelectionComp6(OutFor_native_lambda_1_6) <= APPLY (filteredInputForSelectionComp6(in5), filteredInputForSelectionComp6(), 'SelectionComp_6', 'native_lambda_1', [('lambdaType', 'native_lambda')])\n"
                     "unionOutUnionComp7 (unionOutFor7)<= UNION (unionOutUnionComp4(unionOutFor4),OutFor_native_lambda_1SelectionComp6(OutFor_native_lambda_1_6),'UnionComp_7')\n"
                     "unionOutUnionComp7_out( ) <= OUTPUT ( unionOutUnionComp7 ( unionOutFor7 ), 'chris_db', 'output_set', 'SetWriter_8')";

  // get the string to compile
  tcap.push_back('\0');

  // 1MB for algorithm and stuff
  const pdb::UseTemporaryAllocationBlock tempBlock{1024 * 1024};

  // make a logger
  auto logger = make_shared<pdb::PDBLogger>("log.out");

  // make the mock client
  auto catalogClient = std::make_shared<MockCatalog>();
  ON_CALL(*catalogClient,
          getSet(testing::An<const std::string &>(),
                 testing::An<const std::string &>(),
                 testing::An<std::string &>())).WillByDefault(testing::Invoke(
      [&](const std::string &dbName, const std::string &setName, std::string &errMsg) {

        if (setName == "input_set1") {
          auto tmp = std::make_shared<pdb::PDBCatalogSet>("db",
                                                          "input_set1",
                                                          "Nothing",
                                                          std::numeric_limits<size_t>::max() - 1,
                                                          PDB_CATALOG_SET_NO_CONTAINER);
          return tmp;
        }
        else if(setName == "input_set2") {
          auto tmp = std::make_shared<pdb::PDBCatalogSet>("db",
                                                          "input_set2",
                                                          "Nothing",
                                                          std::numeric_limits<size_t>::max() - 1,
                                                          PDB_CATALOG_SET_NO_CONTAINER);
          return tmp;
        }
        else {
          auto tmp = std::make_shared<pdb::PDBCatalogSet>("db",
                                                          "input_set3",
                                                          "Nothing",
                                                          std::numeric_limits<size_t>::max() - 2,
                                                          PDB_CATALOG_SET_NO_CONTAINER);
          return tmp;
        }
      }));

  EXPECT_CALL(*catalogClient, getSet).Times(testing::Exactly(3));

  // set the computation id
  const size_t compID = 76;

  // init the optimizer
  pdb::PDBPhysicalOptimizer optimizer(compID, tcap, catalogClient, logger);

  /// 1. There should only be one algorithm

  EXPECT_TRUE(optimizer.hasAlgorithmToRun());

  Handle<pdb::PDBStraightPipeAlgorithm> unionAlgorithm = unsafeCast<pdb::PDBStraightPipeAlgorithm>(optimizer.getNextAlgorithm());

  // we should have two sources
  EXPECT_EQ(unionAlgorithm->sources.size(), 3);

  // these are the sources we expect to have
  std::set<string> expectedSources = {"inputDataForSetScanner_0", "inputDataForSetScanner_2", "inputDataForSetScanner_5"};

  // check if the first source is right
  auto &source1 = unionAlgorithm->sources[0];
  auto it = expectedSources.find(source1.firstTupleSet);
  EXPECT_TRUE(it != expectedSources.end());
  EXPECT_EQ((std::string) source1.pageSet->pageSetIdentifier.second, *it);
  EXPECT_EQ(source1.pageSet->pageSetIdentifier.first, compID);
  EXPECT_EQ(source1.pageSet->sourceType, SetScanSource);

  // check if the second source is right
  auto &source2 = unionAlgorithm->sources[1];
  auto jt = expectedSources.find(source2.firstTupleSet);
  EXPECT_TRUE(jt != expectedSources.end());
  EXPECT_TRUE(it != jt);
  EXPECT_EQ((std::string) source2.pageSet->pageSetIdentifier.second, *jt);
  EXPECT_EQ(source2.pageSet->pageSetIdentifier.first, compID);
  EXPECT_EQ(source2.pageSet->sourceType, SetScanSource);

  // check if the third source is right
  auto &source3 = unionAlgorithm->sources[1];
  auto kt = expectedSources.find(source3.firstTupleSet);
  EXPECT_TRUE(jt != expectedSources.end());
  EXPECT_TRUE(it != jt);
  EXPECT_TRUE(it != kt);
  EXPECT_EQ((std::string) source3.pageSet->pageSetIdentifier.second, *jt);
  EXPECT_EQ(source3.pageSet->pageSetIdentifier.first, compID);
  EXPECT_EQ(source3.pageSet->sourceType, SetScanSource);

  // check the sink
  EXPECT_EQ(unionAlgorithm->sink->sinkType, SetSink);
  EXPECT_EQ((std::string) unionAlgorithm->finalTupleSet, "unionOutUnionComp7_out");
  EXPECT_EQ((std::string) unionAlgorithm->sink->pageSetIdentifier.second, "unionOutUnionComp7_out");
  EXPECT_EQ(unionAlgorithm->sink->pageSetIdentifier.first, compID);

  // get the page sets we want to remove
  auto pageSetsToRemove = getPageSetsToRemove(optimizer);
  EXPECT_TRUE(pageSetsToRemove.find(std::make_pair(compID, "unionOutUnionComp7_out")) != pageSetsToRemove.end());
  EXPECT_EQ(pageSetsToRemove.size(), 1);
}

}
