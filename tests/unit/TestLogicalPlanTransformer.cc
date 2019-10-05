#include <string>
#include <gtest/gtest.h>
#include <LogicalPlanTransformer.h>

#include "AtomicComputationList.h"
#include "Parser.h"
#include "LogicalPlan.h"

using namespace pdb;

LogicalPlanPtr makeLogicalPlan() {

  // this is the object allocation block where all of this stuff will reside
  makeObjectAllocatorBlock(128 * 1024 * 1024, true);

  // the TCAP we want to parse
  std::string tcap = "inputDataForSetScanner_0(in0) <= SCAN ('db', 'a', 'SetScanner_0')\n"
                     "inputDataForSetScanner_1(in1) <= SCAN ('db', 'b', 'SetScanner_1')\n"
                     "OutFor_key_2JoinComp2(in0,OutFor_key_2_2) <= APPLY (inputDataForSetScanner_0(in0), inputDataForSetScanner_0(in0), 'JoinComp_2', 'key_2', [('lambdaType', 'key')])\n"
                     "OutFor_attAccess_1JoinComp2(in0,OutFor_key_2_2,OutFor_attAccess_1_2) <= APPLY (OutFor_key_2JoinComp2(OutFor_key_2_2), OutFor_key_2JoinComp2(in0,OutFor_key_2_2), 'JoinComp_2', 'attAccess_1', [('attName', 'colID'), ('attTypeName', 'unsignedint'), ('inputTypeName', 'pdb::matrix::MatrixBlockMeta'), ('lambdaType', 'attAccess')])\n"
                     "OutFor_key_4JoinComp2(in1,OutFor_key_4_2) <= APPLY (inputDataForSetScanner_1(in1), inputDataForSetScanner_1(in1), 'JoinComp_2', 'key_4', [('lambdaType', 'key')])\n"
                     "OutFor_attAccess_3JoinComp2(in1,OutFor_key_4_2,OutFor_attAccess_3_2) <= APPLY (OutFor_key_4JoinComp2(OutFor_key_4_2), OutFor_key_4JoinComp2(in1,OutFor_key_4_2), 'JoinComp_2', 'attAccess_3', [('attName', 'rowID'), ('attTypeName', 'unsignedint'), ('inputTypeName', 'pdb::matrix::MatrixBlockMeta'), ('lambdaType', 'attAccess')])\n"
                     "OutFor_attAccess_1JoinComp2_hashed(in0,OutFor_attAccess_1_2_hash) <= HASHLEFT (OutFor_attAccess_1JoinComp2(OutFor_attAccess_1_2), OutFor_attAccess_1JoinComp2(in0), 'JoinComp_2', '==_0', [])\n"
                     "OutFor_attAccess_3JoinComp2_hashed(in1,OutFor_attAccess_3_2_hash) <= HASHRIGHT (OutFor_attAccess_3JoinComp2(OutFor_attAccess_3_2), OutFor_attAccess_3JoinComp2(in1), 'JoinComp_2', '==_0', [])\n"
                     "OutForJoinedFor_equals_0JoinComp2(in0,in1) <= JOIN (OutFor_attAccess_1JoinComp2_hashed(OutFor_attAccess_1_2_hash), OutFor_attAccess_1JoinComp2_hashed(in0), OutFor_attAccess_3JoinComp2_hashed(OutFor_attAccess_3_2_hash), OutFor_attAccess_3JoinComp2_hashed(in1), 'JoinComp_2')\n"
                     "LExtractedFor0_key_2JoinComp2(in0,in1,LExtractedFor0_key_2_2) <= APPLY (OutForJoinedFor_equals_0JoinComp2(in0), OutForJoinedFor_equals_0JoinComp2(in0,in1), 'JoinComp_2', 'key_2', [('lambdaType', 'key')])\n"
                     "LExtractedFor0_attAccess_1JoinComp2(in0,in1,LExtractedFor0_key_2_2,LExtractedFor0_attAccess_1_2) <= APPLY (LExtractedFor0_key_2JoinComp2(LExtractedFor0_key_2_2), LExtractedFor0_key_2JoinComp2(in0,in1,LExtractedFor0_key_2_2), 'JoinComp_2', 'attAccess_1', [('attName', 'colID'), ('attTypeName', 'unsignedint'), ('inputTypeName', 'pdb::matrix::MatrixBlockMeta'), ('lambdaType', 'attAccess')])\n"
                     "RExtractedFor0_key_4JoinComp2(in0,in1,LExtractedFor0_key_2_2,LExtractedFor0_attAccess_1_2,RExtractedFor0_key_4_2) <= APPLY (LExtractedFor0_attAccess_1JoinComp2(in1), LExtractedFor0_attAccess_1JoinComp2(in0,in1,LExtractedFor0_key_2_2,LExtractedFor0_attAccess_1_2), 'JoinComp_2', 'key_4', [('lambdaType', 'key')])\n"
                     "RExtractedFor0_attAccess_3JoinComp2(in0,in1,LExtractedFor0_key_2_2,LExtractedFor0_attAccess_1_2,RExtractedFor0_key_4_2,RExtractedFor0_attAccess_3_2) <= APPLY (RExtractedFor0_key_4JoinComp2(RExtractedFor0_key_4_2), RExtractedFor0_key_4JoinComp2(in0,in1,LExtractedFor0_key_2_2,LExtractedFor0_attAccess_1_2,RExtractedFor0_key_4_2), 'JoinComp_2', 'attAccess_3', [('attName', 'rowID'), ('attTypeName', 'unsignedint'), ('inputTypeName', 'pdb::matrix::MatrixBlockMeta'), ('lambdaType', 'attAccess')])\n"
                     "OutFor_OutForJoinedFor_equals_0JoinComp2_BOOL(in0,in1,bool_0_2) <= APPLY (RExtractedFor0_attAccess_3JoinComp2(LExtractedFor0_attAccess_1_2,RExtractedFor0_attAccess_3_2), RExtractedFor0_attAccess_3JoinComp2(in0,in1), 'JoinComp_2', '==_0', [('lambdaType', '==')])\n"
                     "OutFor_OutForJoinedFor_equals_0JoinComp2_FILTERED(in0,in1) <= FILTER (OutFor_OutForJoinedFor_equals_0JoinComp2_BOOL(bool_0_2), OutFor_OutForJoinedFor_equals_0JoinComp2_BOOL(in0,in1), 'JoinComp_2')\n"
                     "OutFor_key_7JoinComp2(in0,in1,OutFor_key_7_2) <= APPLY (OutFor_OutForJoinedFor_equals_0JoinComp2_FILTERED(in0), OutFor_OutForJoinedFor_equals_0JoinComp2_FILTERED(in0,in1), 'JoinComp_2', 'key_7', [('lambdaType', 'key')])\n"
                     "OutFor_key_8JoinComp2(in0,in1,OutFor_key_7_2,OutFor_key_8_2) <= APPLY (OutFor_key_7JoinComp2(in1), OutFor_key_7JoinComp2(in0,in1,OutFor_key_7_2), 'JoinComp_2', 'key_8', [('lambdaType', 'key')])\n"
                     "OutFor_native_lambda_6JoinComp2(in0,in1,OutFor_key_7_2,OutFor_key_8_2,OutFor_native_lambda_6_2) <= APPLY (OutFor_key_8JoinComp2(OutFor_key_7_2,OutFor_key_8_2), OutFor_key_8JoinComp2(in0,in1,OutFor_key_7_2,OutFor_key_8_2), 'JoinComp_2', 'native_lambda_6', [('lambdaType', 'native_lambda')])\n"
                     "OutFor_value_10JoinComp2(in0,in1,OutFor_key_7_2,OutFor_key_8_2,OutFor_native_lambda_6_2,OutFor_value_10_2) <= APPLY (OutFor_native_lambda_6JoinComp2(in0), OutFor_native_lambda_6JoinComp2(in0,in1,OutFor_key_7_2,OutFor_key_8_2,OutFor_native_lambda_6_2), 'JoinComp_2', 'value_10', [('lambdaType', 'value')])\n"
                     "OutFor_value_11JoinComp2(in0,in1,OutFor_key_7_2,OutFor_key_8_2,OutFor_native_lambda_6_2,OutFor_value_10_2,OutFor_value_11_2) <= APPLY (OutFor_value_10JoinComp2(in1), OutFor_value_10JoinComp2(in0,in1,OutFor_key_7_2,OutFor_key_8_2,OutFor_native_lambda_6_2,OutFor_value_10_2), 'JoinComp_2', 'value_11', [('lambdaType', 'value')])\n"
                     "OutFor_native_lambda_9JoinComp2(in0,in1,OutFor_key_7_2,OutFor_key_8_2,OutFor_native_lambda_6_2,OutFor_value_10_2,OutFor_value_11_2,OutFor_native_lambda_9_2) <= APPLY (OutFor_value_11JoinComp2(OutFor_value_10_2,OutFor_value_11_2), OutFor_value_11JoinComp2(in0,in1,OutFor_key_7_2,OutFor_key_8_2,OutFor_native_lambda_6_2,OutFor_value_10_2,OutFor_value_11_2), 'JoinComp_2', 'native_lambda_9', [('lambdaType', 'native_lambda')])\n"
                     "OutFor_joinRec_5JoinComp2(OutFor_joinRec_5_2) <= APPLY (OutFor_native_lambda_9JoinComp2(OutFor_native_lambda_6_2,OutFor_native_lambda_9_2), OutFor_native_lambda_9JoinComp2(), 'JoinComp_2', 'joinRec_5', [('lambdaType', 'joinRec')])\n"
                     "OutFor_key_2AggregationComp3(OutFor_joinRec_5_2,OutFor_key_2_3) <= APPLY (OutFor_joinRec_5JoinComp2(OutFor_joinRec_5_2), OutFor_joinRec_5JoinComp2(OutFor_joinRec_5_2), 'AggregationComp_3', 'key_2', [('lambdaType', 'key')])\n"
                     "OutFor_self_1AggregationComp3(OutFor_joinRec_5_2,OutFor_key_2_3,OutFor_self_1_3) <= APPLY (OutFor_key_2AggregationComp3(OutFor_key_2_3), OutFor_key_2AggregationComp3(OutFor_joinRec_5_2,OutFor_key_2_3), 'AggregationComp_3', 'self_1', [('lambdaType', 'self')])\n"
                     "OutFor_deref_0AggregationComp3(OutFor_joinRec_5_2,OutFor_deref_0_3) <= APPLY (OutFor_self_1AggregationComp3(OutFor_self_1_3), OutFor_self_1AggregationComp3(OutFor_joinRec_5_2), 'AggregationComp_3', 'deref_0', [('lambdaType', 'deref')])\n"
                     "OutFor_methodCall_4AggregationComp3(OutFor_joinRec_5_2,OutFor_deref_0_3,OutFor_methodCall_4_3) <= APPLY (OutFor_deref_0AggregationComp3(OutFor_joinRec_5_2), OutFor_deref_0AggregationComp3(OutFor_joinRec_5_2,OutFor_deref_0_3), 'AggregationComp_3', 'methodCall_4', [('inputTypeName', 'pdb::matrix::MatrixBlock'), ('lambdaType', 'methodCall'), ('methodName', 'getValueRef'), ('returnTypeName', 'pdb::matrix::MatrixBlock')])\n"
                     "OutFor_deref_3AggregationComp3(OutFor_deref_0_3,OutFor_deref_3_3) <= APPLY (OutFor_methodCall_4AggregationComp3(OutFor_methodCall_4_3), OutFor_methodCall_4AggregationComp3(OutFor_deref_0_3), 'AggregationComp_3', 'deref_3', [('lambdaType', 'deref')])\n"
                     "aggOutForAggregationComp3 (aggOutFor3)<= AGGREGATE (OutFor_deref_3AggregationComp3(OutFor_deref_0_3,OutFor_deref_3_3),'AggregationComp_3')\n"
                     "inputDataForSetScanner_4(in4) <= SCAN ('db', 'c', 'SetScanner_4')\n"
                     "OutFor_key_2JoinComp5(aggOutFor3,OutFor_key_2_5) <= APPLY (aggOutForAggregationComp3(aggOutFor3), aggOutForAggregationComp3(aggOutFor3), 'JoinComp_5', 'key_2', [('lambdaType', 'key')])\n"
                     "OutFor_attAccess_1JoinComp5(aggOutFor3,OutFor_key_2_5,OutFor_attAccess_1_5) <= APPLY (OutFor_key_2JoinComp5(OutFor_key_2_5), OutFor_key_2JoinComp5(aggOutFor3,OutFor_key_2_5), 'JoinComp_5', 'attAccess_1', [('attName', 'colID'), ('attTypeName', 'unsignedint'), ('inputTypeName', 'pdb::matrix::MatrixBlockMeta'), ('lambdaType', 'attAccess')])\n"
                     "OutFor_key_4JoinComp5(in4,OutFor_key_4_5) <= APPLY (inputDataForSetScanner_4(in4), inputDataForSetScanner_4(in4), 'JoinComp_5', 'key_4', [('lambdaType', 'key')])\n"
                     "OutFor_attAccess_3JoinComp5(in4,OutFor_key_4_5,OutFor_attAccess_3_5) <= APPLY (OutFor_key_4JoinComp5(OutFor_key_4_5), OutFor_key_4JoinComp5(in4,OutFor_key_4_5), 'JoinComp_5', 'attAccess_3', [('attName', 'rowID'), ('attTypeName', 'unsignedint'), ('inputTypeName', 'pdb::matrix::MatrixBlockMeta'), ('lambdaType', 'attAccess')])\n"
                     "OutFor_attAccess_1JoinComp5_hashed(aggOutFor3,OutFor_attAccess_1_5_hash) <= HASHLEFT (OutFor_attAccess_1JoinComp5(OutFor_attAccess_1_5), OutFor_attAccess_1JoinComp5(aggOutFor3), 'JoinComp_5', '==_0', [])\n"
                     "OutFor_attAccess_3JoinComp5_hashed(in4,OutFor_attAccess_3_5_hash) <= HASHRIGHT (OutFor_attAccess_3JoinComp5(OutFor_attAccess_3_5), OutFor_attAccess_3JoinComp5(in4), 'JoinComp_5', '==_0', [])\n"
                     "OutForJoinedFor_equals_0JoinComp5(aggOutFor3,in4) <= JOIN (OutFor_attAccess_1JoinComp5_hashed(OutFor_attAccess_1_5_hash), OutFor_attAccess_1JoinComp5_hashed(aggOutFor3), OutFor_attAccess_3JoinComp5_hashed(OutFor_attAccess_3_5_hash), OutFor_attAccess_3JoinComp5_hashed(in4), 'JoinComp_5')\n"
                     "LExtractedFor0_key_2JoinComp5(aggOutFor3,in4,LExtractedFor0_key_2_5) <= APPLY (OutForJoinedFor_equals_0JoinComp5(aggOutFor3), OutForJoinedFor_equals_0JoinComp5(aggOutFor3,in4), 'JoinComp_5', 'key_2', [('lambdaType', 'key')])\n"
                     "LExtractedFor0_attAccess_1JoinComp5(aggOutFor3,in4,LExtractedFor0_key_2_5,LExtractedFor0_attAccess_1_5) <= APPLY (LExtractedFor0_key_2JoinComp5(LExtractedFor0_key_2_5), LExtractedFor0_key_2JoinComp5(aggOutFor3,in4,LExtractedFor0_key_2_5), 'JoinComp_5', 'attAccess_1', [('attName', 'colID'), ('attTypeName', 'unsignedint'), ('inputTypeName', 'pdb::matrix::MatrixBlockMeta'), ('lambdaType', 'attAccess')])\n"
                     "RExtractedFor0_key_4JoinComp5(aggOutFor3,in4,LExtractedFor0_key_2_5,LExtractedFor0_attAccess_1_5,RExtractedFor0_key_4_5) <= APPLY (LExtractedFor0_attAccess_1JoinComp5(in4), LExtractedFor0_attAccess_1JoinComp5(aggOutFor3,in4,LExtractedFor0_key_2_5,LExtractedFor0_attAccess_1_5), 'JoinComp_5', 'key_4', [('lambdaType', 'key')])\n"
                     "RExtractedFor0_attAccess_3JoinComp5(aggOutFor3,in4,LExtractedFor0_key_2_5,LExtractedFor0_attAccess_1_5,RExtractedFor0_key_4_5,RExtractedFor0_attAccess_3_5) <= APPLY (RExtractedFor0_key_4JoinComp5(RExtractedFor0_key_4_5), RExtractedFor0_key_4JoinComp5(aggOutFor3,in4,LExtractedFor0_key_2_5,LExtractedFor0_attAccess_1_5,RExtractedFor0_key_4_5), 'JoinComp_5', 'attAccess_3', [('attName', 'rowID'), ('attTypeName', 'unsignedint'), ('inputTypeName', 'pdb::matrix::MatrixBlockMeta'), ('lambdaType', 'attAccess')])\n"
                     "OutFor_OutForJoinedFor_equals_0JoinComp5_BOOL(aggOutFor3,in4,bool_0_5) <= APPLY (RExtractedFor0_attAccess_3JoinComp5(LExtractedFor0_attAccess_1_5,RExtractedFor0_attAccess_3_5), RExtractedFor0_attAccess_3JoinComp5(aggOutFor3,in4), 'JoinComp_5', '==_0', [('lambdaType', '==')])\n"
                     "OutFor_OutForJoinedFor_equals_0JoinComp5_FILTERED(aggOutFor3,in4) <= FILTER (OutFor_OutForJoinedFor_equals_0JoinComp5_BOOL(bool_0_5), OutFor_OutForJoinedFor_equals_0JoinComp5_BOOL(aggOutFor3,in4), 'JoinComp_5')\n"
                     "OutFor_key_7JoinComp5(aggOutFor3,in4,OutFor_key_7_5) <= APPLY (OutFor_OutForJoinedFor_equals_0JoinComp5_FILTERED(aggOutFor3), OutFor_OutForJoinedFor_equals_0JoinComp5_FILTERED(aggOutFor3,in4), 'JoinComp_5', 'key_7', [('lambdaType', 'key')])\n"
                     "OutFor_key_8JoinComp5(aggOutFor3,in4,OutFor_key_7_5,OutFor_key_8_5) <= APPLY (OutFor_key_7JoinComp5(in4), OutFor_key_7JoinComp5(aggOutFor3,in4,OutFor_key_7_5), 'JoinComp_5', 'key_8', [('lambdaType', 'key')])\n"
                     "OutFor_native_lambda_6JoinComp5(aggOutFor3,in4,OutFor_key_7_5,OutFor_key_8_5, ABCJoinedKey) <= APPLY (OutFor_key_8JoinComp5(OutFor_key_7_5,OutFor_key_8_5), OutFor_key_8JoinComp5(aggOutFor3,in4,OutFor_key_7_5,OutFor_key_8_5), 'JoinComp_5', 'native_lambda_6', [('lambdaType', 'native_lambda')])\n"
                     "OutFor_value_10JoinComp5(aggOutFor3,in4,OutFor_key_7_5,OutFor_key_8_5, ABCJoinedKey,OutFor_value_10_5) <= APPLY (OutFor_native_lambda_6JoinComp5(aggOutFor3), OutFor_native_lambda_6JoinComp5(aggOutFor3,in4,OutFor_key_7_5,OutFor_key_8_5, ABCJoinedKey), 'JoinComp_5', 'value_10', [('lambdaType', 'value')])\n"
                     "OutFor_value_11JoinComp5(aggOutFor3,in4,OutFor_key_7_5,OutFor_key_8_5, ABCJoinedKey,OutFor_value_10_5,OutFor_value_11_5) <= APPLY (OutFor_value_10JoinComp5(in4), OutFor_value_10JoinComp5(aggOutFor3,in4,OutFor_key_7_5,OutFor_key_8_5, ABCJoinedKey,OutFor_value_10_5), 'JoinComp_5', 'value_11', [('lambdaType', 'value')])\n"
                     "OutFor_native_lambda_9JoinComp5(aggOutFor3,in4,OutFor_key_7_5,OutFor_key_8_5, ABCJoinedKey,OutFor_value_10_5,OutFor_value_11_5,OutFor_native_lambda_9_5) <= APPLY (OutFor_value_11JoinComp5(OutFor_value_10_5,OutFor_value_11_5), OutFor_value_11JoinComp5(aggOutFor3,in4,OutFor_key_7_5,OutFor_key_8_5, ABCJoinedKey,OutFor_value_10_5,OutFor_value_11_5), 'JoinComp_5', 'native_lambda_9', [('lambdaType', 'native_lambda')])\n"
                     "OutFor_joinRec_5JoinComp5(OutFor_joinRec_5_5) <= APPLY (OutFor_native_lambda_9JoinComp5( ABCJoinedKey,OutFor_native_lambda_9_5), OutFor_native_lambda_9JoinComp5(), 'JoinComp_5', 'joinRec_5', [('lambdaType', 'joinRec')])\n"
                     "AggKey(OutFor_joinRec_5_5,OutFor_key_2_6) <= APPLY (OutFor_joinRec_5JoinComp5(OutFor_joinRec_5_5), OutFor_joinRec_5JoinComp5(OutFor_joinRec_5_5), 'AggregationComp_6', 'key_2', [('lambdaType', 'key')])\n"
                     "SelfKey(OutFor_joinRec_5_5,OutFor_key_2_6,OutFor_self_1_6) <= APPLY (AggKey(OutFor_key_2_6), AggKey(OutFor_joinRec_5_5,OutFor_key_2_6), 'AggregationComp_6', 'self_1', [('lambdaType', 'self')])\n"
                     "DerefKey(OutFor_joinRec_5_5,OutFor_deref_0_6) <= APPLY (SelfKey(OutFor_self_1_6), SelfKey(OutFor_joinRec_5_5), 'AggregationComp_6', 'deref_0', [('lambdaType', 'deref')])\n"
                     "AggVal(OutFor_joinRec_5_5,OutFor_deref_0_6,OutFor_methodCall_4_6) <= APPLY (DerefKey(OutFor_joinRec_5_5), DerefKey(OutFor_joinRec_5_5,OutFor_deref_0_6), 'AggregationComp_6', 'methodCall_4', [('inputTypeName', 'pdb::matrix::MatrixBlock'), ('lambdaType', 'methodCall'), ('methodName', 'getValueRef'), ('returnTypeName', 'pdb::matrix::MatrixBlock')])\n"
                     "DerefVal(OutFor_deref_0_6,OutFor_deref_3_6) <= APPLY (AggVal(OutFor_methodCall_4_6), AggVal(OutFor_deref_0_6), 'AggregationComp_6', 'deref_3', [('lambdaType', 'deref')])\n"
                     "Agg (aggOutFor6)<= AGGREGATE (DerefVal(OutFor_deref_0_6,OutFor_deref_3_6),'AggregationComp_6')\n"
                     "Agg_out( ) <= OUTPUT ( Agg ( aggOutFor6 ), 'db', 'd', 'SetWriter_7')";

  // get the string to compile
  tcap.push_back('\0');

  // where the result of the parse goes
  AtomicComputationList *myResult;

  // now, do the compilation
  yyscan_t scanner;
  LexerExtra extra{""};
  yylex_init_extra(&extra, &scanner);
  const YY_BUFFER_STATE buffer{yy_scan_string(tcap.data(), scanner)};
  const int parseFailed{yyparse(scanner, &myResult)};
  yy_delete_buffer(buffer, scanner);
  yylex_destroy(scanner);

  // if it didn't parse, get outta here
  if (parseFailed) {
    std::cout << "Parse error when compiling TCAP: " << extra.errorMessage;
    exit(1);
  }

  // make this empty it is not important
  pdb::Vector<pdb::Handle<pdb::Computation>> allComputations;

  // this is the logical plan to return
  auto myPlan = std::make_shared<pdb::LogicalPlan>(*myResult, allComputations);
  delete myResult;

  // return the plan we just created
  return myPlan;
}

TEST(TestLogicalPlanTransformer, Test1) {

  // make the logical plan
  auto logicalPlan = makeLogicalPlan();

  // print out the TCAP
  std::cout << "The original Plan : \n";
  std::cout << *logicalPlan << std::endl;

  // make the transformer
  auto transformer = std::make_shared<LogicalPlanTransformer>(logicalPlan);

  // the the start computation of the left and right size of the join
  std::string leftInputSet = "aggOutForAggregationComp3";
  std::string rightInputSet = "inputDataForSetScanner_4";

  // the join
  std::string joinTupleSet = "OutForJoinedFor_equals_0JoinComp5";

  // the last computation of the aggregation
  std::string preAggTupleSet = "AggKey";

  // where we want to cut off the plan
  std::string writeSet = "Agg_out";

  // add the transformation
  transformer->addTransformation(std::make_shared<InsertKeyScanSetsTransformation>(leftInputSet));
  transformer->addTransformation(std::make_shared<InsertKeyScanSetsTransformation>(rightInputSet));
  transformer->addTransformation(std::make_shared<JoinKeySideTransformation>(leftInputSet));
  transformer->addTransformation(std::make_shared<JoinKeySideTransformation>(rightInputSet));
  transformer->addTransformation(std::make_shared<JoinKeyTransformation>(joinTupleSet));
  transformer->addTransformation(std::make_shared<AggKeyTransformation>(preAggTupleSet));
  transformer->addTransformation(std::make_shared<DropDependents>(writeSet));
  transformer->addTransformation(std::make_shared<AddJoinTID>(joinTupleSet));

  // apply all the transformations
  auto newPlan = transformer->applyTransformations();

  // modified TCAP
  std::cout << "The new Plan : \n";
  std::cout << *newPlan << std::endl;
}