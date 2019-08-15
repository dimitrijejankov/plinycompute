#include <gtest/gtest.h>
#include <UseTemporaryAllocationBlock.h>
#include <Computation.h>
#include <JoinAggTransformation.h>

#include "../../../../applications/TestMatrixMultiply/sharedLibraries/headers/MatrixBlock.h"
#include "../../../../applications/TestMatrixMultiply/sharedLibraries/headers/MatrixScanner.h"
#include "../../../../applications/TestMatrixMultiply/sharedLibraries/headers/MatrixMultiplyJoin.h"
#include "../../../../applications/TestMatrixMultiply/sharedLibraries/headers/MatrixMultiplyAggregation.h"
#include "../../../../applications/TestMatrixMultiply/sharedLibraries/headers/MatrixWriter.h"

namespace pdb {

using namespace matrix;

TEST(PipelineTest, TestJoinAggPipeline) {

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
  Vector<Handle<Computation>> myComputations;

  // put them in the list of computations
  myComputations.push_back(readA);
  myComputations.push_back(readB);
  myComputations.push_back(join);
  myComputations.push_back(myAggregation);
  myComputations.push_back(myWriter);

  std::string tcap = "inputDataForSetScanner_0(in0) <= SCAN ('myData', 'A', 'SetScanner_0')\n"
                     "inputDataForSetScanner_1(in1) <= SCAN ('myData', 'B', 'SetScanner_1')\n"
                     "\n"
                     "/* Apply join selection */\n"
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
                     "\n"
                     "/* Apply join projection*/\n"
                     "OutFor_key_7JoinComp2(in0,in1,OutFor_key_7_2) <= APPLY (OutFor_OutForJoinedFor_equals_0JoinComp2_FILTERED(in0), OutFor_OutForJoinedFor_equals_0JoinComp2_FILTERED(in0,in1), 'JoinComp_2', 'key_7', [('lambdaType', 'key')])\n"
                     "OutFor_key_8JoinComp2(in0,in1,OutFor_key_7_2,OutFor_key_8_2) <= APPLY (OutFor_key_7JoinComp2(in1), OutFor_key_7JoinComp2(in0,in1,OutFor_key_7_2), 'JoinComp_2', 'key_8', [('lambdaType', 'key')])\n"
                     "OutFor_native_lambda_6JoinComp2(in0,in1,OutFor_key_7_2,OutFor_key_8_2,OutFor_native_lambda_6_2) <= APPLY (OutFor_key_8JoinComp2(OutFor_key_7_2,OutFor_key_8_2), OutFor_key_8JoinComp2(in0,in1,OutFor_key_7_2,OutFor_key_8_2), 'JoinComp_2', 'native_lambda_6', [('lambdaType', 'native_lambda')])\n"
                     "OutFor_value_10JoinComp2(in0,in1,OutFor_key_7_2,OutFor_key_8_2,OutFor_native_lambda_6_2,OutFor_value_10_2) <= APPLY (OutFor_native_lambda_6JoinComp2(in0), OutFor_native_lambda_6JoinComp2(in0,in1,OutFor_key_7_2,OutFor_key_8_2,OutFor_native_lambda_6_2), 'JoinComp_2', 'value_10', [('lambdaType', 'value')])\n"
                     "OutFor_value_11JoinComp2(in0,in1,OutFor_key_7_2,OutFor_key_8_2,OutFor_native_lambda_6_2,OutFor_value_10_2,OutFor_value_11_2) <= APPLY (OutFor_value_10JoinComp2(in1), OutFor_value_10JoinComp2(in0,in1,OutFor_key_7_2,OutFor_key_8_2,OutFor_native_lambda_6_2,OutFor_value_10_2), 'JoinComp_2', 'value_11', [('lambdaType', 'value')])\n"
                     "OutFor_native_lambda_9JoinComp2(in0,in1,OutFor_key_7_2,OutFor_key_8_2,OutFor_native_lambda_6_2,OutFor_value_10_2,OutFor_value_11_2,OutFor_native_lambda_9_2) <= APPLY (OutFor_value_11JoinComp2(OutFor_value_10_2,OutFor_value_11_2), OutFor_value_11JoinComp2(in0,in1,OutFor_key_7_2,OutFor_key_8_2,OutFor_native_lambda_6_2,OutFor_value_10_2,OutFor_value_11_2), 'JoinComp_2', 'native_lambda_9', [('lambdaType', 'native_lambda')])\n"
                     "OutFor_joinRec_5JoinComp2(OutFor_joinRec_5_2) <= APPLY (OutFor_native_lambda_9JoinComp2(OutFor_native_lambda_6_2,OutFor_native_lambda_9_2), OutFor_native_lambda_9JoinComp2(), 'JoinComp_2', 'joinRec_5', [('lambdaType', 'joinRec')])\n"
                     "\n"
                     "/* Extract key for aggregation */\n"
                     "OutFor_key_2AggregationComp3(OutFor_joinRec_5_2,OutFor_key_2_3) <= APPLY (OutFor_joinRec_5JoinComp2(OutFor_joinRec_5_2), OutFor_joinRec_5JoinComp2(OutFor_joinRec_5_2), 'AggregationComp_3', 'key_2', [('lambdaType', 'key')])\n"
                     "OutFor_self_1AggregationComp3(OutFor_joinRec_5_2,OutFor_key_2_3,OutFor_self_1_3) <= APPLY (OutFor_key_2AggregationComp3(OutFor_key_2_3), OutFor_key_2AggregationComp3(OutFor_joinRec_5_2,OutFor_key_2_3), 'AggregationComp_3', 'self_1', [('lambdaType', 'self')])\n"
                     "OutFor_deref_0AggregationComp3(OutFor_joinRec_5_2,OutFor_deref_0_3) <= APPLY (OutFor_self_1AggregationComp3(OutFor_self_1_3), OutFor_self_1AggregationComp3(OutFor_joinRec_5_2), 'AggregationComp_3', 'deref_0', [('lambdaType', 'deref')])\n"
                     "\n"
                     "/* Extract value for aggregation */\n"
                     "OutFor_methodCall_4AggregationComp3(OutFor_joinRec_5_2,OutFor_deref_0_3,OutFor_methodCall_4_3) <= APPLY (OutFor_deref_0AggregationComp3(OutFor_joinRec_5_2), OutFor_deref_0AggregationComp3(OutFor_joinRec_5_2,OutFor_deref_0_3), 'AggregationComp_3', 'methodCall_4', [('inputTypeName', 'pdb::matrix::MatrixBlock'), ('lambdaType', 'methodCall'), ('methodName', 'getValueRef'), ('returnTypeName', 'pdb::matrix::MatrixBlock')])\n"
                     "OutFor_deref_3AggregationComp3(OutFor_deref_0_3,OutFor_deref_3_3) <= APPLY (OutFor_methodCall_4AggregationComp3(OutFor_methodCall_4_3), OutFor_methodCall_4AggregationComp3(OutFor_deref_0_3), 'AggregationComp_3', 'deref_3', [('lambdaType', 'deref')])\n"
                     "\n"
                     "/* Apply aggregation */\n"
                     "aggOutForAggregationComp3 (aggOutFor3)<= AGGREGATE (OutFor_deref_3AggregationComp3(OutFor_deref_0_3,OutFor_deref_3_3),'AggregationComp_3')\n"
                     "aggOutForAggregationComp3_out( ) <= OUTPUT ( aggOutForAggregationComp3 ( aggOutFor3 ), 'myData', 'C', 'SetWriter_4')";

  // and create a query object that contains all of this stuff
  ComputePlan myPlan(tcap, myComputations);

  JoinAggTransformation transformation(myPlan.getPlan(),
                                       "inputDataForSetScanner_0",
                                       {"myData", "A"},
                                       "inputDataForSetScanner_1",
                                       {"myData", "B"},
                                       "OutForJoinedFor_equals_0JoinComp2",
                                       "aggOutForAggregationComp3");

  transformation.transform();

}


}
