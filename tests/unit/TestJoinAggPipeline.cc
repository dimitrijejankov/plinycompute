#include <gtest/gtest.h>
#include <UseTemporaryAllocationBlock.h>
#include <Computation.h>

#include "../../../../applications/TestMatrixMultiply/sharedLibraries/headers/MatrixBlock.h"
#include "../../../../applications/TestMatrixMultiply/sharedLibraries/headers/MatrixScanner.h"
#include "../../../../applications/TestMatrixMultiply/sharedLibraries/headers/MatrixMultiplyJoin.h"
#include "../../../../applications/TestMatrixMultiply/sharedLibraries/headers/MatrixMultiplyAggregation.h"
#include "../../../../applications/TestMatrixMultiply/sharedLibraries/headers/MatrixWriter.h"

namespace pdb {

using namespace matrix;


void buildPipeline(std::shared_ptr<LogicalPlan> &logicalPlan) {

  //
  auto &computations = logicalPlan->getComputations();
  auto &sources = computations.getAllScanSets();

  //
  if(sources.size() != 2) {
    return;
  }

  //
  auto lhsSource = sources.front();
  auto rhsSource = sources.back();


}

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
  Vector<Handle<Computation>> computations;

  // put them in the list of computations
  computations.push_back(readA);
  computations.push_back(readB);
  computations.push_back(join);
  computations.push_back(myAggregation);
  computations.push_back(myWriter);

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

  auto logicalPlan = std::make_shared<LogicalPlan>(tcap, computations);



}


}
