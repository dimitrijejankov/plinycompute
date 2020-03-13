//
// Created by dimitrije on 2/22/19.
//

#include <gtest/gtest.h>
#include <AtomicComputationList.h>
#include <Parser.h>
#include <PDBPipeNodeBuilder.h>
#include <PDBJoinPhysicalNode.h>

TEST(TestPipeBuilder, Test0) {

  std::string myPlan = "inputDataForSetScanner_0(in0) <= SCAN ('myData', 'A', 'SetScanner_0')\n"
                       "inputDataForSetScanner_1(in1) <= SCAN ('myData', 'A', 'SetScanner_1')\n"
                       "inputDataForSetScanner_2(in2) <= SCAN ('myData', 'A', 'SetScanner_2')\n"
                       "inputDataForSetScanner_3(in3) <= SCAN ('myData', 'A', 'SetScanner_3')\n"
                       "inputDataForSetScanner_4(in4) <= SCAN ('myData', 'A', 'SetScanner_4')\n"
                       "inputDataForSetScanner_5(in5) <= SCAN ('myData', 'A', 'SetScanner_5')\n"
                       "inputDataForSetScanner_6(in6) <= SCAN ('myData', 'A', 'SetScanner_6')\n"
                       "inputDataForSetScanner_7(in7) <= SCAN ('myData', 'A', 'SetScanner_7')\n"
                       "\n"
                       "/* Apply join selection */\n"
                       "OutFor_key_23JoinComp8(in0,OutFor_key_23_8) <= APPLY (inputDataForSetScanner_0(in0), inputDataForSetScanner_0(in0), 'JoinComp_8', 'key_23', [('lambdaType', 'key')])\n"
                       "OutFor_methodCall_7JoinComp8(in0,OutFor_key_23_8,OutFor_methodCall_7_8) <= APPLY (OutFor_key_23JoinComp8(OutFor_key_23_8), OutFor_key_23JoinComp8(in0,OutFor_key_23_8), 'JoinComp_8', 'methodCall_7', [('inputTypeName', 'pdb::matrix_3d::MatrixBlockMeta3D'), ('lambdaType', 'methodCall'), ('methodName', 'right'), ('returnTypeName', 'pdb::matrix_3d::MatrixBlockMeta3D')])\n"
                       "OutFor_key_10JoinComp8(in1,OutFor_key_10_8) <= APPLY (inputDataForSetScanner_1(in1), inputDataForSetScanner_1(in1), 'JoinComp_8', 'key_10', [('lambdaType', 'key')])\n"
                       "OutFor_self_9JoinComp8(in1,OutFor_key_10_8,OutFor_self_9_8) <= APPLY (OutFor_key_10JoinComp8(OutFor_key_10_8), OutFor_key_10JoinComp8(in1,OutFor_key_10_8), 'JoinComp_8', 'self_9', [('lambdaType', 'self')])\n"
                       "OutFor_methodCall_7JoinComp8_hashed(in0,OutFor_methodCall_7_8_hash) <= HASHLEFT (OutFor_methodCall_7JoinComp8(OutFor_methodCall_7_8), OutFor_methodCall_7JoinComp8(in0), 'JoinComp_8', '==_6', [])\n"
                       "OutFor_self_9JoinComp8_hashed(in1,OutFor_self_9_8_hash) <= HASHRIGHT (OutFor_self_9JoinComp8(OutFor_self_9_8), OutFor_self_9JoinComp8(in1), 'JoinComp_8', '==_6', [])\n"
                       "OutForJoinedFor_equals_6JoinComp8(in0,in1) <= JOIN (OutFor_methodCall_7JoinComp8_hashed(OutFor_methodCall_7_8_hash), OutFor_methodCall_7JoinComp8_hashed(in0), OutFor_self_9JoinComp8_hashed(OutFor_self_9_8_hash), OutFor_self_9JoinComp8_hashed(in1), 'JoinComp_8')\n"
                       "LExtractedFor6_key_23JoinComp8(in0,in1,LExtractedFor6_key_23_8) <= APPLY (OutForJoinedFor_equals_6JoinComp8(in0), OutForJoinedFor_equals_6JoinComp8(in0,in1), 'JoinComp_8', 'key_23', [('lambdaType', 'key')])\n"
                       "LExtractedFor6_methodCall_7JoinComp8(in0,in1,LExtractedFor6_key_23_8,LExtractedFor6_methodCall_7_8) <= APPLY (LExtractedFor6_key_23JoinComp8(LExtractedFor6_key_23_8), LExtractedFor6_key_23JoinComp8(in0,in1,LExtractedFor6_key_23_8), 'JoinComp_8', 'methodCall_7', [('inputTypeName', 'pdb::matrix_3d::MatrixBlockMeta3D'), ('lambdaType', 'methodCall'), ('methodName', 'right'), ('returnTypeName', 'pdb::matrix_3d::MatrixBlockMeta3D')])\n"
                       "RExtractedFor6_key_10JoinComp8(in0,in1,LExtractedFor6_key_23_8,LExtractedFor6_methodCall_7_8,RExtractedFor6_key_10_8) <= APPLY (LExtractedFor6_methodCall_7JoinComp8(in1), LExtractedFor6_methodCall_7JoinComp8(in0,in1,LExtractedFor6_key_23_8,LExtractedFor6_methodCall_7_8), 'JoinComp_8', 'key_10', [('lambdaType', 'key')])\n"
                       "RExtractedFor6_self_9JoinComp8(in0,in1,LExtractedFor6_key_23_8,LExtractedFor6_methodCall_7_8,RExtractedFor6_key_10_8,RExtractedFor6_self_9_8) <= APPLY (RExtractedFor6_key_10JoinComp8(RExtractedFor6_key_10_8), RExtractedFor6_key_10JoinComp8(in0,in1,LExtractedFor6_key_23_8,LExtractedFor6_methodCall_7_8,RExtractedFor6_key_10_8), 'JoinComp_8', 'self_9', [('lambdaType', 'self')])\n"
                       "OutFor_OutForJoinedFor_equals_6JoinComp8_BOOL(in0,in1,bool_6_8) <= APPLY (RExtractedFor6_self_9JoinComp8(LExtractedFor6_methodCall_7_8,RExtractedFor6_self_9_8), RExtractedFor6_self_9JoinComp8(in0,in1), 'JoinComp_8', '==_6', [('lambdaType', '==')])\n"
                       "OutFor_OutForJoinedFor_equals_6JoinComp8_FILTERED(in0,in1) <= FILTER (OutFor_OutForJoinedFor_equals_6JoinComp8_BOOL(bool_6_8), OutFor_OutForJoinedFor_equals_6JoinComp8_BOOL(in0,in1), 'JoinComp_8')\n"
                       "LExtractedFor6_key_23JoinComp8(in0,in1,LExtractedFor6_key_23_8) <= APPLY (OutFor_OutForJoinedFor_equals_6JoinComp8_FILTERED(in0), OutFor_OutForJoinedFor_equals_6JoinComp8_FILTERED(in0,in1), 'JoinComp_8', 'key_23', [('lambdaType', 'key')])\n"
                       "OutFor_methodCall_12JoinComp8(in0,in1,LExtractedFor6_key_23_8,OutFor_methodCall_12_8) <= APPLY (LExtractedFor6_key_23JoinComp8(LExtractedFor6_key_23_8), LExtractedFor6_key_23JoinComp8(in0,in1,LExtractedFor6_key_23_8), 'JoinComp_8', 'methodCall_12', [('inputTypeName', 'pdb::matrix_3d::MatrixBlockMeta3D'), ('lambdaType', 'methodCall'), ('methodName', 'below'), ('returnTypeName', 'pdb::matrix_3d::MatrixBlockMeta3D')])\n"
                       "OutFor_key_15JoinComp8(in2,OutFor_key_15_8) <= APPLY (inputDataForSetScanner_2(in2), inputDataForSetScanner_2(in2), 'JoinComp_8', 'key_15', [('lambdaType', 'key')])\n"
                       "OutFor_self_14JoinComp8(in2,OutFor_key_15_8,OutFor_self_14_8) <= APPLY (OutFor_key_15JoinComp8(OutFor_key_15_8), OutFor_key_15JoinComp8(in2,OutFor_key_15_8), 'JoinComp_8', 'self_14', [('lambdaType', 'self')])\n"
                       "OutFor_methodCall_12JoinComp8_hashed(in0,in1,OutFor_methodCall_12_8_hash) <= HASHLEFT (OutFor_methodCall_12JoinComp8(OutFor_methodCall_12_8), OutFor_methodCall_12JoinComp8(in0,in1), 'JoinComp_8', '==_11', [])\n"
                       "OutFor_self_14JoinComp8_hashed(in2,OutFor_self_14_8_hash) <= HASHRIGHT (OutFor_self_14JoinComp8(OutFor_self_14_8), OutFor_self_14JoinComp8(in2), 'JoinComp_8', '==_11', [])\n"
                       "OutForJoinedFor_equals_11JoinComp8(in0,in1,in2) <= JOIN (OutFor_methodCall_12JoinComp8_hashed(OutFor_methodCall_12_8_hash), OutFor_methodCall_12JoinComp8_hashed(in0,in1), OutFor_self_14JoinComp8_hashed(OutFor_self_14_8_hash), OutFor_self_14JoinComp8_hashed(in2), 'JoinComp_8')\n"
                       "LExtractedFor11_key_23JoinComp8(in0,in1,in2,LExtractedFor11_key_23_8) <= APPLY (OutForJoinedFor_equals_11JoinComp8(in0), OutForJoinedFor_equals_11JoinComp8(in0,in1,in2), 'JoinComp_8', 'key_23', [('lambdaType', 'key')])\n"
                       "LExtractedFor11_methodCall_12JoinComp8(in0,in1,in2,LExtractedFor11_key_23_8,LExtractedFor11_methodCall_12_8) <= APPLY (LExtractedFor11_key_23JoinComp8(LExtractedFor11_key_23_8), LExtractedFor11_key_23JoinComp8(in0,in1,in2,LExtractedFor11_key_23_8), 'JoinComp_8', 'methodCall_12', [('inputTypeName', 'pdb::matrix_3d::MatrixBlockMeta3D'), ('lambdaType', 'methodCall'), ('methodName', 'below'), ('returnTypeName', 'pdb::matrix_3d::MatrixBlockMeta3D')])\n"
                       "RExtractedFor11_key_15JoinComp8(in0,in1,in2,LExtractedFor11_key_23_8,LExtractedFor11_methodCall_12_8,RExtractedFor11_key_15_8) <= APPLY (LExtractedFor11_methodCall_12JoinComp8(in2), LExtractedFor11_methodCall_12JoinComp8(in0,in1,in2,LExtractedFor11_key_23_8,LExtractedFor11_methodCall_12_8), 'JoinComp_8', 'key_15', [('lambdaType', 'key')])\n"
                       "RExtractedFor11_self_14JoinComp8(in0,in1,in2,LExtractedFor11_key_23_8,LExtractedFor11_methodCall_12_8,RExtractedFor11_key_15_8,RExtractedFor11_self_14_8) <= APPLY (RExtractedFor11_key_15JoinComp8(RExtractedFor11_key_15_8), RExtractedFor11_key_15JoinComp8(in0,in1,in2,LExtractedFor11_key_23_8,LExtractedFor11_methodCall_12_8,RExtractedFor11_key_15_8), 'JoinComp_8', 'self_14', [('lambdaType', 'self')])\n"
                       "OutFor_OutForJoinedFor_equals_11JoinComp8_BOOL(in0,in1,in2,bool_11_8) <= APPLY (RExtractedFor11_self_14JoinComp8(LExtractedFor11_methodCall_12_8,RExtractedFor11_self_14_8), RExtractedFor11_self_14JoinComp8(in0,in1,in2), 'JoinComp_8', '==_11', [('lambdaType', '==')])\n"
                       "OutFor_OutForJoinedFor_equals_11JoinComp8_FILTERED(in0,in1,in2) <= FILTER (OutFor_OutForJoinedFor_equals_11JoinComp8_BOOL(bool_11_8), OutFor_OutForJoinedFor_equals_11JoinComp8_BOOL(in0,in1,in2), 'JoinComp_8')\n"
                       "LExtractedFor11_key_23JoinComp8(in0,in1,in2,LExtractedFor11_key_23_8) <= APPLY (OutFor_OutForJoinedFor_equals_11JoinComp8_FILTERED(in0), OutFor_OutForJoinedFor_equals_11JoinComp8_FILTERED(in0,in1,in2), 'JoinComp_8', 'key_23', [('lambdaType', 'key')])\n"
                       "OutFor_methodCall_17JoinComp8(in0,in1,in2,LExtractedFor11_key_23_8,OutFor_methodCall_17_8) <= APPLY (LExtractedFor11_key_23JoinComp8(LExtractedFor11_key_23_8), LExtractedFor11_key_23JoinComp8(in0,in1,in2,LExtractedFor11_key_23_8), 'JoinComp_8', 'methodCall_17', [('inputTypeName', 'pdb::matrix_3d::MatrixBlockMeta3D'), ('lambdaType', 'methodCall'), ('methodName', 'right'), ('returnTypeName', 'pdb::matrix_3d::MatrixBlockMeta3D')])\n"
                       "OutFor_key_20JoinComp8(in3,OutFor_key_20_8) <= APPLY (inputDataForSetScanner_3(in3), inputDataForSetScanner_3(in3), 'JoinComp_8', 'key_20', [('lambdaType', 'key')])\n"
                       "OutFor_methodCall_19JoinComp8(in3,OutFor_key_20_8,OutFor_methodCall_19_8) <= APPLY (OutFor_key_20JoinComp8(OutFor_key_20_8), OutFor_key_20JoinComp8(in3,OutFor_key_20_8), 'JoinComp_8', 'methodCall_19', [('inputTypeName', 'pdb::matrix_3d::MatrixBlockMeta3D'), ('lambdaType', 'methodCall'), ('methodName', 'above'), ('returnTypeName', 'pdb::matrix_3d::MatrixBlockMeta3D')])\n"
                       "OutFor_methodCall_17JoinComp8_hashed(in0,in1,in2,OutFor_methodCall_17_8_hash) <= HASHLEFT (OutFor_methodCall_17JoinComp8(OutFor_methodCall_17_8), OutFor_methodCall_17JoinComp8(in0,in1,in2), 'JoinComp_8', '==_16', [])\n"
                       "OutFor_methodCall_19JoinComp8_hashed(in3,OutFor_methodCall_19_8_hash) <= HASHRIGHT (OutFor_methodCall_19JoinComp8(OutFor_methodCall_19_8), OutFor_methodCall_19JoinComp8(in3), 'JoinComp_8', '==_16', [])\n"
                       "OutForJoinedFor_equals_16JoinComp8(in0,in1,in2,in3) <= JOIN (OutFor_methodCall_17JoinComp8_hashed(OutFor_methodCall_17_8_hash), OutFor_methodCall_17JoinComp8_hashed(in0,in1,in2), OutFor_methodCall_19JoinComp8_hashed(OutFor_methodCall_19_8_hash), OutFor_methodCall_19JoinComp8_hashed(in3), 'JoinComp_8')\n"
                       "LExtractedFor16_key_23JoinComp8(in0,in1,in2,in3,LExtractedFor16_key_23_8) <= APPLY (OutForJoinedFor_equals_16JoinComp8(in0), OutForJoinedFor_equals_16JoinComp8(in0,in1,in2,in3), 'JoinComp_8', 'key_23', [('lambdaType', 'key')])\n"
                       "LExtractedFor16_methodCall_17JoinComp8(in0,in1,in2,in3,LExtractedFor16_key_23_8,LExtractedFor16_methodCall_17_8) <= APPLY (LExtractedFor16_key_23JoinComp8(LExtractedFor16_key_23_8), LExtractedFor16_key_23JoinComp8(in0,in1,in2,in3,LExtractedFor16_key_23_8), 'JoinComp_8', 'methodCall_17', [('inputTypeName', 'pdb::matrix_3d::MatrixBlockMeta3D'), ('lambdaType', 'methodCall'), ('methodName', 'right'), ('returnTypeName', 'pdb::matrix_3d::MatrixBlockMeta3D')])\n"
                       "RExtractedFor16_key_20JoinComp8(in0,in1,in2,in3,LExtractedFor16_key_23_8,LExtractedFor16_methodCall_17_8,RExtractedFor16_key_20_8) <= APPLY (LExtractedFor16_methodCall_17JoinComp8(in3), LExtractedFor16_methodCall_17JoinComp8(in0,in1,in2,in3,LExtractedFor16_key_23_8,LExtractedFor16_methodCall_17_8), 'JoinComp_8', 'key_20', [('lambdaType', 'key')])\n"
                       "RExtractedFor16_methodCall_19JoinComp8(in0,in1,in2,in3,LExtractedFor16_key_23_8,LExtractedFor16_methodCall_17_8,RExtractedFor16_key_20_8,RExtractedFor16_methodCall_19_8) <= APPLY (RExtractedFor16_key_20JoinComp8(RExtractedFor16_key_20_8), RExtractedFor16_key_20JoinComp8(in0,in1,in2,in3,LExtractedFor16_key_23_8,LExtractedFor16_methodCall_17_8,RExtractedFor16_key_20_8), 'JoinComp_8', 'methodCall_19', [('inputTypeName', 'pdb::matrix_3d::MatrixBlockMeta3D'), ('lambdaType', 'methodCall'), ('methodName', 'above'), ('returnTypeName', 'pdb::matrix_3d::MatrixBlockMeta3D')])\n"
                       "OutFor_OutForJoinedFor_equals_16JoinComp8_BOOL(in0,in1,in2,in3,bool_16_8) <= APPLY (RExtractedFor16_methodCall_19JoinComp8(LExtractedFor16_methodCall_17_8,RExtractedFor16_methodCall_19_8), RExtractedFor16_methodCall_19JoinComp8(in0,in1,in2,in3), 'JoinComp_8', '==_16', [('lambdaType', '==')])\n"
                       "OutFor_OutForJoinedFor_equals_16JoinComp8_FILTERED(in0,in1,in2,in3) <= FILTER (OutFor_OutForJoinedFor_equals_16JoinComp8_BOOL(bool_16_8), OutFor_OutForJoinedFor_equals_16JoinComp8_BOOL(in0,in1,in2,in3), 'JoinComp_8')\n"
                       "LExtractedFor16_key_23JoinComp8(in0,in1,in2,in3,LExtractedFor16_key_23_8) <= APPLY (OutFor_OutForJoinedFor_equals_16JoinComp8_FILTERED(in0), OutFor_OutForJoinedFor_equals_16JoinComp8_FILTERED(in0,in1,in2,in3), 'JoinComp_8', 'key_23', [('lambdaType', 'key')])\n"
                       "OutFor_methodCall_22JoinComp8(in0,in1,in2,in3,LExtractedFor16_key_23_8,OutFor_methodCall_22_8) <= APPLY (LExtractedFor16_key_23JoinComp8(LExtractedFor16_key_23_8), LExtractedFor16_key_23JoinComp8(in0,in1,in2,in3,LExtractedFor16_key_23_8), 'JoinComp_8', 'methodCall_22', [('inputTypeName', 'pdb::matrix_3d::MatrixBlockMeta3D'), ('lambdaType', 'methodCall'), ('methodName', 'back'), ('returnTypeName', 'pdb::matrix_3d::MatrixBlockMeta3D')])\n"
                       "OutFor_key_38JoinComp8(in4,OutFor_key_38_8) <= APPLY (inputDataForSetScanner_4(in4), inputDataForSetScanner_4(in4), 'JoinComp_8', 'key_38', [('lambdaType', 'key')])\n"
                       "OutFor_self_24JoinComp8(in4,OutFor_key_38_8,OutFor_self_24_8) <= APPLY (OutFor_key_38JoinComp8(OutFor_key_38_8), OutFor_key_38JoinComp8(in4,OutFor_key_38_8), 'JoinComp_8', 'self_24', [('lambdaType', 'self')])\n"
                       "OutFor_methodCall_22JoinComp8_hashed(in0,in1,in2,in3,OutFor_methodCall_22_8_hash) <= HASHLEFT (OutFor_methodCall_22JoinComp8(OutFor_methodCall_22_8), OutFor_methodCall_22JoinComp8(in0,in1,in2,in3), 'JoinComp_8', '==_21', [])\n"
                       "OutFor_self_24JoinComp8_hashed(in4,OutFor_self_24_8_hash) <= HASHRIGHT (OutFor_self_24JoinComp8(OutFor_self_24_8), OutFor_self_24JoinComp8(in4), 'JoinComp_8', '==_21', [])\n"
                       "OutForJoinedFor_equals_21JoinComp8(in0,in1,in2,in3,in4) <= JOIN (OutFor_methodCall_22JoinComp8_hashed(OutFor_methodCall_22_8_hash), OutFor_methodCall_22JoinComp8_hashed(in0,in1,in2,in3), OutFor_self_24JoinComp8_hashed(OutFor_self_24_8_hash), OutFor_self_24JoinComp8_hashed(in4), 'JoinComp_8')\n"
                       "LExtractedFor21_key_23JoinComp8(in0,in1,in2,in3,in4,LExtractedFor21_key_23_8) <= APPLY (OutForJoinedFor_equals_21JoinComp8(in0), OutForJoinedFor_equals_21JoinComp8(in0,in1,in2,in3,in4), 'JoinComp_8', 'key_23', [('lambdaType', 'key')])\n"
                       "LExtractedFor21_methodCall_22JoinComp8(in0,in1,in2,in3,in4,LExtractedFor21_key_23_8,LExtractedFor21_methodCall_22_8) <= APPLY (LExtractedFor21_key_23JoinComp8(LExtractedFor21_key_23_8), LExtractedFor21_key_23JoinComp8(in0,in1,in2,in3,in4,LExtractedFor21_key_23_8), 'JoinComp_8', 'methodCall_22', [('inputTypeName', 'pdb::matrix_3d::MatrixBlockMeta3D'), ('lambdaType', 'methodCall'), ('methodName', 'back'), ('returnTypeName', 'pdb::matrix_3d::MatrixBlockMeta3D')])\n"
                       "RExtractedFor21_key_38JoinComp8(in0,in1,in2,in3,in4,LExtractedFor21_key_23_8,LExtractedFor21_methodCall_22_8,RExtractedFor21_key_38_8) <= APPLY (LExtractedFor21_methodCall_22JoinComp8(in4), LExtractedFor21_methodCall_22JoinComp8(in0,in1,in2,in3,in4,LExtractedFor21_key_23_8,LExtractedFor21_methodCall_22_8), 'JoinComp_8', 'key_38', [('lambdaType', 'key')])\n"
                       "RExtractedFor21_self_24JoinComp8(in0,in1,in2,in3,in4,LExtractedFor21_key_23_8,LExtractedFor21_methodCall_22_8,RExtractedFor21_key_38_8,RExtractedFor21_self_24_8) <= APPLY (RExtractedFor21_key_38JoinComp8(RExtractedFor21_key_38_8), RExtractedFor21_key_38JoinComp8(in0,in1,in2,in3,in4,LExtractedFor21_key_23_8,LExtractedFor21_methodCall_22_8,RExtractedFor21_key_38_8), 'JoinComp_8', 'self_24', [('lambdaType', 'self')])\n"
                       "OutFor_OutForJoinedFor_equals_21JoinComp8_BOOL(in0,in1,in2,in3,in4,bool_21_8) <= APPLY (RExtractedFor21_self_24JoinComp8(LExtractedFor21_methodCall_22_8,RExtractedFor21_self_24_8), RExtractedFor21_self_24JoinComp8(in0,in1,in2,in3,in4), 'JoinComp_8', '==_21', [('lambdaType', '==')])\n"
                       "OutFor_OutForJoinedFor_equals_21JoinComp8_FILTERED(in0,in1,in2,in3,in4) <= FILTER (OutFor_OutForJoinedFor_equals_21JoinComp8_BOOL(bool_21_8), OutFor_OutForJoinedFor_equals_21JoinComp8_BOOL(in0,in1,in2,in3,in4), 'JoinComp_8')\n"
                       "RExtractedFor21_key_38JoinComp8(in0,in1,in2,in3,in4,RExtractedFor21_key_38_8) <= APPLY (OutFor_OutForJoinedFor_equals_21JoinComp8_FILTERED(in4), OutFor_OutForJoinedFor_equals_21JoinComp8_FILTERED(in0,in1,in2,in3,in4), 'JoinComp_8', 'key_38', [('lambdaType', 'key')])\n"
                       "OutFor_methodCall_27JoinComp8(in0,in1,in2,in3,in4,RExtractedFor21_key_38_8,OutFor_methodCall_27_8) <= APPLY (RExtractedFor21_key_38JoinComp8(RExtractedFor21_key_38_8), RExtractedFor21_key_38JoinComp8(in0,in1,in2,in3,in4,RExtractedFor21_key_38_8), 'JoinComp_8', 'methodCall_27', [('inputTypeName', 'pdb::matrix_3d::MatrixBlockMeta3D'), ('lambdaType', 'methodCall'), ('methodName', 'right'), ('returnTypeName', 'pdb::matrix_3d::MatrixBlockMeta3D')])\n"
                       "OutFor_key_30JoinComp8(in5,OutFor_key_30_8) <= APPLY (inputDataForSetScanner_5(in5), inputDataForSetScanner_5(in5), 'JoinComp_8', 'key_30', [('lambdaType', 'key')])\n"
                       "OutFor_self_29JoinComp8(in5,OutFor_key_30_8,OutFor_self_29_8) <= APPLY (OutFor_key_30JoinComp8(OutFor_key_30_8), OutFor_key_30JoinComp8(in5,OutFor_key_30_8), 'JoinComp_8', 'self_29', [('lambdaType', 'self')])\n"
                       "OutFor_methodCall_27JoinComp8_hashed(in0,in1,in2,in3,in4,OutFor_methodCall_27_8_hash) <= HASHLEFT (OutFor_methodCall_27JoinComp8(OutFor_methodCall_27_8), OutFor_methodCall_27JoinComp8(in0,in1,in2,in3,in4), 'JoinComp_8', '==_26', [])\n"
                       "OutFor_self_29JoinComp8_hashed(in5,OutFor_self_29_8_hash) <= HASHRIGHT (OutFor_self_29JoinComp8(OutFor_self_29_8), OutFor_self_29JoinComp8(in5), 'JoinComp_8', '==_26', [])\n"
                       "OutForJoinedFor_equals_26JoinComp8(in0,in1,in2,in3,in4,in5) <= JOIN (OutFor_methodCall_27JoinComp8_hashed(OutFor_methodCall_27_8_hash), OutFor_methodCall_27JoinComp8_hashed(in0,in1,in2,in3,in4), OutFor_self_29JoinComp8_hashed(OutFor_self_29_8_hash), OutFor_self_29JoinComp8_hashed(in5), 'JoinComp_8')\n"
                       "LExtractedFor26_key_38JoinComp8(in0,in1,in2,in3,in4,in5,LExtractedFor26_key_38_8) <= APPLY (OutForJoinedFor_equals_26JoinComp8(in4), OutForJoinedFor_equals_26JoinComp8(in0,in1,in2,in3,in4,in5), 'JoinComp_8', 'key_38', [('lambdaType', 'key')])\n"
                       "LExtractedFor26_methodCall_27JoinComp8(in0,in1,in2,in3,in4,in5,LExtractedFor26_key_38_8,LExtractedFor26_methodCall_27_8) <= APPLY (LExtractedFor26_key_38JoinComp8(LExtractedFor26_key_38_8), LExtractedFor26_key_38JoinComp8(in0,in1,in2,in3,in4,in5,LExtractedFor26_key_38_8), 'JoinComp_8', 'methodCall_27', [('inputTypeName', 'pdb::matrix_3d::MatrixBlockMeta3D'), ('lambdaType', 'methodCall'), ('methodName', 'right'), ('returnTypeName', 'pdb::matrix_3d::MatrixBlockMeta3D')])\n"
                       "RExtractedFor26_key_30JoinComp8(in0,in1,in2,in3,in4,in5,LExtractedFor26_key_38_8,LExtractedFor26_methodCall_27_8,RExtractedFor26_key_30_8) <= APPLY (LExtractedFor26_methodCall_27JoinComp8(in5), LExtractedFor26_methodCall_27JoinComp8(in0,in1,in2,in3,in4,in5,LExtractedFor26_key_38_8,LExtractedFor26_methodCall_27_8), 'JoinComp_8', 'key_30', [('lambdaType', 'key')])\n"
                       "RExtractedFor26_self_29JoinComp8(in0,in1,in2,in3,in4,in5,LExtractedFor26_key_38_8,LExtractedFor26_methodCall_27_8,RExtractedFor26_key_30_8,RExtractedFor26_self_29_8) <= APPLY (RExtractedFor26_key_30JoinComp8(RExtractedFor26_key_30_8), RExtractedFor26_key_30JoinComp8(in0,in1,in2,in3,in4,in5,LExtractedFor26_key_38_8,LExtractedFor26_methodCall_27_8,RExtractedFor26_key_30_8), 'JoinComp_8', 'self_29', [('lambdaType', 'self')])\n"
                       "OutFor_OutForJoinedFor_equals_26JoinComp8_BOOL(in0,in1,in2,in3,in4,in5,bool_26_8) <= APPLY (RExtractedFor26_self_29JoinComp8(LExtractedFor26_methodCall_27_8,RExtractedFor26_self_29_8), RExtractedFor26_self_29JoinComp8(in0,in1,in2,in3,in4,in5), 'JoinComp_8', '==_26', [('lambdaType', '==')])\n"
                       "OutFor_OutForJoinedFor_equals_26JoinComp8_FILTERED(in0,in1,in2,in3,in4,in5) <= FILTER (OutFor_OutForJoinedFor_equals_26JoinComp8_BOOL(bool_26_8), OutFor_OutForJoinedFor_equals_26JoinComp8_BOOL(in0,in1,in2,in3,in4,in5), 'JoinComp_8')\n"
                       "LExtractedFor26_key_38JoinComp8(in0,in1,in2,in3,in4,in5,LExtractedFor26_key_38_8) <= APPLY (OutFor_OutForJoinedFor_equals_26JoinComp8_FILTERED(in4), OutFor_OutForJoinedFor_equals_26JoinComp8_FILTERED(in0,in1,in2,in3,in4,in5), 'JoinComp_8', 'key_38', [('lambdaType', 'key')])\n"
                       "OutFor_methodCall_32JoinComp8(in0,in1,in2,in3,in4,in5,LExtractedFor26_key_38_8,OutFor_methodCall_32_8) <= APPLY (LExtractedFor26_key_38JoinComp8(LExtractedFor26_key_38_8), LExtractedFor26_key_38JoinComp8(in0,in1,in2,in3,in4,in5,LExtractedFor26_key_38_8), 'JoinComp_8', 'methodCall_32', [('inputTypeName', 'pdb::matrix_3d::MatrixBlockMeta3D'), ('lambdaType', 'methodCall'), ('methodName', 'below'), ('returnTypeName', 'pdb::matrix_3d::MatrixBlockMeta3D')])\n"
                       "OutFor_key_35JoinComp8(in6,OutFor_key_35_8) <= APPLY (inputDataForSetScanner_6(in6), inputDataForSetScanner_6(in6), 'JoinComp_8', 'key_35', [('lambdaType', 'key')])\n"
                       "OutFor_self_34JoinComp8(in6,OutFor_key_35_8,OutFor_self_34_8) <= APPLY (OutFor_key_35JoinComp8(OutFor_key_35_8), OutFor_key_35JoinComp8(in6,OutFor_key_35_8), 'JoinComp_8', 'self_34', [('lambdaType', 'self')])\n"
                       "OutFor_methodCall_32JoinComp8_hashed(in0,in1,in2,in3,in4,in5,OutFor_methodCall_32_8_hash) <= HASHLEFT (OutFor_methodCall_32JoinComp8(OutFor_methodCall_32_8), OutFor_methodCall_32JoinComp8(in0,in1,in2,in3,in4,in5), 'JoinComp_8', '==_31', [])\n"
                       "OutFor_self_34JoinComp8_hashed(in6,OutFor_self_34_8_hash) <= HASHRIGHT (OutFor_self_34JoinComp8(OutFor_self_34_8), OutFor_self_34JoinComp8(in6), 'JoinComp_8', '==_31', [])\n"
                       "OutForJoinedFor_equals_31JoinComp8(in0,in1,in2,in3,in4,in5,in6) <= JOIN (OutFor_methodCall_32JoinComp8_hashed(OutFor_methodCall_32_8_hash), OutFor_methodCall_32JoinComp8_hashed(in0,in1,in2,in3,in4,in5), OutFor_self_34JoinComp8_hashed(OutFor_self_34_8_hash), OutFor_self_34JoinComp8_hashed(in6), 'JoinComp_8')\n"
                       "LExtractedFor31_key_38JoinComp8(in0,in1,in2,in3,in4,in5,in6,LExtractedFor31_key_38_8) <= APPLY (OutForJoinedFor_equals_31JoinComp8(in4), OutForJoinedFor_equals_31JoinComp8(in0,in1,in2,in3,in4,in5,in6), 'JoinComp_8', 'key_38', [('lambdaType', 'key')])\n"
                       "LExtractedFor31_methodCall_32JoinComp8(in0,in1,in2,in3,in4,in5,in6,LExtractedFor31_key_38_8,LExtractedFor31_methodCall_32_8) <= APPLY (LExtractedFor31_key_38JoinComp8(LExtractedFor31_key_38_8), LExtractedFor31_key_38JoinComp8(in0,in1,in2,in3,in4,in5,in6,LExtractedFor31_key_38_8), 'JoinComp_8', 'methodCall_32', [('inputTypeName', 'pdb::matrix_3d::MatrixBlockMeta3D'), ('lambdaType', 'methodCall'), ('methodName', 'below'), ('returnTypeName', 'pdb::matrix_3d::MatrixBlockMeta3D')])\n"
                       "RExtractedFor31_key_35JoinComp8(in0,in1,in2,in3,in4,in5,in6,LExtractedFor31_key_38_8,LExtractedFor31_methodCall_32_8,RExtractedFor31_key_35_8) <= APPLY (LExtractedFor31_methodCall_32JoinComp8(in6), LExtractedFor31_methodCall_32JoinComp8(in0,in1,in2,in3,in4,in5,in6,LExtractedFor31_key_38_8,LExtractedFor31_methodCall_32_8), 'JoinComp_8', 'key_35', [('lambdaType', 'key')])\n"
                       "RExtractedFor31_self_34JoinComp8(in0,in1,in2,in3,in4,in5,in6,LExtractedFor31_key_38_8,LExtractedFor31_methodCall_32_8,RExtractedFor31_key_35_8,RExtractedFor31_self_34_8) <= APPLY (RExtractedFor31_key_35JoinComp8(RExtractedFor31_key_35_8), RExtractedFor31_key_35JoinComp8(in0,in1,in2,in3,in4,in5,in6,LExtractedFor31_key_38_8,LExtractedFor31_methodCall_32_8,RExtractedFor31_key_35_8), 'JoinComp_8', 'self_34', [('lambdaType', 'self')])\n"
                       "OutFor_OutForJoinedFor_equals_31JoinComp8_BOOL(in0,in1,in2,in3,in4,in5,in6,bool_31_8) <= APPLY (RExtractedFor31_self_34JoinComp8(LExtractedFor31_methodCall_32_8,RExtractedFor31_self_34_8), RExtractedFor31_self_34JoinComp8(in0,in1,in2,in3,in4,in5,in6), 'JoinComp_8', '==_31', [('lambdaType', '==')])\n"
                       "OutFor_OutForJoinedFor_equals_31JoinComp8_FILTERED(in0,in1,in2,in3,in4,in5,in6) <= FILTER (OutFor_OutForJoinedFor_equals_31JoinComp8_BOOL(bool_31_8), OutFor_OutForJoinedFor_equals_31JoinComp8_BOOL(in0,in1,in2,in3,in4,in5,in6), 'JoinComp_8')\n"
                       "LExtractedFor31_key_38JoinComp8(in0,in1,in2,in3,in4,in5,in6,LExtractedFor31_key_38_8) <= APPLY (OutFor_OutForJoinedFor_equals_31JoinComp8_FILTERED(in4), OutFor_OutForJoinedFor_equals_31JoinComp8_FILTERED(in0,in1,in2,in3,in4,in5,in6), 'JoinComp_8', 'key_38', [('lambdaType', 'key')])\n"
                       "OutFor_methodCall_37JoinComp8(in0,in1,in2,in3,in4,in5,in6,LExtractedFor31_key_38_8,OutFor_methodCall_37_8) <= APPLY (LExtractedFor31_key_38JoinComp8(LExtractedFor31_key_38_8), LExtractedFor31_key_38JoinComp8(in0,in1,in2,in3,in4,in5,in6,LExtractedFor31_key_38_8), 'JoinComp_8', 'methodCall_37', [('inputTypeName', 'pdb::matrix_3d::MatrixBlockMeta3D'), ('lambdaType', 'methodCall'), ('methodName', 'right'), ('returnTypeName', 'pdb::matrix_3d::MatrixBlockMeta3D')])\n"
                       "OutFor_key_40JoinComp8(in7,OutFor_key_40_8) <= APPLY (inputDataForSetScanner_7(in7), inputDataForSetScanner_7(in7), 'JoinComp_8', 'key_40', [('lambdaType', 'key')])\n"
                       "OutFor_methodCall_39JoinComp8(in7,OutFor_key_40_8,OutFor_methodCall_39_8) <= APPLY (OutFor_key_40JoinComp8(OutFor_key_40_8), OutFor_key_40JoinComp8(in7,OutFor_key_40_8), 'JoinComp_8', 'methodCall_39', [('inputTypeName', 'pdb::matrix_3d::MatrixBlockMeta3D'), ('lambdaType', 'methodCall'), ('methodName', 'above'), ('returnTypeName', 'pdb::matrix_3d::MatrixBlockMeta3D')])\n"
                       "OutFor_methodCall_37JoinComp8_hashed(in0,in1,in2,in3,in4,in5,in6,OutFor_methodCall_37_8_hash) <= HASHLEFT (OutFor_methodCall_37JoinComp8(OutFor_methodCall_37_8), OutFor_methodCall_37JoinComp8(in0,in1,in2,in3,in4,in5,in6), 'JoinComp_8', '==_36', [])\n"
                       "OutFor_methodCall_39JoinComp8_hashed(in7,OutFor_methodCall_39_8_hash) <= HASHRIGHT (OutFor_methodCall_39JoinComp8(OutFor_methodCall_39_8), OutFor_methodCall_39JoinComp8(in7), 'JoinComp_8', '==_36', [])\n"
                       "OutForJoinedFor_equals_36JoinComp8(in0,in1,in2,in3,in4,in5,in6,in7) <= JOIN (OutFor_methodCall_37JoinComp8_hashed(OutFor_methodCall_37_8_hash), OutFor_methodCall_37JoinComp8_hashed(in0,in1,in2,in3,in4,in5,in6), OutFor_methodCall_39JoinComp8_hashed(OutFor_methodCall_39_8_hash), OutFor_methodCall_39JoinComp8_hashed(in7), 'JoinComp_8')\n"
                       "LExtractedFor36_key_38JoinComp8(in0,in1,in2,in3,in4,in5,in6,in7,LExtractedFor36_key_38_8) <= APPLY (OutForJoinedFor_equals_36JoinComp8(in4), OutForJoinedFor_equals_36JoinComp8(in0,in1,in2,in3,in4,in5,in6,in7), 'JoinComp_8', 'key_38', [('lambdaType', 'key')])\n"
                       "LExtractedFor36_methodCall_37JoinComp8(in0,in1,in2,in3,in4,in5,in6,in7,LExtractedFor36_key_38_8,LExtractedFor36_methodCall_37_8) <= APPLY (LExtractedFor36_key_38JoinComp8(LExtractedFor36_key_38_8), LExtractedFor36_key_38JoinComp8(in0,in1,in2,in3,in4,in5,in6,in7,LExtractedFor36_key_38_8), 'JoinComp_8', 'methodCall_37', [('inputTypeName', 'pdb::matrix_3d::MatrixBlockMeta3D'), ('lambdaType', 'methodCall'), ('methodName', 'right'), ('returnTypeName', 'pdb::matrix_3d::MatrixBlockMeta3D')])\n"
                       "RExtractedFor36_key_40JoinComp8(in0,in1,in2,in3,in4,in5,in6,in7,LExtractedFor36_key_38_8,LExtractedFor36_methodCall_37_8,RExtractedFor36_key_40_8) <= APPLY (LExtractedFor36_methodCall_37JoinComp8(in7), LExtractedFor36_methodCall_37JoinComp8(in0,in1,in2,in3,in4,in5,in6,in7,LExtractedFor36_key_38_8,LExtractedFor36_methodCall_37_8), 'JoinComp_8', 'key_40', [('lambdaType', 'key')])\n"
                       "RExtractedFor36_methodCall_39JoinComp8(in0,in1,in2,in3,in4,in5,in6,in7,LExtractedFor36_key_38_8,LExtractedFor36_methodCall_37_8,RExtractedFor36_key_40_8,RExtractedFor36_methodCall_39_8) <= APPLY (RExtractedFor36_key_40JoinComp8(RExtractedFor36_key_40_8), RExtractedFor36_key_40JoinComp8(in0,in1,in2,in3,in4,in5,in6,in7,LExtractedFor36_key_38_8,LExtractedFor36_methodCall_37_8,RExtractedFor36_key_40_8), 'JoinComp_8', 'methodCall_39', [('inputTypeName', 'pdb::matrix_3d::MatrixBlockMeta3D'), ('lambdaType', 'methodCall'), ('methodName', 'above'), ('returnTypeName', 'pdb::matrix_3d::MatrixBlockMeta3D')])\n"
                       "OutFor_OutForJoinedFor_equals_36JoinComp8_BOOL(in0,in1,in2,in3,in4,in5,in6,in7,bool_36_8) <= APPLY (RExtractedFor36_methodCall_39JoinComp8(LExtractedFor36_methodCall_37_8,RExtractedFor36_methodCall_39_8), RExtractedFor36_methodCall_39JoinComp8(in0,in1,in2,in3,in4,in5,in6,in7), 'JoinComp_8', '==_36', [('lambdaType', '==')])\n"
                       "OutFor_OutForJoinedFor_equals_36JoinComp8_FILTERED(in0,in1,in2,in3,in4,in5,in6,in7) <= FILTER (OutFor_OutForJoinedFor_equals_36JoinComp8_BOOL(bool_36_8), OutFor_OutForJoinedFor_equals_36JoinComp8_BOOL(in0,in1,in2,in3,in4,in5,in6,in7), 'JoinComp_8')\n"
                       "\n"
                       "/* Apply join projection*/\n"
                       "OutFor_key_43JoinComp8(in0,in1,in2,in3,in4,in5,in6,in7,OutFor_key_43_8) <= APPLY (OutFor_OutForJoinedFor_equals_36JoinComp8_FILTERED(in0), OutFor_OutForJoinedFor_equals_36JoinComp8_FILTERED(in0,in1,in2,in3,in4,in5,in6,in7), 'JoinComp_8', 'key_43', [('lambdaType', 'key')])\n"
                       "OutFor_key_44JoinComp8(in0,in1,in2,in3,in4,in5,in6,in7,OutFor_key_43_8,OutFor_key_44_8) <= APPLY (OutFor_key_43JoinComp8(in1), OutFor_key_43JoinComp8(in0,in1,in2,in3,in4,in5,in6,in7,OutFor_key_43_8), 'JoinComp_8', 'key_44', [('lambdaType', 'key')])\n"
                       "OutFor_key_45JoinComp8(in0,in1,in2,in3,in4,in5,in6,in7,OutFor_key_43_8,OutFor_key_44_8,OutFor_key_45_8) <= APPLY (OutFor_key_44JoinComp8(in2), OutFor_key_44JoinComp8(in0,in1,in2,in3,in4,in5,in6,in7,OutFor_key_43_8,OutFor_key_44_8), 'JoinComp_8', 'key_45', [('lambdaType', 'key')])\n"
                       "OutFor_key_46JoinComp8(in0,in1,in2,in3,in4,in5,in6,in7,OutFor_key_43_8,OutFor_key_44_8,OutFor_key_45_8,OutFor_key_46_8) <= APPLY (OutFor_key_45JoinComp8(in3), OutFor_key_45JoinComp8(in0,in1,in2,in3,in4,in5,in6,in7,OutFor_key_43_8,OutFor_key_44_8,OutFor_key_45_8), 'JoinComp_8', 'key_46', [('lambdaType', 'key')])\n"
                       "OutFor_key_47JoinComp8(in0,in1,in2,in3,in4,in5,in6,in7,OutFor_key_43_8,OutFor_key_44_8,OutFor_key_45_8,OutFor_key_46_8,OutFor_key_47_8) <= APPLY (OutFor_key_46JoinComp8(in4), OutFor_key_46JoinComp8(in0,in1,in2,in3,in4,in5,in6,in7,OutFor_key_43_8,OutFor_key_44_8,OutFor_key_45_8,OutFor_key_46_8), 'JoinComp_8', 'key_47', [('lambdaType', 'key')])\n"
                       "OutFor_key_48JoinComp8(in0,in1,in2,in3,in4,in5,in6,in7,OutFor_key_43_8,OutFor_key_44_8,OutFor_key_45_8,OutFor_key_46_8,OutFor_key_47_8,OutFor_key_48_8) <= APPLY (OutFor_key_47JoinComp8(in5), OutFor_key_47JoinComp8(in0,in1,in2,in3,in4,in5,in6,in7,OutFor_key_43_8,OutFor_key_44_8,OutFor_key_45_8,OutFor_key_46_8,OutFor_key_47_8), 'JoinComp_8', 'key_48', [('lambdaType', 'key')])\n"
                       "OutFor_key_49JoinComp8(in0,in1,in2,in3,in4,in5,in6,in7,OutFor_key_43_8,OutFor_key_44_8,OutFor_key_45_8,OutFor_key_46_8,OutFor_key_47_8,OutFor_key_48_8,OutFor_key_49_8) <= APPLY (OutFor_key_48JoinComp8(in6), OutFor_key_48JoinComp8(in0,in1,in2,in3,in4,in5,in6,in7,OutFor_key_43_8,OutFor_key_44_8,OutFor_key_45_8,OutFor_key_46_8,OutFor_key_47_8,OutFor_key_48_8), 'JoinComp_8', 'key_49', [('lambdaType', 'key')])\n"
                       "OutFor_key_50JoinComp8(in0,in1,in2,in3,in4,in5,in6,in7,OutFor_key_43_8,OutFor_key_44_8,OutFor_key_45_8,OutFor_key_46_8,OutFor_key_47_8,OutFor_key_48_8,OutFor_key_49_8,OutFor_key_50_8) <= APPLY (OutFor_key_49JoinComp8(in7), OutFor_key_49JoinComp8(in0,in1,in2,in3,in4,in5,in6,in7,OutFor_key_43_8,OutFor_key_44_8,OutFor_key_45_8,OutFor_key_46_8,OutFor_key_47_8,OutFor_key_48_8,OutFor_key_49_8), 'JoinComp_8', 'key_50', [('lambdaType', 'key')])\n"
                       "OutFor_native_lambda_42JoinComp8(in0,in1,in2,in3,in4,in5,in6,in7,OutFor_key_43_8,OutFor_key_44_8,OutFor_key_45_8,OutFor_key_46_8,OutFor_key_47_8,OutFor_key_48_8,OutFor_key_49_8,OutFor_key_50_8,OutFor_native_lambda_42_8) <= APPLY (OutFor_key_50JoinComp8(OutFor_key_43_8,OutFor_key_44_8,OutFor_key_45_8,OutFor_key_46_8,OutFor_key_47_8,OutFor_key_48_8,OutFor_key_49_8,OutFor_key_50_8), OutFor_key_50JoinComp8(in0,in1,in2,in3,in4,in5,in6,in7,OutFor_key_43_8,OutFor_key_44_8,OutFor_key_45_8,OutFor_key_46_8,OutFor_key_47_8,OutFor_key_48_8,OutFor_key_49_8,OutFor_key_50_8), 'JoinComp_8', 'native_lambda_42', [('lambdaType', 'native_lambda')])\n"
                       "OutFor_value_52JoinComp8(in0,in1,in2,in3,in4,in5,in6,in7,OutFor_key_43_8,OutFor_key_44_8,OutFor_key_45_8,OutFor_key_46_8,OutFor_key_47_8,OutFor_key_48_8,OutFor_key_49_8,OutFor_key_50_8,OutFor_native_lambda_42_8,OutFor_value_52_8) <= APPLY (OutFor_native_lambda_42JoinComp8(in0), OutFor_native_lambda_42JoinComp8(in0,in1,in2,in3,in4,in5,in6,in7,OutFor_key_43_8,OutFor_key_44_8,OutFor_key_45_8,OutFor_key_46_8,OutFor_key_47_8,OutFor_key_48_8,OutFor_key_49_8,OutFor_key_50_8,OutFor_native_lambda_42_8), 'JoinComp_8', 'value_52', [('lambdaType', 'value')])\n"
                       "OutFor_value_53JoinComp8(in0,in1,in2,in3,in4,in5,in6,in7,OutFor_key_43_8,OutFor_key_44_8,OutFor_key_45_8,OutFor_key_46_8,OutFor_key_47_8,OutFor_key_48_8,OutFor_key_49_8,OutFor_key_50_8,OutFor_native_lambda_42_8,OutFor_value_52_8,OutFor_value_53_8) <= APPLY (OutFor_value_52JoinComp8(in1), OutFor_value_52JoinComp8(in0,in1,in2,in3,in4,in5,in6,in7,OutFor_key_43_8,OutFor_key_44_8,OutFor_key_45_8,OutFor_key_46_8,OutFor_key_47_8,OutFor_key_48_8,OutFor_key_49_8,OutFor_key_50_8,OutFor_native_lambda_42_8,OutFor_value_52_8), 'JoinComp_8', 'value_53', [('lambdaType', 'value')])\n"
                       "OutFor_value_54JoinComp8(in0,in1,in2,in3,in4,in5,in6,in7,OutFor_key_43_8,OutFor_key_44_8,OutFor_key_45_8,OutFor_key_46_8,OutFor_key_47_8,OutFor_key_48_8,OutFor_key_49_8,OutFor_key_50_8,OutFor_native_lambda_42_8,OutFor_value_52_8,OutFor_value_53_8,OutFor_value_54_8) <= APPLY (OutFor_value_53JoinComp8(in2), OutFor_value_53JoinComp8(in0,in1,in2,in3,in4,in5,in6,in7,OutFor_key_43_8,OutFor_key_44_8,OutFor_key_45_8,OutFor_key_46_8,OutFor_key_47_8,OutFor_key_48_8,OutFor_key_49_8,OutFor_key_50_8,OutFor_native_lambda_42_8,OutFor_value_52_8,OutFor_value_53_8), 'JoinComp_8', 'value_54', [('lambdaType', 'value')])\n"
                       "OutFor_value_55JoinComp8(in0,in1,in2,in3,in4,in5,in6,in7,OutFor_key_43_8,OutFor_key_44_8,OutFor_key_45_8,OutFor_key_46_8,OutFor_key_47_8,OutFor_key_48_8,OutFor_key_49_8,OutFor_key_50_8,OutFor_native_lambda_42_8,OutFor_value_52_8,OutFor_value_53_8,OutFor_value_54_8,OutFor_value_55_8) <= APPLY (OutFor_value_54JoinComp8(in3), OutFor_value_54JoinComp8(in0,in1,in2,in3,in4,in5,in6,in7,OutFor_key_43_8,OutFor_key_44_8,OutFor_key_45_8,OutFor_key_46_8,OutFor_key_47_8,OutFor_key_48_8,OutFor_key_49_8,OutFor_key_50_8,OutFor_native_lambda_42_8,OutFor_value_52_8,OutFor_value_53_8,OutFor_value_54_8), 'JoinComp_8', 'value_55', [('lambdaType', 'value')])\n"
                       "OutFor_value_56JoinComp8(in0,in1,in2,in3,in4,in5,in6,in7,OutFor_key_43_8,OutFor_key_44_8,OutFor_key_45_8,OutFor_key_46_8,OutFor_key_47_8,OutFor_key_48_8,OutFor_key_49_8,OutFor_key_50_8,OutFor_native_lambda_42_8,OutFor_value_52_8,OutFor_value_53_8,OutFor_value_54_8,OutFor_value_55_8,OutFor_value_56_8) <= APPLY (OutFor_value_55JoinComp8(in4), OutFor_value_55JoinComp8(in0,in1,in2,in3,in4,in5,in6,in7,OutFor_key_43_8,OutFor_key_44_8,OutFor_key_45_8,OutFor_key_46_8,OutFor_key_47_8,OutFor_key_48_8,OutFor_key_49_8,OutFor_key_50_8,OutFor_native_lambda_42_8,OutFor_value_52_8,OutFor_value_53_8,OutFor_value_54_8,OutFor_value_55_8), 'JoinComp_8', 'value_56', [('lambdaType', 'value')])\n"
                       "OutFor_value_57JoinComp8(in0,in1,in2,in3,in4,in5,in6,in7,OutFor_key_43_8,OutFor_key_44_8,OutFor_key_45_8,OutFor_key_46_8,OutFor_key_47_8,OutFor_key_48_8,OutFor_key_49_8,OutFor_key_50_8,OutFor_native_lambda_42_8,OutFor_value_52_8,OutFor_value_53_8,OutFor_value_54_8,OutFor_value_55_8,OutFor_value_56_8,OutFor_value_57_8) <= APPLY (OutFor_value_56JoinComp8(in5), OutFor_value_56JoinComp8(in0,in1,in2,in3,in4,in5,in6,in7,OutFor_key_43_8,OutFor_key_44_8,OutFor_key_45_8,OutFor_key_46_8,OutFor_key_47_8,OutFor_key_48_8,OutFor_key_49_8,OutFor_key_50_8,OutFor_native_lambda_42_8,OutFor_value_52_8,OutFor_value_53_8,OutFor_value_54_8,OutFor_value_55_8,OutFor_value_56_8), 'JoinComp_8', 'value_57', [('lambdaType', 'value')])\n"
                       "OutFor_value_58JoinComp8(in0,in1,in2,in3,in4,in5,in6,in7,OutFor_key_43_8,OutFor_key_44_8,OutFor_key_45_8,OutFor_key_46_8,OutFor_key_47_8,OutFor_key_48_8,OutFor_key_49_8,OutFor_key_50_8,OutFor_native_lambda_42_8,OutFor_value_52_8,OutFor_value_53_8,OutFor_value_54_8,OutFor_value_55_8,OutFor_value_56_8,OutFor_value_57_8,OutFor_value_58_8) <= APPLY (OutFor_value_57JoinComp8(in6), OutFor_value_57JoinComp8(in0,in1,in2,in3,in4,in5,in6,in7,OutFor_key_43_8,OutFor_key_44_8,OutFor_key_45_8,OutFor_key_46_8,OutFor_key_47_8,OutFor_key_48_8,OutFor_key_49_8,OutFor_key_50_8,OutFor_native_lambda_42_8,OutFor_value_52_8,OutFor_value_53_8,OutFor_value_54_8,OutFor_value_55_8,OutFor_value_56_8,OutFor_value_57_8), 'JoinComp_8', 'value_58', [('lambdaType', 'value')])\n"
                       "OutFor_value_59JoinComp8(in0,in1,in2,in3,in4,in5,in6,in7,OutFor_key_43_8,OutFor_key_44_8,OutFor_key_45_8,OutFor_key_46_8,OutFor_key_47_8,OutFor_key_48_8,OutFor_key_49_8,OutFor_key_50_8,OutFor_native_lambda_42_8,OutFor_value_52_8,OutFor_value_53_8,OutFor_value_54_8,OutFor_value_55_8,OutFor_value_56_8,OutFor_value_57_8,OutFor_value_58_8,OutFor_value_59_8) <= APPLY (OutFor_value_58JoinComp8(in7), OutFor_value_58JoinComp8(in0,in1,in2,in3,in4,in5,in6,in7,OutFor_key_43_8,OutFor_key_44_8,OutFor_key_45_8,OutFor_key_46_8,OutFor_key_47_8,OutFor_key_48_8,OutFor_key_49_8,OutFor_key_50_8,OutFor_native_lambda_42_8,OutFor_value_52_8,OutFor_value_53_8,OutFor_value_54_8,OutFor_value_55_8,OutFor_value_56_8,OutFor_value_57_8,OutFor_value_58_8), 'JoinComp_8', 'value_59', [('lambdaType', 'value')])\n"
                       "OutFor_native_lambda_51JoinComp8(in0,in1,in2,in3,in4,in5,in6,in7,OutFor_key_43_8,OutFor_key_44_8,OutFor_key_45_8,OutFor_key_46_8,OutFor_key_47_8,OutFor_key_48_8,OutFor_key_49_8,OutFor_key_50_8,OutFor_native_lambda_42_8,OutFor_value_52_8,OutFor_value_53_8,OutFor_value_54_8,OutFor_value_55_8,OutFor_value_56_8,OutFor_value_57_8,OutFor_value_58_8,OutFor_value_59_8,OutFor_native_lambda_51_8) <= APPLY (OutFor_value_59JoinComp8(OutFor_value_52_8,OutFor_value_53_8,OutFor_value_54_8,OutFor_value_55_8,OutFor_value_56_8,OutFor_value_57_8,OutFor_value_58_8,OutFor_value_59_8), OutFor_value_59JoinComp8(in0,in1,in2,in3,in4,in5,in6,in7,OutFor_key_43_8,OutFor_key_44_8,OutFor_key_45_8,OutFor_key_46_8,OutFor_key_47_8,OutFor_key_48_8,OutFor_key_49_8,OutFor_key_50_8,OutFor_native_lambda_42_8,OutFor_value_52_8,OutFor_value_53_8,OutFor_value_54_8,OutFor_value_55_8,OutFor_value_56_8,OutFor_value_57_8,OutFor_value_58_8,OutFor_value_59_8), 'JoinComp_8', 'native_lambda_51', [('lambdaType', 'native_lambda')])\n"
                       "OutFor_joinRec_41JoinComp8(OutFor_joinRec_41_8) <= APPLY (OutFor_native_lambda_51JoinComp8(OutFor_native_lambda_42_8,OutFor_native_lambda_51_8), OutFor_native_lambda_51JoinComp8(), 'JoinComp_8', 'joinRec_41', [('lambdaType', 'joinRec')])\n"
                       "OutFor_joinRec_41JoinComp8_out( ) <= OUTPUT ( OutFor_joinRec_41JoinComp8 ( OutFor_joinRec_41_8 ), 'myData', 'B', 'SetWriter_9')";

  // get the string to compile
  myPlan.push_back('\0');

  // where the result of the parse goes
  AtomicComputationList *myResult;

  // now, do the compilation
  yyscan_t scanner;
  LexerExtra extra{""};
  yylex_init_extra(&extra, &scanner);
  const YY_BUFFER_STATE buffer{yy_scan_string(myPlan.data(), scanner)};
  const int parseFailed{yyparse(scanner, &myResult)};
  yy_delete_buffer(buffer, scanner);
  yylex_destroy(scanner);

  std::unordered_map<uint64_t, bool> keyedComputations = {
      {0, true},
      {1, true},
      {2, true},
      {3, true},
      {4, true},
      {5, true},
      {6, true},
      {7, true},
      {8, true},
      {9, false}};
  // if it didn't parse, get outta here
  if (parseFailed) {
    std::cout << "Parse error when compiling TCAP: " << extra.errorMessage;
    exit(1);
  }

  // this is the logical plan to return
  auto atomicComputations = std::shared_ptr<AtomicComputationList>(myResult);

  pdb::PDBPipeNodeBuilder factory(1, keyedComputations, atomicComputations);

  auto out = factory.generateAnalyzerGraph();

  std::cout << "";
}
//
//TEST(TestPipeBuilder, Test1) {
//
//  std::string myLogicalPlan =
//      "inputDataForScanSet_0(in0) <= SCAN ('input_set', 'by8_db', 'ScanSet_0') \n"\
//      "nativ_0OutForSelectionComp1(in0,nativ_0_1OutFor) <= APPLY (inputDataForScanSet_0(in0), inputDataForScanSet_0(in0), 'SelectionComp_1', 'native_lambda_0', [('lambdaType', 'native_lambda')]) \n"\
//      "filteredInputForSelectionComp1(in0) <= FILTER (nativ_0OutForSelectionComp1(nativ_0_1OutFor), nativ_0OutForSelectionComp1(in0), 'SelectionComp_1') \n"\
//      "nativ_1OutForSelectionComp1 (nativ_1_1OutFor) <= APPLY (filteredInputForSelectionComp1(in0), filteredInputForSelectionComp1(), 'SelectionComp_1', 'native_lambda_1', [('lambdaType', 'native_lambda')]) \n"\
//      "nativ_1OutForSelectionComp1_out( ) <= OUTPUT ( nativ_1OutForSelectionComp1 ( nativ_1_1OutFor ), 'output_set', 'by8_db', 'SetWriter_2') \n";
//
//  // get the string to compile
//  myLogicalPlan.push_back('\0');
//
//  // where the result of the parse goes
//  AtomicComputationList *myResult;
//
//  // now, do the compilation
//  yyscan_t scanner;
//  LexerExtra extra{""};
//  yylex_init_extra(&extra, &scanner);
//  const YY_BUFFER_STATE buffer{yy_scan_string(myLogicalPlan.data(), scanner)};
//  const int parseFailed{yyparse(scanner, &myResult)};
//  yy_delete_buffer(buffer, scanner);
//  yylex_destroy(scanner);
//
//  // if it didn't parse, get outta here
//  if (parseFailed) {
//    std::cout << "Parse error when compiling TCAP: " << extra.errorMessage;
//    exit(1);
//  }
//
//  // this is the logical plan to return
//  auto atomicComputations = std::shared_ptr<AtomicComputationList>(myResult);
//
//  pdb::PDBPipeNodeBuilder factory(1, atomicComputations);
//
//  auto out = factory.generateAnalyzerGraph();
//
//  EXPECT_EQ(out.size(), 1);
//
//  int i = 0;
//  for (auto &it : out.front()->getPipeComputations()) {
//
//    switch (i) {
//
//      case 0: {
//
//        EXPECT_EQ(it->getAtomicComputationTypeID(), ScanSetAtomicTypeID);
//        EXPECT_EQ(it->getOutputName(), "inputDataForScanSet_0");
//
//        break;
//      };
//      case 1: {
//
//        EXPECT_EQ(it->getAtomicComputationTypeID(), ApplyLambdaTypeID);
//        EXPECT_EQ(it->getOutputName(), "nativ_0OutForSelectionComp1");
//
//        break;
//      };
//      case 2: {
//
//        EXPECT_EQ(it->getAtomicComputationTypeID(), ApplyFilterTypeID);
//        EXPECT_EQ(it->getOutputName(), "filteredInputForSelectionComp1");
//
//        break;
//      };
//      case 3: {
//
//        EXPECT_EQ(it->getAtomicComputationTypeID(), ApplyLambdaTypeID);
//        EXPECT_EQ(it->getOutputName(), "nativ_1OutForSelectionComp1");
//
//        break;
//      };
//      case 4: {
//
//        EXPECT_EQ(it->getAtomicComputationTypeID(), WriteSetTypeID);
//        EXPECT_EQ(it->getOutputName(), "nativ_1OutForSelectionComp1_out");
//
//        break;
//      };
//      default: {
//        EXPECT_FALSE(true);
//        break;
//      };
//    }
//
//    // increment
//    i++;
//  }
//
//}
//
//TEST(TestPipeBuilder, Test2) {
//
//  std::string myLogicalPlan =
//      "inputData (in) <= SCAN ('mySet', 'myData', 'ScanSet_0', []) \n"\
//      "inputWithAtt (in, att) <= APPLY (inputData (in), inputData (in), 'SelectionComp_1', 'methodCall_0', []) \n"\
//      "inputWithAttAndMethod (in, att, method) <= APPLY (inputWithAtt (in), inputWithAtt (in, att), 'SelectionComp_1', 'attAccess_1', []) \n"\
//      "inputWithBool (in, bool) <= APPLY (inputWithAttAndMethod (att, method), inputWithAttAndMethod (in), 'SelectionComp_1', '==_2', []) \n"\
//      "filteredInput (in) <= FILTER (inputWithBool (bool), inputWithBool (in), 'SelectionComp_1', []) \n"\
//      "projectedInputWithPtr (out) <= APPLY (filteredInput (in), filteredInput (), 'SelectionComp_1', 'methodCall_3', []) \n"\
//      "projectedInput (out) <= APPLY (projectedInputWithPtr (out), projectedInputWithPtr (), 'SelectionComp_1', 'deref_4', []) \n"\
//      "aggWithKeyWithPtr (out, key) <= APPLY (projectedInput (out), projectedInput (out), 'AggregationComp_2', 'attAccess_0', []) \n"\
//      "aggWithKey (out, key) <= APPLY (aggWithKeyWithPtr (key), aggWithKeyWithPtr (out), 'AggregationComp_2', 'deref_1', []) \n"\
//      "aggWithValue (key, value) <= APPLY (aggWithKey (out), aggWithKey (key), 'AggregationComp_2', 'methodCall_2', []) \n"\
//      "agg (aggOut) <=	AGGREGATE (aggWithValue (key, value), 'AggregationComp_2', []) \n"\
//      "checkSales (aggOut, isSales) <= APPLY (agg (aggOut), agg (aggOut), 'SelectionComp_3', 'methodCall_0', []) \n"\
//      "justSales (aggOut, isSales) <= FILTER (checkSales (isSales), checkSales (aggOut), 'SelectionComp_3', []) \n"\
//      "final (result) <= APPLY (justSales (aggOut), justSales (), 'SelectionComp_3', 'methodCall_1', []) \n"\
//      "nothing () <= OUTPUT (final (result), 'outSet', 'myDB', 'SetWriter_4', [])";
//
//  // get the string to compile
//  myLogicalPlan.push_back('\0');
//
//  // where the result of the parse goes
//  AtomicComputationList *myResult;
//
//  // now, do the compilation
//  yyscan_t scanner;
//  LexerExtra extra{""};
//  yylex_init_extra(&extra, &scanner);
//  const YY_BUFFER_STATE buffer{yy_scan_string(myLogicalPlan.data(), scanner)};
//  const int parseFailed{yyparse(scanner, &myResult)};
//  yy_delete_buffer(buffer, scanner);
//  yylex_destroy(scanner);
//
//  // if it didn't parse, get outta here
//  if (parseFailed) {
//    std::cout << "Parse error when compiling TCAP: " << extra.errorMessage;
//    exit(1);
//  }
//
//  // this is the logical plan to return
//  auto atomicComputations = std::shared_ptr<AtomicComputationList>(myResult);
//
//  pdb::PDBPipeNodeBuilder factory(2, atomicComputations);
//
//  auto out = factory.generateAnalyzerGraph();
//
//  EXPECT_EQ(out.size(), 1);
//
//  auto c = out.front();
//  int i = 0;
//  for (auto &it : c->getPipeComputations()) {
//
//    switch (i) {
//
//      case 0: {
//
//        EXPECT_EQ(it->getAtomicComputationTypeID(), ScanSetAtomicTypeID);
//        EXPECT_EQ(it->getOutputName(), "inputData");
//
//        break;
//      };
//      case 1: {
//
//        EXPECT_EQ(it->getAtomicComputationTypeID(), ApplyLambdaTypeID);
//        EXPECT_EQ(it->getOutputName(), "inputWithAtt");
//
//        break;
//      };
//      case 2: {
//
//        EXPECT_EQ(it->getAtomicComputationTypeID(), ApplyLambdaTypeID);
//        EXPECT_EQ(it->getOutputName(), "inputWithAttAndMethod");
//
//        break;
//      };
//      case 3: {
//
//        EXPECT_EQ(it->getAtomicComputationTypeID(), ApplyLambdaTypeID);
//        EXPECT_EQ(it->getOutputName(), "inputWithBool");
//
//        break;
//      };
//      case 4: {
//
//        EXPECT_EQ(it->getAtomicComputationTypeID(), ApplyFilterTypeID);
//        EXPECT_EQ(it->getOutputName(), "filteredInput");
//
//        break;
//      };
//      case 5: {
//
//        EXPECT_EQ(it->getAtomicComputationTypeID(), ApplyLambdaTypeID);
//        EXPECT_EQ(it->getOutputName(), "projectedInputWithPtr");
//
//        break;
//      };
//      case 6: {
//
//        EXPECT_EQ(it->getAtomicComputationTypeID(), ApplyLambdaTypeID);
//        EXPECT_EQ(it->getOutputName(), "projectedInput");
//
//        break;
//      };
//      case 7: {
//
//        EXPECT_EQ(it->getAtomicComputationTypeID(), ApplyLambdaTypeID);
//        EXPECT_EQ(it->getOutputName(), "aggWithKeyWithPtr");
//
//        break;
//      };
//      case 8: {
//
//        EXPECT_EQ(it->getAtomicComputationTypeID(), ApplyLambdaTypeID);
//        EXPECT_EQ(it->getOutputName(), "aggWithKey");
//
//        break;
//      };
//      case 9: {
//
//        EXPECT_EQ(it->getAtomicComputationTypeID(), ApplyLambdaTypeID);
//        EXPECT_EQ(it->getOutputName(), "aggWithValue");
//
//        break;
//      };
//      default: {
//        EXPECT_FALSE(true);
//        break;
//      };
//    }
//
//    // increment
//    i++;
//  }
//
//  auto producers = c->getConsumers();
//  EXPECT_EQ(producers.size(), 1);
//
//  i = 0;
//  for (auto &it : producers.front()->getPipeComputations()) {
//    switch (i) {
//
//      case 0: {
//
//        EXPECT_EQ(it->getAtomicComputationTypeID(), ApplyAggTypeID);
//        EXPECT_EQ(it->getOutputName(), "agg");
//
//        break;
//      };
//      case 1: {
//
//        EXPECT_EQ(it->getAtomicComputationTypeID(), ApplyLambdaTypeID);
//        EXPECT_EQ(it->getOutputName(), "checkSales");
//
//        break;
//      };
//      case 2: {
//
//        EXPECT_EQ(it->getAtomicComputationTypeID(), ApplyFilterTypeID);
//        EXPECT_EQ(it->getOutputName(), "justSales");
//
//        break;
//      };
//      case 3: {
//
//        EXPECT_EQ(it->getAtomicComputationTypeID(), ApplyLambdaTypeID);
//        EXPECT_EQ(it->getOutputName(), "final");
//
//        break;
//      };
//      case 4: {
//
//        EXPECT_EQ(it->getAtomicComputationTypeID(), WriteSetTypeID);
//        EXPECT_EQ(it->getOutputName(), "nothing");
//
//        break;
//      };
//      default: {
//        EXPECT_FALSE(true);
//        break;
//      };
//    }
//
//    i++;
//  }
//
//}
//
//TEST(TestPipeBuilder, Test3) {
//  std::string myLogicalPlan = "/* scan the three inputs */ \n"\
//      "A (a) <= SCAN ('mySet', 'myData', 'ScanSet_0', []) \n"\
//      "B (aAndC) <= SCAN ('mySet', 'myData', 'ScanSet_1', []) \n"\
//      "C (c) <= SCAN ('mySet', 'myData', 'ScanSet_2', []) \n"\
//      "\n"\
//      "/* extract and hash a from A */ \n"\
//      "AWithAExtracted (a, aExtracted) <= APPLY (A (a), A(a), 'JoinComp_3', 'self_0', []) \n"\
//      "AHashed (a, hash) <= HASHLEFT (AWithAExtracted (aExtracted), A (a), 'JoinComp_3', '==_2', []) \n"\
//      "\n"\
//      "/* extract and hash a from B */ \n"\
//      "BWithAExtracted (aAndC, a) <= APPLY (B (aAndC), B (aAndC), 'JoinComp_3', 'attAccess_1', []) \n"\
//      "BHashedOnA (aAndC, hash) <= HASHRIGHT (BWithAExtracted (a), BWithAExtracted (aAndC), 'JoinComp_3', '==_2', []) \n"\
//      "\n"\
//      "/* now, join the two of them */ \n"\
//      "AandBJoined (a, aAndC) <= JOIN (AHashed (hash), AHashed (a), BHashedOnA (hash), BHashedOnA (aAndC), 'JoinComp_3', []) \n"\
//      "\n"\
//      "/* and extract the two atts and check for equality */ \n"\
//      "AandBJoinedWithAExtracted (a, aAndC, aExtracted) <= APPLY (AandBJoined (a), AandBJoined (a, aAndC), 'JoinComp_3', 'self_0', []) \n"\
//      "AandBJoinedWithBothExtracted (a, aAndC, aExtracted, otherA) <= APPLY (AandBJoinedWithAExtracted (aAndC), AandBJoinedWithAExtracted (a, aAndC, aExtracted), 'JoinComp_3', 'attAccess_1', []) \n"\
//      "AandBJoinedWithBool (aAndC, a, bool) <= APPLY (AandBJoinedWithBothExtracted (aExtracted, otherA), AandBJoinedWithBothExtracted (aAndC, a), 'JoinComp_3', '==_2', []) \n"\
//      "AandBJoinedFiltered (a, aAndC) <= FILTER (AandBJoinedWithBool (bool), AandBJoinedWithBool (a, aAndC), 'JoinComp_3', []) \n"\
//      "\n"\
//      "/* now get ready to join the strings */ \n"\
//      "AandBJoinedFilteredWithC (a, aAndC, cExtracted) <= APPLY (AandBJoinedFiltered (aAndC), AandBJoinedFiltered (a, aAndC), 'JoinComp_3', 'attAccess_3', []) \n"\
//      "BHashedOnC (a, aAndC, hash) <= HASHLEFT (AandBJoinedFilteredWithC (cExtracted), AandBJoinedFilteredWithC (a, aAndC), 'JoinComp_3', '==_5', []) \n"\
//      "CwithCExtracted (c, cExtracted) <= APPLY (C (c), C (c), 'JoinComp_3', 'self_0', []) \n"\
//      "CHashedOnC (c, hash) <= HASHRIGHT (CwithCExtracted (cExtracted), CwithCExtracted (c), 'JoinComp_3', '==_5', []) \n"\
//      "\n"\
//      "/* join the two of them */ \n"\
//      "BandCJoined (a, aAndC, c) <= JOIN (BHashedOnC (hash), BHashedOnC (a, aAndC), CHashedOnC (hash), CHashedOnC (c), 'JoinComp_3', []) \n"\
//      "\n"\
//      "/* and extract the two atts and check for equality */ \n"\
//      "BandCJoinedWithCExtracted (a, aAndC, c, cFromLeft) <= APPLY (BandCJoined (aAndC), BandCJoined (a, aAndC, c), 'JoinComp_3', 'attAccess_3', []) \n"\
//      "BandCJoinedWithBoth (a, aAndC, c, cFromLeft, cFromRight) <= APPLY (BandCJoinedWithCExtracted (c), BandCJoinedWithCExtracted (a, aAndC, c, cFromLeft), 'JoinComp_3', 'self_4', []) \n"\
//      "BandCJoinedWithBool (a, aAndC, c, bool) <= APPLY (BandCJoinedWithBoth (cFromLeft, cFromRight), BandCJoinedWithBoth (a, aAndC, c), 'JoinComp_3', '==_5', []) \n"\
//      "last (a, aAndC, c) <= FILTER (BandCJoinedWithBool (bool), BandCJoinedWithBool (a, aAndC, c), 'JoinComp_3', []) \n"\
//      "\n"\
//      "/* and here is the answer */ \n"\
//      "almostFinal (result) <= APPLY (last (a, aAndC, c), last (), 'JoinComp_3', 'native_lambda_7', []) \n"\
//      "nothing () <= OUTPUT (almostFinal (result), 'outSet', 'myDB', 'SetWriter_4', [])";
//
//  // get the string to compile
//  myLogicalPlan.push_back('\0');
//
//  // where the result of the parse goes
//  AtomicComputationList *myResult;
//
//  // now, do the compilation
//  yyscan_t scanner;
//  LexerExtra extra{""};
//  yylex_init_extra(&extra, &scanner);
//  const YY_BUFFER_STATE buffer{yy_scan_string(myLogicalPlan.data(), scanner)};
//  const int parseFailed{yyparse(scanner, &myResult)};
//  yy_delete_buffer(buffer, scanner);
//  yylex_destroy(scanner);
//
//  // if it didn't parse, get outta here
//  if (parseFailed) {
//    std::cout << "Parse error when compiling TCAP: " << extra.errorMessage;
//    exit(1);
//  }
//
//  // this is the logical plan to return
//  auto atomicComputations = std::shared_ptr<AtomicComputationList>(myResult);
//
//  pdb::PDBPipeNodeBuilder factory(3, atomicComputations);
//
//  auto out = factory.generateAnalyzerGraph();
//  std::set<pdb::PDBAbstractPhysicalNodePtr> visitedNodes;
//
//  // check size
//  EXPECT_EQ(out.size(), 3);
//
//  // chec preaggregationPipelines
//  while (!out.empty()) {
//
//    auto firstComp = out.back()->getPipeComputations().front();
//
//    if (firstComp->getAtomicComputationTypeID() == ScanSetAtomicTypeID && firstComp->getOutputName() == "A") {
//
//      int i = 0;
//      for (auto &it : out.back()->getPipeComputations()) {
//        switch (i) {
//
//          case 0: {
//
//            EXPECT_EQ(it->getAtomicComputationTypeID(), ScanSetAtomicTypeID);
//            EXPECT_EQ(it->getOutputName(), "A");
//
//            break;
//          };
//          case 1: {
//
//            EXPECT_EQ(it->getAtomicComputationTypeID(), ApplyLambdaTypeID);
//            EXPECT_EQ(it->getOutputName(), "AWithAExtracted");
//
//            break;
//          };
//          case 2: {
//
//            EXPECT_EQ(it->getAtomicComputationTypeID(), HashLeftTypeID);
//            EXPECT_EQ(it->getOutputName(), "AHashed");
//
//            break;
//          };
//          default: {
//            EXPECT_FALSE(true);
//            break;
//          };
//        }
//
//        i++;
//      }
//
//      // this must be the first time we visited this
//      EXPECT_EQ(visitedNodes.find(out.back()), visitedNodes.end());
//      visitedNodes.insert(out.back());
//
//      // do we have one consumer
//      EXPECT_EQ(out.back()->getConsumers().size(), 1);
//      EXPECT_EQ(out.back()->getConsumers().front()->getPipeComputations().front()->getOutputName(), "AandBJoined");
//
//      // check the other side
//      auto otherSide = ((pdb::PDBJoinPhysicalNode *) out.back().get())->otherSide.lock();
//      firstComp = otherSide->getPipeComputations().front();
//      EXPECT_TRUE(firstComp->getAtomicComputationTypeID() == ScanSetAtomicTypeID && firstComp->getOutputName() == "B");
//
//    } else if (firstComp->getAtomicComputationTypeID() == ScanSetAtomicTypeID && firstComp->getOutputName() == "B") {
//
//      int i = 0;
//      for (auto &it : out.back()->getPipeComputations()) {
//        switch (i) {
//
//          case 0: {
//
//            EXPECT_EQ(it->getAtomicComputationTypeID(), ScanSetAtomicTypeID);
//            EXPECT_EQ(it->getOutputName(), "B");
//
//            break;
//          };
//          case 1: {
//
//            EXPECT_EQ(it->getAtomicComputationTypeID(), ApplyLambdaTypeID);
//            EXPECT_EQ(it->getOutputName(), "BWithAExtracted");
//
//            break;
//          };
//          case 2: {
//
//            EXPECT_EQ(it->getAtomicComputationTypeID(), HashRightTypeID);
//            EXPECT_EQ(it->getOutputName(), "BHashedOnA");
//
//            break;
//          };
//          default: {
//            EXPECT_FALSE(true);
//            break;
//          };
//        }
//
//        i++;
//      }
//
//      // this must be the first time we visited this
//      EXPECT_EQ(visitedNodes.find(out.back()), visitedNodes.end());
//      visitedNodes.insert(out.back());
//
//      // do we have one consumer
//      EXPECT_EQ(out.back()->getConsumers().size(), 1);
//      EXPECT_EQ(out.back()->getConsumers().front()->getPipeComputations().front()->getOutputName(), "AandBJoined");
//
//      // check the other side
//      auto otherSide = ((pdb::PDBJoinPhysicalNode *) out.back().get())->otherSide.lock();
//      firstComp = otherSide->getPipeComputations().front();
//      EXPECT_TRUE(firstComp->getAtomicComputationTypeID() == ScanSetAtomicTypeID && firstComp->getOutputName() == "A");
//
//    } else if (firstComp->getAtomicComputationTypeID() == ScanSetAtomicTypeID && firstComp->getOutputName() == "C") {
//
//      int i = 0;
//      for (auto &it : out.back()->getPipeComputations()) {
//        switch (i) {
//
//          case 0: {
//
//            EXPECT_EQ(it->getAtomicComputationTypeID(), ScanSetAtomicTypeID);
//            EXPECT_EQ(it->getOutputName(), "C");
//
//            break;
//          };
//          case 1: {
//
//            EXPECT_EQ(it->getAtomicComputationTypeID(), ApplyLambdaTypeID);
//            EXPECT_EQ(it->getOutputName(), "CwithCExtracted");
//
//            break;
//          };
//          case 2: {
//
//            EXPECT_EQ(it->getAtomicComputationTypeID(), HashRightTypeID);
//            EXPECT_EQ(it->getOutputName(), "CHashedOnC");
//
//            break;
//          };
//          default: {
//            EXPECT_FALSE(true);
//            break;
//          };
//        }
//
//        i++;
//      }
//
//      // this must be the first time we visited this
//      EXPECT_EQ(visitedNodes.find(out.back()), visitedNodes.end());
//      visitedNodes.insert(out.back());
//
//      // do we have one consumer
//      EXPECT_EQ(out.back()->getConsumers().size(), 1);
//      EXPECT_EQ(out.back()->getConsumers().front()->getPipeComputations().front()->getOutputName(), "BandCJoined");
//
//      // check the other side
//      auto otherSide = ((pdb::PDBJoinPhysicalNode *) out.back().get())->otherSide.lock();
//      firstComp = otherSide->getPipeComputations().front();
//      EXPECT_TRUE(
//          firstComp->getAtomicComputationTypeID() == ApplyJoinTypeID && firstComp->getOutputName() == "AandBJoined");
//
//    } else if (firstComp->getAtomicComputationTypeID() == ApplyJoinTypeID
//        && firstComp->getOutputName() == "AandBJoined") {
//
//      int i = 0;
//      for (auto &it : out.back()->getPipeComputations()) {
//        switch (i) {
//
//          case 0: {
//
//            EXPECT_EQ(it->getAtomicComputationTypeID(), ApplyJoinTypeID);
//            EXPECT_EQ(it->getOutputName(), "AandBJoined");
//
//            break;
//          };
//          case 1: {
//
//            EXPECT_EQ(it->getAtomicComputationTypeID(), ApplyLambdaTypeID);
//            EXPECT_EQ(it->getOutputName(), "AandBJoinedWithAExtracted");
//
//            break;
//          };
//          case 2: {
//
//            EXPECT_EQ(it->getAtomicComputationTypeID(), ApplyLambdaTypeID);
//            EXPECT_EQ(it->getOutputName(), "AandBJoinedWithBothExtracted");
//
//            break;
//          };
//          case 3: {
//
//            EXPECT_EQ(it->getAtomicComputationTypeID(), ApplyLambdaTypeID);
//            EXPECT_EQ(it->getOutputName(), "AandBJoinedWithBool");
//
//            break;
//          };
//          case 4: {
//
//            EXPECT_EQ(it->getAtomicComputationTypeID(), ApplyFilterTypeID);
//            EXPECT_EQ(it->getOutputName(), "AandBJoinedFiltered");
//
//            break;
//          };
//          case 5: {
//
//            EXPECT_EQ(it->getAtomicComputationTypeID(), ApplyLambdaTypeID);
//            EXPECT_EQ(it->getOutputName(), "AandBJoinedFilteredWithC");
//
//            break;
//          };
//          case 6: {
//
//            EXPECT_EQ(it->getAtomicComputationTypeID(), HashLeftTypeID);
//            EXPECT_EQ(it->getOutputName(), "BHashedOnC");
//
//            break;
//          };
//          default: {
//            EXPECT_FALSE(true);
//            break;
//          };
//        }
//
//        i++;
//      }
//
//      // this must be the first time we visited this
//      EXPECT_EQ(visitedNodes.find(out.back()), visitedNodes.end());
//      visitedNodes.insert(out.back());
//
//      // do we have one consumer
//      EXPECT_EQ(out.back()->getConsumers().size(), 1);
//      EXPECT_EQ(out.back()->getConsumers().front()->getPipeComputations().front()->getOutputName(), "BandCJoined");
//
//      // check the other side
//      auto otherSide = ((pdb::PDBJoinPhysicalNode *) out.back().get())->otherSide.lock();
//      firstComp = otherSide->getPipeComputations().front();
//      EXPECT_TRUE(firstComp->getAtomicComputationTypeID() == ScanSetAtomicTypeID && firstComp->getOutputName() == "C");
//    } else if (firstComp->getAtomicComputationTypeID() == ApplyJoinTypeID
//        && firstComp->getOutputName() == "BandCJoined") {
//
//      int i = 0;
//      for (auto &it : out.back()->getPipeComputations()) {
//        switch (i) {
//
//          case 0: {
//
//            EXPECT_EQ(it->getAtomicComputationTypeID(), ApplyJoinTypeID);
//            EXPECT_EQ(it->getOutputName(), "BandCJoined");
//
//            break;
//          };
//          case 1: {
//
//            EXPECT_EQ(it->getAtomicComputationTypeID(), ApplyLambdaTypeID);
//            EXPECT_EQ(it->getOutputName(), "BandCJoinedWithCExtracted");
//
//            break;
//          };
//          case 2: {
//
//            EXPECT_EQ(it->getAtomicComputationTypeID(), ApplyLambdaTypeID);
//            EXPECT_EQ(it->getOutputName(), "BandCJoinedWithBoth");
//
//            break;
//          };
//          case 3: {
//
//            EXPECT_EQ(it->getAtomicComputationTypeID(), ApplyLambdaTypeID);
//            EXPECT_EQ(it->getOutputName(), "BandCJoinedWithBool");
//
//            break;
//          };
//          case 4: {
//
//            EXPECT_EQ(it->getAtomicComputationTypeID(), ApplyFilterTypeID);
//            EXPECT_EQ(it->getOutputName(), "last");
//
//            break;
//          };
//          case 5: {
//
//            EXPECT_EQ(it->getAtomicComputationTypeID(), ApplyLambdaTypeID);
//            EXPECT_EQ(it->getOutputName(), "almostFinal");
//
//            break;
//          };
//          case 6: {
//
//            EXPECT_EQ(it->getAtomicComputationTypeID(), WriteSetTypeID);
//            EXPECT_EQ(it->getOutputName(), "nothing");
//
//            break;
//          };
//          default: {
//            EXPECT_FALSE(true);
//            break;
//          };
//        }
//
//        ++i;
//      }
//
//      // this must be the first time we visited this
//      EXPECT_EQ(visitedNodes.find(out.back()), visitedNodes.end());
//      visitedNodes.insert(out.back());
//
//    } else {
//      EXPECT_FALSE(true);
//    }
//
//    // get the last
//    auto me = out.back();
//    out.pop_back();
//
//    // go through all consumers if they are not visited visit them
//    for (auto &it : me->getConsumers()) {
//      if (visitedNodes.find(it) == visitedNodes.end()) {
//        out.push_back(it);
//      }
//    }
//  }
//
//}
//
//TEST(TestPipeBuilder, Test4) {
//  std::string myLogicalPlan = "inputDataForSetScanner_0(in0) <= SCAN ('LDA_db', 'LDA_input_set', 'SetScanner_0')\n"
//                              "\n"
//                              "/* Extract key for aggregation */\n"
//                              "nativ_0OutForAggregationComp1(in0,nativ_0_1OutFor) <= APPLY (inputDataForSetScanner_0(in0), inputDataForSetScanner_0(in0), 'AggregationComp_1', 'native_lambda_0', [('lambdaType', 'native_lambda')])\n"
//                              "\n"
//                              "/* Extract value for aggregation */\n"
//                              "nativ_1OutForAggregationComp1(nativ_0_1OutFor,nativ_1_1OutFor) <= APPLY (nativ_0OutForAggregationComp1(in0), nativ_0OutForAggregationComp1(nativ_0_1OutFor), 'AggregationComp_1', 'native_lambda_1', [('lambdaType', 'native_lambda')])\n"
//                              "\n"
//                              "/* Apply aggregation */\n"
//                              "aggOutForAggregationComp1 (aggOutFor1)<= AGGREGATE (nativ_1OutForAggregationComp1(nativ_0_1OutFor, nativ_1_1OutFor),'AggregationComp_1')\n"
//                              "\n"
//                              "/* Apply selection filtering */\n"
//                              "nativ_0OutForSelectionComp2(aggOutFor1,nativ_0_2OutFor) <= APPLY (aggOutForAggregationComp1(aggOutFor1), aggOutForAggregationComp1(aggOutFor1), 'SelectionComp_2', 'native_lambda_0', [('lambdaType', 'native_lambda')])\n"
//                              "filteredInputForSelectionComp2(aggOutFor1) <= FILTER (nativ_0OutForSelectionComp2(nativ_0_2OutFor), nativ_0OutForSelectionComp2(aggOutFor1), 'SelectionComp_2')\n"
//                              "\n"
//                              "/* Apply selection projection */\n"
//                              "nativ_1OutForSelectionComp2 (nativ_1_2OutFor) <= APPLY (filteredInputForSelectionComp2(aggOutFor1), filteredInputForSelectionComp2(), 'SelectionComp_2', 'native_lambda_1', [('lambdaType', 'native_lambda')])\n"
//                              "inputDataForSetScanner_3(in3) <= SCAN ('LDA_db', 'LDA_meta_data_set', 'SetScanner_3')\n"
//                              "\n"
//                              "/* Apply selection filtering */\n"
//                              "nativ_0OutForSelectionComp4(in3,nativ_0_4OutFor) <= APPLY (inputDataForSetScanner_3(in3), inputDataForSetScanner_3(in3), 'SelectionComp_4', 'native_lambda_0', [('lambdaType', 'native_lambda')])\n"
//                              "filteredInputForSelectionComp4(in3) <= FILTER (nativ_0OutForSelectionComp4(nativ_0_4OutFor), nativ_0OutForSelectionComp4(in3), 'SelectionComp_4')\n"
//                              "\n"
//                              "/* Apply selection projection */\n"
//                              "nativ_1OutForSelectionComp4 (nativ_1_4OutFor) <= APPLY (filteredInputForSelectionComp4(in3), filteredInputForSelectionComp4(), 'SelectionComp_4', 'native_lambda_1', [('lambdaType', 'native_lambda')])\n"
//                              "attAccess_0ExtractedForJoinComp5(in0,att_0ExtractedFor_docID) <= APPLY (inputDataForSetScanner_0(in0), inputDataForSetScanner_0(in0), 'JoinComp_5', 'attAccess_0', [('attName', 'docID'), ('attTypeName', 'unsignedint'), ('inputTypeName', 'LDADocument'), ('lambdaType', 'attAccess')])\n"
//                              "attAccess_0ExtractedForJoinComp5_hashed(in0,att_0ExtractedFor_docID_hash) <= HASHLEFT (attAccess_0ExtractedForJoinComp5(att_0ExtractedFor_docID), attAccess_0ExtractedForJoinComp5(in0), 'JoinComp_5', '==_2', [])\n"
//                              "attAccess_1ExtractedForJoinComp5(nativ_1_2OutFor,att_1ExtractedFor_myInt) <= APPLY (nativ_1OutForSelectionComp2(nativ_1_2OutFor), nativ_1OutForSelectionComp2(nativ_1_2OutFor), 'JoinComp_5', 'attAccess_1', [('attName', 'myInt'), ('attTypeName', 'int'), ('inputTypeName', 'IntDoubleVectorPair'), ('lambdaType', 'attAccess')])\n"
//                              "attAccess_1ExtractedForJoinComp5_hashed(nativ_1_2OutFor,att_1ExtractedFor_myInt_hash) <= HASHRIGHT (attAccess_1ExtractedForJoinComp5(att_1ExtractedFor_myInt), attAccess_1ExtractedForJoinComp5(nativ_1_2OutFor), 'JoinComp_5', '==_2', [])\n"
//                              "\n"
//                              "/* Join ( in0 ) and ( nativ_1_2OutFor ) */\n"
//                              "JoinedFor_equals2JoinComp5(in0, nativ_1_2OutFor) <= JOIN (attAccess_0ExtractedForJoinComp5_hashed(att_0ExtractedFor_docID_hash), attAccess_0ExtractedForJoinComp5_hashed(in0), attAccess_1ExtractedForJoinComp5_hashed(att_1ExtractedFor_myInt_hash), attAccess_1ExtractedForJoinComp5_hashed(nativ_1_2OutFor), 'JoinComp_5')\n"
//                              "JoinedFor_equals2JoinComp5_WithLHSExtracted(in0,nativ_1_2OutFor,LHSExtractedFor_2_5) <= APPLY (JoinedFor_equals2JoinComp5(in0), JoinedFor_equals2JoinComp5(in0,nativ_1_2OutFor), 'JoinComp_5', 'attAccess_0', [('attName', 'docID'), ('attTypeName', 'unsignedint'), ('inputTypeName', 'LDADocument'), ('lambdaType', 'attAccess')])\n"
//                              "JoinedFor_equals2JoinComp5_WithBOTHExtracted(in0,nativ_1_2OutFor,LHSExtractedFor_2_5,RHSExtractedFor_2_5) <= APPLY (JoinedFor_equals2JoinComp5_WithLHSExtracted(nativ_1_2OutFor), JoinedFor_equals2JoinComp5_WithLHSExtracted(in0,nativ_1_2OutFor,LHSExtractedFor_2_5), 'JoinComp_5', 'attAccess_1', [('attName', 'myInt'), ('attTypeName', 'int'), ('inputTypeName', 'IntDoubleVectorPair'), ('lambdaType', 'attAccess')])\n"
//                              "JoinedFor_equals2JoinComp5_BOOL(in0,nativ_1_2OutFor,bool_2_5) <= APPLY (JoinedFor_equals2JoinComp5_WithBOTHExtracted(LHSExtractedFor_2_5,RHSExtractedFor_2_5), JoinedFor_equals2JoinComp5_WithBOTHExtracted(in0,nativ_1_2OutFor), 'JoinComp_5', '==_2', [('lambdaType', '==')])\n"
//                              "JoinedFor_equals2JoinComp5_FILTERED(in0, nativ_1_2OutFor) <= FILTER (JoinedFor_equals2JoinComp5_BOOL(bool_2_5), JoinedFor_equals2JoinComp5_BOOL(in0, nativ_1_2OutFor), 'JoinComp_5')\n"
//                              "attAccess_3ExtractedForJoinComp5(in0,nativ_1_2OutFor,att_3ExtractedFor_wordID) <= APPLY (JoinedFor_equals2JoinComp5_FILTERED(in0), JoinedFor_equals2JoinComp5_FILTERED(in0,nativ_1_2OutFor), 'JoinComp_5', 'attAccess_3', [('attName', 'wordID'), ('attTypeName', 'unsignedint'), ('inputTypeName', 'LDADocument'), ('lambdaType', 'attAccess')])\n"
//                              "attAccess_3ExtractedForJoinComp5_hashed(in0,nativ_1_2OutFor,att_3ExtractedFor_wordID_hash) <= HASHLEFT (attAccess_3ExtractedForJoinComp5(att_3ExtractedFor_wordID), attAccess_3ExtractedForJoinComp5(in0,nativ_1_2OutFor), 'JoinComp_5', '==_5', [])\n"
//                              "attAccess_4ExtractedForJoinComp5(nativ_1_4OutFor,att_4ExtractedFor_whichWord) <= APPLY (nativ_1OutForSelectionComp4(nativ_1_4OutFor), nativ_1OutForSelectionComp4(nativ_1_4OutFor), 'JoinComp_5', 'attAccess_4', [('attName', 'whichWord'), ('attTypeName', 'unsignedint'), ('inputTypeName', 'LDATopicWordProb'), ('lambdaType', 'attAccess')])\n"
//                              "attAccess_4ExtractedForJoinComp5_hashed(nativ_1_4OutFor,att_4ExtractedFor_whichWord_hash) <= HASHRIGHT (attAccess_4ExtractedForJoinComp5(att_4ExtractedFor_whichWord), attAccess_4ExtractedForJoinComp5(nativ_1_4OutFor), 'JoinComp_5', '==_5', [])\n"
//                              "\n"
//                              "/* Join ( in0 nativ_1_2OutFor ) and ( nativ_1_4OutFor ) */\n"
//                              "JoinedFor_equals5JoinComp5(in0, nativ_1_2OutFor, nativ_1_4OutFor) <= JOIN (attAccess_3ExtractedForJoinComp5_hashed(att_3ExtractedFor_wordID_hash), attAccess_3ExtractedForJoinComp5_hashed(in0, nativ_1_2OutFor), attAccess_4ExtractedForJoinComp5_hashed(att_4ExtractedFor_whichWord_hash), attAccess_4ExtractedForJoinComp5_hashed(nativ_1_4OutFor), 'JoinComp_5')\n"
//                              "JoinedFor_equals5JoinComp5_WithLHSExtracted(in0,nativ_1_2OutFor,nativ_1_4OutFor,LHSExtractedFor_5_5) <= APPLY (JoinedFor_equals5JoinComp5(in0), JoinedFor_equals5JoinComp5(in0,nativ_1_2OutFor,nativ_1_4OutFor), 'JoinComp_5', 'attAccess_3', [('attName', 'wordID'), ('attTypeName', 'unsignedint'), ('inputTypeName', 'LDADocument'), ('lambdaType', 'attAccess')])\n"
//                              "JoinedFor_equals5JoinComp5_WithBOTHExtracted(in0,nativ_1_2OutFor,nativ_1_4OutFor,LHSExtractedFor_5_5,RHSExtractedFor_5_5) <= APPLY (JoinedFor_equals5JoinComp5_WithLHSExtracted(nativ_1_4OutFor), JoinedFor_equals5JoinComp5_WithLHSExtracted(in0,nativ_1_2OutFor,nativ_1_4OutFor,LHSExtractedFor_5_5), 'JoinComp_5', 'attAccess_4', [('attName', 'whichWord'), ('attTypeName', 'unsignedint'), ('inputTypeName', 'LDATopicWordProb'), ('lambdaType', 'attAccess')])\n"
//                              "JoinedFor_equals5JoinComp5_BOOL(in0,nativ_1_2OutFor,nativ_1_4OutFor,bool_5_5) <= APPLY (JoinedFor_equals5JoinComp5_WithBOTHExtracted(LHSExtractedFor_5_5,RHSExtractedFor_5_5), JoinedFor_equals5JoinComp5_WithBOTHExtracted(in0,nativ_1_2OutFor,nativ_1_4OutFor), 'JoinComp_5', '==_5', [('lambdaType', '==')])\n"
//                              "JoinedFor_equals5JoinComp5_FILTERED(in0, nativ_1_2OutFor, nativ_1_4OutFor) <= FILTER (JoinedFor_equals5JoinComp5_BOOL(bool_5_5), JoinedFor_equals5JoinComp5_BOOL(in0, nativ_1_2OutFor, nativ_1_4OutFor), 'JoinComp_5')\n"
//                              "\n"
//                              "/* run Join projection on ( in0 nativ_1_2OutFor nativ_1_4OutFor )*/\n"
//                              "nativ_7OutForJoinComp5 (nativ_7_5OutFor) <= APPLY (JoinedFor_equals5JoinComp5_FILTERED(in0,nativ_1_2OutFor,nativ_1_4OutFor), JoinedFor_equals5JoinComp5_FILTERED(), 'JoinComp_5', 'native_lambda_7', [('lambdaType', 'native_lambda')])\n"
//                              "\n"
//                              "/* Apply selection filtering */\n"
//                              "nativ_0OutForSelectionComp6(nativ_7_5OutFor,nativ_0_6OutFor) <= APPLY (nativ_7OutForJoinComp5(nativ_7_5OutFor), nativ_7OutForJoinComp5(nativ_7_5OutFor), 'SelectionComp_6', 'native_lambda_0', [('lambdaType', 'native_lambda')])\n"
//                              "filteredInputForSelectionComp6(nativ_7_5OutFor) <= FILTER (nativ_0OutForSelectionComp6(nativ_0_6OutFor), nativ_0OutForSelectionComp6(nativ_7_5OutFor), 'SelectionComp_6')\n"
//                              "\n"
//                              "/* Apply selection projection */\n"
//                              "nativ_1OutForSelectionComp6 (nativ_1_6OutFor) <= APPLY (filteredInputForSelectionComp6(nativ_7_5OutFor), filteredInputForSelectionComp6(), 'SelectionComp_6', 'native_lambda_1', [('lambdaType', 'native_lambda')])\n"
//                              "\n"
//                              "/* Apply MultiSelection filtering */\n"
//                              "nativ_0OutForMultiSelectionComp7(nativ_1_6OutFor,nativ_0_7OutFor) <= APPLY (nativ_1OutForSelectionComp6(nativ_1_6OutFor), nativ_1OutForSelectionComp6(nativ_1_6OutFor), 'MultiSelectionComp_7', 'native_lambda_0', [('lambdaType', 'native_lambda')])\n"
//                              "filteredInputForMultiSelectionComp7(nativ_1_6OutFor) <= FILTER (nativ_0OutForMultiSelectionComp7(nativ_0_7OutFor), nativ_0OutForMultiSelectionComp7(nativ_1_6OutFor), 'MultiSelectionComp_7')\n"
//                              "\n"
//                              "/* Apply MultiSelection projection */\n"
//                              "methodCall_1OutFor_MultiSelectionComp7(nativ_1_6OutFor,methodCall_1OutFor__getTopicAssigns) <= APPLY (filteredInputForMultiSelectionComp7(nativ_1_6OutFor), filteredInputForMultiSelectionComp7(nativ_1_6OutFor), 'MultiSelectionComp_7', 'methodCall_1', [('inputTypeName', 'LDADocWordTopicAssignment'), ('lambdaType', 'methodCall'), ('methodName', 'getTopicAssigns'), ('returnTypeName', 'LDADocWordTopicAssignment')])\n"
//                              "deref_2OutForMultiSelectionComp7 (methodCall_1OutFor__getTopicAssigns) <= APPLY (methodCall_1OutFor_MultiSelectionComp7(methodCall_1OutFor__getTopicAssigns), methodCall_1OutFor_MultiSelectionComp7(), 'MultiSelectionComp_7', 'deref_2')\n"
//                              "flattenedOutForMultiSelectionComp7(flattened_methodCall_1OutFor__getTopicAssigns) <= FLATTEN (deref_2OutForMultiSelectionComp7(methodCall_1OutFor__getTopicAssigns), deref_2OutForMultiSelectionComp7(), 'MultiSelectionComp_7')\n"
//                              "\n"
//                              "/* Extract key for aggregation */\n"
//                              "methodCall_0OutFor_AggregationComp8(flattened_methodCall_1OutFor__getTopicAssigns,methodCall_0OutFor__getKey) <= APPLY (flattenedOutForMultiSelectionComp7(flattened_methodCall_1OutFor__getTopicAssigns), flattenedOutForMultiSelectionComp7(flattened_methodCall_1OutFor__getTopicAssigns), 'AggregationComp_8', 'methodCall_0', [('inputTypeName', 'TopicAssignment'), ('lambdaType', 'methodCall'), ('methodName', 'getKey'), ('returnTypeName', 'TopicAssignment')])\n"
//                              "deref_1OutForAggregationComp8(flattened_methodCall_1OutFor__getTopicAssigns, methodCall_0OutFor__getKey) <= APPLY (methodCall_0OutFor_AggregationComp8(methodCall_0OutFor__getKey), methodCall_0OutFor_AggregationComp8(flattened_methodCall_1OutFor__getTopicAssigns), 'AggregationComp_8', 'deref_1')\n"
//                              "\n"
//                              "/* Extract value for aggregation */\n"
//                              "methodCall_2OutFor_AggregationComp8(methodCall_0OutFor__getKey,methodCall_2OutFor__getValue) <= APPLY (deref_1OutForAggregationComp8(flattened_methodCall_1OutFor__getTopicAssigns), deref_1OutForAggregationComp8(methodCall_0OutFor__getKey), 'AggregationComp_8', 'methodCall_2', [('inputTypeName', 'TopicAssignment'), ('lambdaType', 'methodCall'), ('methodName', 'getValue'), ('returnTypeName', 'TopicAssignment')])\n"
//                              "deref_3OutForAggregationComp8(methodCall_0OutFor__getKey, methodCall_2OutFor__getValue) <= APPLY (methodCall_2OutFor_AggregationComp8(methodCall_2OutFor__getValue), methodCall_2OutFor_AggregationComp8(methodCall_0OutFor__getKey), 'AggregationComp_8', 'deref_3')\n"
//                              "\n"
//                              "/* Apply aggregation */\n"
//                              "aggOutForAggregationComp8 (aggOutFor8)<= AGGREGATE (deref_3OutForAggregationComp8(methodCall_0OutFor__getKey, methodCall_2OutFor__getValue),'AggregationComp_8')\n"
//                              "\n"
//                              "/* Apply MultiSelection filtering */\n"
//                              "nativ_0OutForMultiSelectionComp9(aggOutFor8,nativ_0_9OutFor) <= APPLY (aggOutForAggregationComp8(aggOutFor8), aggOutForAggregationComp8(aggOutFor8), 'MultiSelectionComp_9', 'native_lambda_0', [('lambdaType', 'native_lambda')])\n"
//                              "filteredInputForMultiSelectionComp9(aggOutFor8) <= FILTER (nativ_0OutForMultiSelectionComp9(nativ_0_9OutFor), nativ_0OutForMultiSelectionComp9(aggOutFor8), 'MultiSelectionComp_9')\n"
//                              "\n"
//                              "/* Apply MultiSelection projection */\n"
//                              "nativ_1OutForMultiSelectionComp9 (nativ_1_9OutFor) <= APPLY (filteredInputForMultiSelectionComp9(aggOutFor8), filteredInputForMultiSelectionComp9(), 'MultiSelectionComp_9', 'native_lambda_1', [('lambdaType', 'native_lambda')])\n"
//                              "flattenedOutForMultiSelectionComp9(flattened_nativ_1_9OutFor) <= FLATTEN (nativ_1OutForMultiSelectionComp9(nativ_1_9OutFor), nativ_1OutForMultiSelectionComp9(), 'MultiSelectionComp_9')\n"
//                              "\n"
//                              "/* Extract key for aggregation */\n"
//                              "methodCall_0OutFor_AggregationComp10(flattened_nativ_1_9OutFor,methodCall_0OutFor__getKey) <= APPLY (flattenedOutForMultiSelectionComp9(flattened_nativ_1_9OutFor), flattenedOutForMultiSelectionComp9(flattened_nativ_1_9OutFor), 'AggregationComp_10', 'methodCall_0', [('inputTypeName', 'LDATopicWordProb'), ('lambdaType', 'methodCall'), ('methodName', 'getKey'), ('returnTypeName', 'LDATopicWordProb')])\n"
//                              "deref_1OutForAggregationComp10(flattened_nativ_1_9OutFor, methodCall_0OutFor__getKey) <= APPLY (methodCall_0OutFor_AggregationComp10(methodCall_0OutFor__getKey), methodCall_0OutFor_AggregationComp10(flattened_nativ_1_9OutFor), 'AggregationComp_10', 'deref_1')\n"
//                              "\n"
//                              "/* Extract value for aggregation */\n"
//                              "methodCall_2OutFor_AggregationComp10(methodCall_0OutFor__getKey,methodCall_2OutFor__getValue) <= APPLY (deref_1OutForAggregationComp10(flattened_nativ_1_9OutFor), deref_1OutForAggregationComp10(methodCall_0OutFor__getKey), 'AggregationComp_10', 'methodCall_2', [('inputTypeName', 'LDATopicWordProb'), ('lambdaType', 'methodCall'), ('methodName', 'getValue'), ('returnTypeName', 'LDATopicWordProb')])\n"
//                              "deref_3OutForAggregationComp10(methodCall_0OutFor__getKey, methodCall_2OutFor__getValue) <= APPLY (methodCall_2OutFor_AggregationComp10(methodCall_2OutFor__getValue), methodCall_2OutFor_AggregationComp10(methodCall_0OutFor__getKey), 'AggregationComp_10', 'deref_3')\n"
//                              "\n"
//                              "/* Apply aggregation */\n"
//                              "aggOutForAggregationComp10 (aggOutFor10)<= AGGREGATE (deref_3OutForAggregationComp10(methodCall_0OutFor__getKey, methodCall_2OutFor__getValue),'AggregationComp_10')\n"
//                              "aggOutForAggregationComp10_out( ) <= OUTPUT ( aggOutForAggregationComp10 ( aggOutFor10 ), 'LDA_db', 'TopicsPerWord1', 'SetWriter_11')\n"
//                              "\n"
//                              "/* Apply MultiSelection filtering */\n"
//                              "nativ_0OutForMultiSelectionComp12(nativ_1_6OutFor,nativ_0_12OutFor) <= APPLY (nativ_1OutForSelectionComp6(nativ_1_6OutFor), nativ_1OutForSelectionComp6(nativ_1_6OutFor), 'MultiSelectionComp_12', 'native_lambda_0', [('lambdaType', 'native_lambda')])\n"
//                              "filteredInputForMultiSelectionComp12(nativ_1_6OutFor) <= FILTER (nativ_0OutForMultiSelectionComp12(nativ_0_12OutFor), nativ_0OutForMultiSelectionComp12(nativ_1_6OutFor), 'MultiSelectionComp_12')\n"
//                              "\n"
//                              "/* Apply MultiSelection projection */\n"
//                              "methodCall_1OutFor_MultiSelectionComp12(nativ_1_6OutFor,methodCall_1OutFor__getDocAssigns) <= APPLY (filteredInputForMultiSelectionComp12(nativ_1_6OutFor), filteredInputForMultiSelectionComp12(nativ_1_6OutFor), 'MultiSelectionComp_12', 'methodCall_1', [('inputTypeName', 'LDADocWordTopicAssignment'), ('lambdaType', 'methodCall'), ('methodName', 'getDocAssigns'), ('returnTypeName', 'LDADocWordTopicAssignment')])\n"
//                              "deref_2OutForMultiSelectionComp12 (methodCall_1OutFor__getDocAssigns) <= APPLY (methodCall_1OutFor_MultiSelectionComp12(methodCall_1OutFor__getDocAssigns), methodCall_1OutFor_MultiSelectionComp12(), 'MultiSelectionComp_12', 'deref_2')\n"
//                              "flattenedOutForMultiSelectionComp12(flattened_methodCall_1OutFor__getDocAssigns) <= FLATTEN (deref_2OutForMultiSelectionComp12(methodCall_1OutFor__getDocAssigns), deref_2OutForMultiSelectionComp12(), 'MultiSelectionComp_12')\n"
//                              "\n"
//                              "/* Extract key for aggregation */\n"
//                              "methodCall_0OutFor_AggregationComp13(flattened_methodCall_1OutFor__getDocAssigns,methodCall_0OutFor__getKey) <= APPLY (flattenedOutForMultiSelectionComp12(flattened_methodCall_1OutFor__getDocAssigns), flattenedOutForMultiSelectionComp12(flattened_methodCall_1OutFor__getDocAssigns), 'AggregationComp_13', 'methodCall_0', [('inputTypeName', 'DocAssignment'), ('lambdaType', 'methodCall'), ('methodName', 'getKey'), ('returnTypeName', 'DocAssignment')])\n"
//                              "deref_1OutForAggregationComp13(flattened_methodCall_1OutFor__getDocAssigns, methodCall_0OutFor__getKey) <= APPLY (methodCall_0OutFor_AggregationComp13(methodCall_0OutFor__getKey), methodCall_0OutFor_AggregationComp13(flattened_methodCall_1OutFor__getDocAssigns), 'AggregationComp_13', 'deref_1')\n"
//                              "\n"
//                              "/* Extract value for aggregation */\n"
//                              "methodCall_2OutFor_AggregationComp13(methodCall_0OutFor__getKey,methodCall_2OutFor__getValue) <= APPLY (deref_1OutForAggregationComp13(flattened_methodCall_1OutFor__getDocAssigns), deref_1OutForAggregationComp13(methodCall_0OutFor__getKey), 'AggregationComp_13', 'methodCall_2', [('inputTypeName', 'DocAssignment'), ('lambdaType', 'methodCall'), ('methodName', 'getValue'), ('returnTypeName', 'DocAssignment')])\n"
//                              "deref_3OutForAggregationComp13(methodCall_0OutFor__getKey, methodCall_2OutFor__getValue) <= APPLY (methodCall_2OutFor_AggregationComp13(methodCall_2OutFor__getValue), methodCall_2OutFor_AggregationComp13(methodCall_0OutFor__getKey), 'AggregationComp_13', 'deref_3')\n"
//                              "\n"
//                              "/* Apply aggregation */\n"
//                              "aggOutForAggregationComp13 (aggOutFor13)<= AGGREGATE (deref_3OutForAggregationComp13(methodCall_0OutFor__getKey, methodCall_2OutFor__getValue),'AggregationComp_13')\n"
//                              "\n"
//                              "/* Apply selection filtering */\n"
//                              "nativ_0OutForSelectionComp14(aggOutFor13,nativ_0_14OutFor) <= APPLY (aggOutForAggregationComp13(aggOutFor13), aggOutForAggregationComp13(aggOutFor13), 'SelectionComp_14', 'native_lambda_0', [('lambdaType', 'native_lambda')])\n"
//                              "filteredInputForSelectionComp14(aggOutFor13) <= FILTER (nativ_0OutForSelectionComp14(nativ_0_14OutFor), nativ_0OutForSelectionComp14(aggOutFor13), 'SelectionComp_14')\n"
//                              "\n"
//                              "/* Apply selection projection */\n"
//                              "nativ_1OutForSelectionComp14 (nativ_1_14OutFor) <= APPLY (filteredInputForSelectionComp14(aggOutFor13), filteredInputForSelectionComp14(), 'SelectionComp_14', 'native_lambda_1', [('lambdaType', 'native_lambda')])\n"
//                              "nativ_1OutForSelectionComp14_out( ) <= OUTPUT ( nativ_1OutForSelectionComp14 ( nativ_1_14OutFor ), 'LDA_db', 'TopicsPerDoc1', 'SetWriter_15')";
//
//  // get the string to compile
//  myLogicalPlan.push_back('\0');
//
//  // where the result of the parse goes
//  AtomicComputationList *myResult;
//
//  // now, do the compilation
//  yyscan_t scanner;
//  LexerExtra extra{""};
//  yylex_init_extra(&extra, &scanner);
//  const YY_BUFFER_STATE buffer{yy_scan_string(myLogicalPlan.data(), scanner)};
//  const int parseFailed{yyparse(scanner, &myResult)};
//  yy_delete_buffer(buffer, scanner);
//  yylex_destroy(scanner);
//
//  // if it didn't parse, get outta here
//  if (parseFailed) {
//    std::cout << "Parse error when compiling TCAP: " << extra.errorMessage;
//    exit(1);
//  }
//
//  // this is the logical plan to return
//  auto atomicComputations = std::shared_ptr<AtomicComputationList>(myResult);
//
//  pdb::PDBPipeNodeBuilder factory(3, atomicComputations);
//
//  set<pdb::PDBAbstractPhysicalNodePtr> check;
//  auto out = factory.generateAnalyzerGraph();
//
//  for (int j = 0; j < out.size(); ++j) {
//
//    std::cout << "Pipeline " << j << std::endl;
//
//    auto &o = out[j];
//
//    for (auto &c : o->getPipeComputations()) {
//      std::cout << c->getOutputName() << std::endl;
//    }
//
//    std::cout << "Num consumers " << o->getConsumers().size() << std::endl;
//
//    for (auto &con : o->getConsumers()) {
//
//      if (check.find(con) == check.end()) {
//        out.push_back(con);
//        check.insert(con);
//      }
//
//      std::cout << "Starts  : " << con->getPipeComputations().front()->getOutputName() << std::endl;
//    }
//
//    std::cout << "\n\n\n";
//  }
//
//}
//
//TEST(TestPipeBuilder, Test5) {
//
//  std::string myLogicalPlan = "inputDataForSetScanner_0(in0) <= SCAN ('chris_db', 'input_set1', 'SetScanner_0')\n"
//                              "inputDataForSetScanner_1(in1) <= SCAN ('chris_db', 'input_set2', 'SetScanner_1')\n"
//                              "unionOutUnionComp2 (unionOutFor2 )<= UNION (inputDataForSetScanner_0(in0), inputDataForSetScanner_1(in1),'UnionComp_2')\n"
//                              "unionOutUnionComp2_out( ) <= OUTPUT ( unionOutUnionComp2 ( unionOutFor2 ), 'chris_db', 'outputSet', 'SetWriter_3')";
//
//  // get the string to compile
//  myLogicalPlan.push_back('\0');
//
//  // where the result of the parse goes
//  AtomicComputationList *myResult;
//
//  // now, do the compilation
//  yyscan_t scanner;
//  LexerExtra extra{""};
//  yylex_init_extra(&extra, &scanner);
//  const YY_BUFFER_STATE buffer{yy_scan_string(myLogicalPlan.data(), scanner)};
//  const int parseFailed = yyparse(scanner, &myResult);
//  yy_delete_buffer(buffer, scanner);
//  yylex_destroy(scanner);
//
//  // if it didn't parse, get outta here
//  if (parseFailed) {
//    std::cout << "Parse error when compiling TCAP: " << extra.errorMessage;
//    exit(1);
//  }
//
//  // this is the logical plan to return
//  auto atomicComputations = std::shared_ptr<AtomicComputationList>(myResult);
//
//  pdb::PDBPipeNodeBuilder factory(3, atomicComputations);
//
//  set<pdb::PDBAbstractPhysicalNodePtr> check;
//  auto out = factory.generateAnalyzerGraph();
//
//  for (int j = 0; j < out.size(); ++j) {
//
//    std::cout << "\nPipeline " << j << std::endl;
//
//    auto &o = out[j];
//
//    for (auto &c : o->getPipeComputations()) {
//      std::cout << c->getOutputName() << std::endl;
//    }
//
//    std::cout << "Num consumers " << o->getConsumers().size() << std::endl;
//
//    for (auto &con : o->getConsumers()) {
//
//      if (check.find(con) == check.end()) {
//        out.push_back(con);
//        check.insert(con);
//      }
//
//      std::cout << "Starts  : " << con->getPipeComputations().front()->getOutputName() << std::endl;
//    }
//
//    std::cout << "\n\n\n";
//  }
//
//}
