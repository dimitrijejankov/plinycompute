/*****************************************************************************
 *                                                                           *
 *  Copyright 2018 Rice University                                           *
 *                                                                           *
 *  Licensed under the Apache License, Version 2.0 (the "License");          *
 *  you may not use this file except in compliance with the License.         *
 *  You may obtain a copy of the License at                                  *
 *                                                                           *
 *      http://www.apache.org/licenses/LICENSE-2.0                           *
 *                                                                           *
 *  Unless required by applicable law or agreed to in writing, software      *
 *  distributed under the License is distributed on an "AS IS" BASIS,        *
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. *
 *  See the License for the specific language governing permissions and      *
 *  limitations under the License.                                           *
 *                                                                           *
 *****************************************************************************/

#include <PDBClient.h>
#include <StringIntPair.h>
#include <SumResult.h>
#include <ReadInt.h>
#include <ReadStringIntPair.h>
#include <StringSelectionOfStringIntPair.h>
#include <IntSimpleJoin.h>
#include <WriteSumResult.h>
#include <IntAggregation.h>

using namespace pdb;

const size_t blockSize = 64;
const size_t replicateSet1 = 3;
const size_t repilcateSet2 = 2;

// the number of keys that are going to be joined
size_t numToJoin = std::numeric_limits<size_t>::max();

void fillSet1(PDBClient &pdbClient){

  // make the allocation block
  const pdb::UseTemporaryAllocationBlock tempBlock{blockSize * 1024 * 1024};

  // write a bunch of supervisors to it
  Handle<Vector<Handle<int>>> data = pdb::makeObject<Vector<Handle<int>>>();
  size_t i = 0;
  try {

    // fill the vector up
    for (; true; i++) {
      Handle<int> myInt = makeObject<int>(i);
      data->push_back(myInt);
    }

  } catch (pdb::NotEnoughSpace &e) {

    // remove the last int
    data->pop_back();

    // how many did we have
    numToJoin = std::min(numToJoin, i - 1);

    // send the data a bunch of times
    for(size_t j = 0; j < replicateSet1; ++j) {
      pdbClient.sendData<int>("test78_db", "test78_set1", data);
    }
  }
}

void fillSet2(PDBClient &pdbClient) {

  // make the allocation block
  const pdb::UseTemporaryAllocationBlock tempBlock{blockSize * 1024 * 1024};

  // write a bunch of supervisors to it
  Handle <Vector <Handle <StringIntPair>>> data = pdb::makeObject<Vector <Handle <StringIntPair>>>();

  size_t i = 0;
  try {

    // fill the vector up
    for (; true; i++) {
      std::ostringstream oss;
      oss << "My string is " << i;
      oss.str();
      Handle <StringIntPair> myPair = makeObject <StringIntPair> (oss.str (), i);
      data->push_back (myPair);
    }

  } catch (pdb::NotEnoughSpace &e) {

    // remove the last string int pair
    data->pop_back();

    // how many did we have
    numToJoin = std::min(numToJoin, i - 1);

    // send the data a bunch of times
    for(size_t j = 0; j < repilcateSet2; ++j) {
      pdbClient.sendData<StringIntPair>("test78_db", "test78_set2", data);
    }
  }
}

int main(int argc, char* argv[]) {

  // make the client
  PDBClient pdbClient(8108, "localhost");

  /// 1. Register the classes

  pdbClient.registerType("libraries/libReadInt.so");
  pdbClient.registerType("libraries/libReadStringIntPair.so");
  pdbClient.registerType("libraries/libStringSelectionOfStringIntPair.so");
  pdbClient.registerType("libraries/libIntSimpleJoin.so");
  pdbClient.registerType("libraries/libIntAggregation.so");
  pdbClient.registerType("libraries/libWriteSumResult.so");

  /// 2. Create the set

  // now, create a new database
  pdbClient.createDatabase("test78_db");

  // now, create the int set in that database
  pdbClient.createSet<int>("test78_db", "test78_set1");

  // now, create the StringIntPair set in that database
  pdbClient.createSet<StringIntPair>("test78_db", "test78_set2");

  // now, create a new set in that database to store output data
  pdbClient.createSet<SumResult>("test78_db", "output_set1");

  /// 3. Fill in the data (single threaded)

  fillSet1(pdbClient);
  fillSet2(pdbClient);

  /// 4. Make query graph an run query

  // this is the object allocation block where all of this stuff will reside
  const UseTemporaryAllocationBlock tempBlock{1024 * 1024 * 128};

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


  // here is the list of computations
  Handle<Vector<Handle<Computation>>> myComputations = makeObject<Vector<Handle<Computation>>>();

  // create all of the computation objects
  Handle<Computation> myScanSet1 = makeObject<ReadInt>("test78_db", "test78_set1");
  Handle<Computation> myScanSet2 = makeObject<ReadStringIntPair>("test78_db", "test78_set2");
  Handle<Computation> mySelection = makeObject<StringSelectionOfStringIntPair>();
  mySelection->setInput(myScanSet2);
  Handle<Computation> myJoin = makeObject<IntSimpleJoin>();
  myJoin->setInput(0, myScanSet1);
  myJoin->setInput(1, myScanSet2);
  myJoin->setInput(2, mySelection);
  Handle<Computation> myAggregation = makeObject<IntAggregation>();
  myAggregation->setInput(myJoin);
  Handle<Computation> myWriter = makeObject<WriteSumResult>("test78_db", "output_set1");
  myWriter->setInput(myAggregation);

  // put them in the list of computations
  myComputations->push_back(myScanSet1);
  myComputations->push_back(myScanSet2);
  myComputations->push_back(mySelection);
  myComputations->push_back(myJoin);
  myComputations->push_back(myAggregation);
  myComputations->push_back(myWriter);

  // TODO this is just a preliminary version of the execute computation before we add back the TCAP generation
  pdbClient.executeComputations(myComputations, tcap);

  /// 5. Evaluate the results

  // grab the iterator
  auto it = pdbClient.getSetIterator<SumResult>("test78_db", "output_set1");
  while(it->hasNextRecord()) {

    // grab the record
    auto r = it->getNextRecord();
    std::cout <<"Identifier : " << r->identifier << " Value : " << r->total << std::endl;
  }
}