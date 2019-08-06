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

  // execute the computation
  pdbClient.executeComputations({ myWriter });

  /// 5. Evaluate the results

  // grab the iterator
  auto it = pdbClient.getSetIterator<SumResult>("test78_db", "output_set1");
  while(it->hasNextRecord()) {

    // grab the record
    auto r = it->getNextRecord();
    std::cout << "Value : " << r->getValue() << " Key : " << r->getKey() << std::endl;
  }

  // shutdown the server
  pdbClient.shutDownServer();
}