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

#include <gtest/gtest.h>
#include <Supervisor.h>

#include "Handle.h"
#include "PDBVector.h"
#include "InterfaceFunctions.h"
#include "IntDoubleVectorPair.h"

//a test case for pdb::Vector

using namespace pdb;

TEST(VectorTest, VectorOfVectors) {

  // the constants
  const int numWord = 10;
  const int numTopic = 4;

  // load up the allocator with RAM
  makeObjectAllocatorBlock(1024 * 1024 * 24, true);

  // create a vector
  Handle<Vector<Handle<IntDoubleVectorPair>>>
      result = makeObject<Vector<Handle<IntDoubleVectorPair>>>(numWord, numWord);

  // setup
  for (int i = 0; i < numWord; i++) {

    // create the vector of doubles of size numTopic
    Handle<Vector<double>> topicProb = makeObject<Vector<double>>(numTopic, numTopic);

    // fill it up with is
    topicProb->fill(1.0 * i);

    // make the IntDoubleVectorPair
    Handle<IntDoubleVectorPair> wordTopicProb = makeObject<IntDoubleVectorPair>(i, topicProb);
    (*result)[i] = wordTopicProb;
  }

  // test
  for (int i = 0; i < numWord; i++) {

    // grab the int double pair
    Handle<IntDoubleVectorPair> intDoublePair = (*result)[i];

    // check if the int is ok
    EXPECT_EQ(intDoublePair->myInt, i);

    // grab the vector
    EXPECT_EQ(intDoublePair->myVector.size(), numTopic);

    // check the vector
    for (int j = 0; j < numTopic; ++j) {
      EXPECT_EQ(intDoublePair->myVector[j], 1.0 * i);
    }
  }
}

TEST(VectorTest, SerializeVector) {

  // how many objects
  const int NUM_OBJECTS = 12000;

  // we are going to put a copy here
  void *buffer = malloc(1024 * 1024 * 24);

  {
    // load up the allocator with RAM
    makeObjectAllocatorBlock(1024 * 1024 * 24, true);

    // create a vector
    Handle<Vector<Handle<Supervisor>>> supers = makeObject<Vector<Handle<Supervisor>>>(10);

    try {

      // put a lot of copies of it into a vector
      for (int i = 0; i < NUM_OBJECTS; i++) {

        // push the object
        supers->push_back(makeObject<Supervisor>("Joe Johnson", 20 + (i % 29)));

        // create 10 employee objects and push them
        for (int j = 0; j < 10; j++) {
          Handle<Employee> temp = makeObject<Employee>("Steve Stevens", 20 + ((i + j) % 29));
          (*supers)[supers->size() - 1]->addEmp(temp);
        }
      }

    } catch (NotEnoughSpace &e) {

      // so we are out of memory on the block finish
      FAIL();
    }

    // grab the bytes
    Record<Vector<Handle<Supervisor>>>* myBytes = getRecord<Vector<Handle<Supervisor>>>(supers);

    // the record must be less than the allocation block or else we have a problem
    EXPECT_LE(myBytes->numBytes(), 1024 * 1024 * 24);

    // copy the vector to a some buffer
    memcpy(buffer, myBytes, myBytes->numBytes());
  }

  // cast the place where we copied the thing
  auto* recordCopy = (Record<Vector<Handle<Supervisor>>>*) buffer;

  // grab the copy of the supervisor object
  Handle<Vector<Handle<Supervisor>>> mySupers = recordCopy->getRootObject();

  // check if the number of objects is right
  EXPECT_EQ(NUM_OBJECTS, mySupers->size());

  // check if the objects inside the vectors are ok
  for (int i = 0; i < NUM_OBJECTS; i++) {

    // grab the supervisor
    Handle<Supervisor> temp = (*mySupers)[i];

    // check if it is ok
    EXPECT_TRUE(*temp->me->getName() == "Joe Johnson");
    EXPECT_EQ(temp->me->getAge(), 20 + (i % 29));
  }

  // free
  free(buffer);
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
