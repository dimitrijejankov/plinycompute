#include <gtest/gtest.h>
#include <JoinMap.h>
#include <JoinTuple.h>
#include <UseTemporaryAllocationBlock.h>
#include <StringIntPair.h>

TEST(TestJoinMapWithJoinTuple, Test1) {

  // allocate the memory
  void *memory = malloc(1024 * 1024);

  // allocate the memory where I want to copy
  void *copyMemory = malloc(1024 * 1024);

  {
    const pdb::UseTemporaryAllocationBlock tempBlock{memory, 1024 * 1024};

    // create a join map
    pdb::Handle<pdb::JoinMap<pdb::JoinTuple<pdb::StringIntPair, char[0]>>> joinMap = pdb::makeObject<pdb::JoinMap<pdb::JoinTuple<pdb::StringIntPair, char[0]>>>();

    // store the records
    for(size_t i = 0; i < 100; ++i) {

      // add a record to the map
      auto &r = joinMap->push(i / 2);
      r = pdb::JoinTuple<pdb::StringIntPair, char[0]>();
      r.myData.myInt = (int32_t) i;
      r.myData.myString = pdb::makeObject<pdb::String>("Record " + std::to_string(i));
    }

    // use copy memory as allocation block
    const pdb::UseTemporaryAllocationBlock copyBlock{copyMemory, 1024 * 1024};

    // copy the join mpa
    pdb::Handle<pdb::JoinMap<pdb::JoinTuple<pdb::StringIntPair, char[0]>>> mapCopy = pdb::deepCopyJoinMap(joinMap);

    // clear everything in the first block
    memset(memory, 0, 1024 * 1024);

    uint64_t currentHash = 0;
    uint64_t loop = 0;

    // check if the map was actually copied
    auto it = mapCopy->begin();
    while(it != mapCopy->end()) {

      // print out the results
      for(int i = 0; i < (*it)->size(); ++i) {

        auto tmp = *it;
        EXPECT_EQ(currentHash, (*tmp).getHash());
        EXPECT_EQ(currentHash, (*tmp)[i].myData.myInt / 2);
        EXPECT_EQ((std::string) (*(*tmp)[i].myData.myString), (std::string) "Record " + std::to_string((*tmp)[i].myData.myInt));

        // every two loops we increment the hash
        loop++;
        if(loop % 2 == 0) {
          currentHash++;
        }
      }

      // go to the next value
      it.operator++();
    }
  }

  // free the copy memory
  free(copyMemory);
  free(memory);
}

TEST(TestJoinMapWithJoinTuple, Test2) {

  // allocate the memory
  void *memory = malloc(1024 * 1024);

  // allocate the memory where I want to copy
  void *copyMemory = malloc(1024 * 1024);

  {
    const pdb::UseTemporaryAllocationBlock tempBlock{memory, 1024 * 1024};

    // create a join map
    pdb::Handle<pdb::JoinMap<pdb::JoinTuple<pdb::StringIntPair, char[0]>>> joinMap = pdb::makeObject<pdb::JoinMap<pdb::JoinTuple<pdb::StringIntPair, char[0]>>>();

    // store the records
    for(size_t i = 0; i < 100; ++i) {

      // add a record to the map
      auto &r = joinMap->push(i / 2);
      r = pdb::JoinTuple<pdb::StringIntPair, char[0]>();
      r.myData.myInt = (int32_t) i;
      r.myData.myString = pdb::makeObject<pdb::String>("Record " + std::to_string(i));
    }

    // use copy memory as allocation block
    const pdb::UseTemporaryAllocationBlock copyBlock{copyMemory, 1024 * 1024};

    // copy the join mpa
    pdb::Handle<pdb::JoinMap<pdb::JoinTuple<pdb::StringIntPair, char[0]>>> mapCopy = pdb::deepCopyJoinMap(joinMap);

    // clear everything in the first block
    memset(memory, 0, 1024 * 1024);

    // just write some random stuff to the block unit we break
    pdb::Vector<int32_t> vector;
    while(true) {
      try {

        vector.push_back(342);
      }
      catch (const std::exception& e) {
        break;
      }
    }

    // stuff for testing
    uint64_t currentHash = 0;
    uint64_t loop = 0;

    // check if the map was actually copied
    auto it = mapCopy->begin();
    while(it != mapCopy->end()) {

      // print out the results
      for(int i = 0; i < (*it)->size(); ++i) {

        auto tmp = *it;
        EXPECT_EQ(currentHash, (*tmp).getHash());
        EXPECT_EQ(currentHash, (*tmp)[i].myData.myInt / 2);
        EXPECT_EQ((std::string) (*(*tmp)[i].myData.myString), (std::string) "Record " + std::to_string((*tmp)[i].myData.myInt));

        // every two loops we increment the hash
        loop++;
        if(loop % 2 == 0) {
          currentHash++;
        }
      }

      // go to the next value
      it.operator++();
    }
  }

  // free the copy memory
  free(copyMemory);
  free(memory);
}

TEST(TestJoinMapWithJoinTuple, Test3) {

  // allocate the memory
  void *memory = malloc(1024 * 1024);

  // allocate the memory where I want to copy
  void *copyMemory = malloc(1024 * 1024);

  using doubleTuple = pdb::JoinTuple<double, pdb::JoinTuple<pdb::StringIntPair, char[0]>>;

  {
    const pdb::UseTemporaryAllocationBlock tempBlock{memory, 1024 * 1024};

    // create a join map
    pdb::Handle<pdb::JoinMap<doubleTuple>> joinMap = pdb::makeObject<pdb::JoinMap<doubleTuple>>();

    // store the records
    for(size_t i = 0; i < 100; ++i) {

      // add a record to the map
      auto &t = joinMap->push(i / 2);
      t.myData = 3.14 * i;

      auto &r = t.myOtherData;
      r = pdb::JoinTuple<pdb::StringIntPair, char[0]>();
      r.myData.myInt = (int32_t) i;
      r.myData.myString = pdb::makeObject<pdb::String>("Record " + std::to_string(i));
    }

    // use copy memory as allocation block
    const pdb::UseTemporaryAllocationBlock copyBlock{copyMemory, 1024 * 1024};

    // copy the join mpa
    pdb::Handle<pdb::JoinMap<doubleTuple>> mapCopy = pdb::deepCopyJoinMap(joinMap);

    // clear everything in the first block
    memset(memory, 0, 1024 * 1024);

    // just write some random stuff to the block unit we break
    pdb::Vector<int32_t> vector;
    while(true) {
      try {

        vector.push_back(342);
      }
      catch (const std::exception& e) {
        break;
      }
    }

    // stuff for testing
    uint64_t currentHash = 0;
    uint64_t loop = 0;

    // check if the map was actually copied
    auto it = mapCopy->begin();
    while(it != mapCopy->end()) {

      // print out the results
      for(int i = 0; i < (*it)->size(); ++i) {

        auto tmp = *it;
        EXPECT_EQ(currentHash, (*tmp).getHash());

        auto &t = (*tmp)[i];
        auto &r = (*tmp)[i].myOtherData;

        EXPECT_EQ(t.myData, 3.14 * r.myData.myInt);
        EXPECT_EQ(currentHash, r.myData.myInt / 2);
        EXPECT_EQ((std::string) (*r.myData.myString), (std::string) "Record " + std::to_string(r.myData.myInt));

        // every two loops we increment the hash
        loop++;
        if(loop % 2 == 0) {
          currentHash++;
        }
      }

      // go to the next value
      it.operator++();
    }
  }

  // free the copy memory
  free(copyMemory);
  free(memory);
}