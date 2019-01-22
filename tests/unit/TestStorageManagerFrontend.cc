//
// Created by dimitrije on 1/21/19.
//
#include <cstring>
#include <iostream>
#include <time.h>
#include <unistd.h>
#include <vector>
#include <thread>
#include <random>
#include <gtest/gtest.h>
#include <gtest/gtest.h>

#include <PDBStorageManagerFrontEnd.h>

class CommunicatorMock {

public:

  template <class ObjType>
  bool sendObject(pdb::Handle<ObjType>& sendMe, std::string& errMsg) {

    return true;
  }

};

TEST(StorageManagerFrontendTest, Test1) {

  // create the frontend
  pdb::PDBStorageManagerFrontEnd frontEnd("tempDSFSD", 64, 16, "metadata", ".");

  // call the init method the server would usually call
  frontEnd.init();

  // create a get page request
  pdb::Handle<pdb::StoGetPageRequest> request = pdb::makeObject<pdb::StoGetPageRequest>("set1", "db1", 0);

  // make the mock communicator
  auto comm = std::make_shared<CommunicatorMock>();

  frontEnd.handleGetPageRequest(request, comm);
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
