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
#include <gmock/gmock.h>

#include <PDBStorageManagerFrontEnd.h>

class CommunicatorMock {

public:

  MOCK_METHOD2(sendObject, bool(pdb::Handle<pdb::StoGetPageResult>&, std::string& errMsg));

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

  // make sure the mock function returns true
  ON_CALL(*comm, sendObject).WillByDefault(testing::Return(true));

  // it should call send object exactly once
  EXPECT_CALL(*comm, sendObject).Times(1);

  frontEnd.handleGetPageRequest(request, comm);
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
