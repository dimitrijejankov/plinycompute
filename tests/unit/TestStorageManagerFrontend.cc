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

TEST(StorageManagerFrontendTest, Test1) {

  // create the frontend
  pdb::PDBStorageManagerFrontEnd frontEnd("tempDSFSD", 64, 16, "metadata", ".");

  // call the init method the server would usually call
  frontEnd.init();



}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
