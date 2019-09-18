#include <gtest/gtest.h>
#include <JoinMap.h>
#include <JoinTuple.h>
#include <UseTemporaryAllocationBlock.h>
#include <StringIntPair.h>
#include "../../../../applications/TestMatrixMultiply/sharedLibraries/headers/MatrixBlock.h"

namespace pdb {

using namespace matrix;

TEST(TestJoinMapWithJoinTuple, Test1) {

  //using doubleTuple = pdb::JoinTuple<matrix::MatrixBlock, char[0]>;
  using doubleTuple = pdb::JoinTuple<MatrixBlock, pdb::JoinTuple<MatrixBlock, char[0]>>;

  doubleTuple t;
  using keyTuple = decltype(t.getKey());

  keyTuple tt;

  return;
}

}
