#pragma once

#include <PDBAnonymousPageSet.h>
#include <JoinTuple.h>
#include <PipJoinAggPlanResult.h>
#include "../../../../applications/TestMatrixMultiply/sharedLibraries/headers/MatrixBlock.h"

namespace pdb {


class JoinMapCreator {
public:

  using record_t = pdb::matrix::MatrixBlock;
  using tuple_t = JoinTuple<record_t, char[0]>;

  JoinMapCreator() = default;

  JoinMapCreator(int32_t numThreads,
                 int32_t nodeId,
                 bool isLeft,
                 PDBPageHandle planPage,
                 PDBAnonymousPageSetPtr pageSet,
                 PDBCommunicatorPtr communicator,
                 PDBPageHandle page,
                 PDBLoggerPtr logger);
  void run();

  bool getSuccess();

  const std::string &getError();

private:

  // the number of threads
  int32_t numThreads{};

  // the id of this node
  int32_t nodeID{};

  // is it the left side or right side
  bool isLeft{};

  // the plan page
  pdb::PDBPageHandle planPage;

  // the page set we are writing to
  PDBAnonymousPageSetPtr pageSet;

  // the communicator we are getting stuff from
  PDBCommunicatorPtr communicator;

  // the page we write stuff to
  PDBPageHandle page;

  // the logger
  PDBLoggerPtr logger;

  // did we succeed
  bool success = true;

  // was there an error
  std::string error;
};

// make the shared ptr for this
using JoinMapCreatorPtr = std::shared_ptr<JoinMapCreator>;
}