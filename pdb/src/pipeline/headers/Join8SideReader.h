#pragma once

#include <cstdint>
#include <PDBAbstractPageSet.h>
#include <JoinPlannerResult.h>
#include <Join8SideSender.h>
#include "../../../../applications/TestConvolution/sharedLibraries/headers/MatrixBlock.h"

namespace pdb {

class Join8SideReader {
public:

  Join8SideReader(pdb::PDBAbstractPageSetPtr pageSet,
                  int32_t workerID,
                  int32_t numNodes,
                  std::shared_ptr<std::vector<Join8SideSenderPtr>> joinSideSenders,
                  PDBPageHandle page);

  void run();

  int32_t workerID;
  int32_t numNodes;

  PDBAbstractPageSetPtr pageSet;
  PDBPageHandle planPage;
  Handle<JoinPlannerResult> planResult;
  std::vector<std::vector<pdb::Handle<MatrixBlock>>> toSend;
  std::shared_ptr<std::vector<Join8SideSenderPtr>> joinSideSenders;
};

using Join8SideReaderPtr = std::shared_ptr<Join8SideReader>;

}