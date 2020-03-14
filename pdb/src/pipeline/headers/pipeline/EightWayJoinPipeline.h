#pragma once

#include <unordered_map>
#include <PDBAbstractPageSet.h>

namespace pdb {

class EightWayJoinPipeline {
public:

  explicit EightWayJoinPipeline(std::unordered_map<int32_t , PDBAbstractPageSetPtr> &keySourcePageSets) : keySourcePageSets(keySourcePageSets) {}

  void runSide(int node);

  // where we get the keys
  std::unordered_map<int32_t , PDBAbstractPageSetPtr> &keySourcePageSets;

};

}