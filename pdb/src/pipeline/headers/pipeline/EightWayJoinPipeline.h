#pragma once

#include <unordered_map>
#include <PDBAbstractPageSet.h>

namespace pdb {

class EightWayJoinPipeline {
public:

  explicit EightWayJoinPipeline(std::unordered_map<int32_t , PDBAbstractPageSetPtr> &keySourcePageSets) : keySourcePageSets(keySourcePageSets) {}

  void runSide(int node);

  void runJoin();

  // where we get the keys
  std::unordered_map<int32_t , PDBAbstractPageSetPtr> &keySourcePageSets;

  using key = std::tuple<int32_t, int32_t, int32_t>;

  using joined_record = std::tuple<int32_t, int32_t, int32_t, int32_t, int32_t, int32_t, int32_t, int32_t>;

  class HashFunction {
   public:

    // Use sum of lengths of first and last names
    // as hash function.
    size_t operator()(const key& k) const
    {
      return std::get<0>(k) + std::get<1>(k) * 100 + std::get<2>(k) * 10000;
    }
  };

  // these are the node records
  std::unordered_map<key, std::pair<int32_t, int32_t>, HashFunction> nodeRecords;

  // the joined records
  std::vector<joined_record> joined;

  // we use this to sync
  std::mutex m;

  // assign a tid
  int tid = 0;


};

}