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

  struct key{
    int32_t first;
    int32_t second;
    int32_t third;
    friend bool operator==(const key &lhs, const key &rhs);
    friend bool operator!=(const key &lhs, const key &rhs);
  };

  struct joined_record{int32_t first;
                                int32_t second;
                                int32_t third;
                                int32_t fourth;
                                int32_t fifth;
                                int32_t sixth;
                                int32_t seventh;
                                int32_t eight;
    friend bool operator==(const joined_record &lhs, const joined_record &rhs);
    friend bool operator!=(const joined_record &lhs, const joined_record &rhs);
  };

  class HashFunction {
   public:

    // Use sum of lengths of first and last names
    // as hash function.
    size_t operator()(const key& k) const
    {
      return k.first + k.second * 100 + k.third * 10000;
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