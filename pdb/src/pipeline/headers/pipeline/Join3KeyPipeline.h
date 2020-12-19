#pragma once

#include <unordered_map>
#include <PDBAbstractPageSet.h>

namespace pdb {

class Join3KeyPipeline {
 public:

  explicit Join3KeyPipeline(std::unordered_map<int32_t, PDBAbstractPageSetPtr> &keySourcePageSets0,
                            std::unordered_map<int32_t, PDBAbstractPageSetPtr> &keySourcePageSets1,
                            std::unordered_map<int32_t, PDBAbstractPageSetPtr> &keySourcePageSets2)
      : keySourcePageSets0(keySourcePageSets0),
        keySourcePageSets1(keySourcePageSets1),
        keySourcePageSets2(keySourcePageSets2){}

  void runSide(int32_t node, int32_t set);

  void runJoin();

  // where we get the keys
  std::unordered_map<int32_t, PDBAbstractPageSetPtr> &keySourcePageSets0;
  std::unordered_map<int32_t, PDBAbstractPageSetPtr> &keySourcePageSets1;
  std::unordered_map<int32_t, PDBAbstractPageSetPtr> &keySourcePageSets2;

  struct joined_record {
    int32_t first;
    int32_t second;
    int32_t third;
    friend bool operator==(const joined_record &lhs, const joined_record &rhs);
    friend bool operator!=(const joined_record &lhs, const joined_record &rhs);
    joined_record() = default;
  };


  struct key {
    int32_t rowID;
    int32_t colID;
    friend bool operator==(const key &lhs, const key &rhs);
    friend bool operator!=(const key &lhs, const key &rhs);
  };

  class HashFunction {
   public:

    // Use sum of lengths of first and last names
    // as hash function.
    size_t operator()(const key &k) const {
      return k.rowID + k.colID * 10000;
    }
  };

  // these are the node records
  std::unordered_map<key, std::pair<int32_t, int32_t>, HashFunction> nodeRecords0;
  std::unordered_map<key, std::pair<int32_t, int32_t>, HashFunction> nodeRecords1;
  std::unordered_map<key, std::pair<int32_t, int32_t>, HashFunction> nodeRecords2;

  // the joined records
  std::vector<joined_record> joined;

  // we use this to sync
  std::mutex m0;
  std::mutex m1;
  std::mutex m2;

  // assign a tid
  int tid0 = 0;
  int tid1 = 0;
  int tid2 = 0;

};

}