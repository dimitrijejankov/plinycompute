#pragma once

//  PRELOAD %TRABlockMeta%

#include <Object.h>
#include <PDBVector.h>

namespace pdb {

class TRABlockMeta : public pdb::Object {
public:

  /**
   * The default constructor
   *
   */
  TRABlockMeta() = default;

  TRABlockMeta(uint32_t key0, uint32_t key1) : indices(2, 2) {
    indices[0] = key0;
    indices[1] = key1;
  }

  TRABlockMeta(uint32_t key0, uint32_t key1, uint32_t key2) : indices(3, 3) {
    indices[0] = key0;
    indices[1] = key1;
    indices[2] = key2;
  }

  TRABlockMeta(uint32_t key0, uint32_t key1, uint32_t key2, uint32_t key3) : indices(4, 4) {
    indices[0] = key0;
    indices[1] = key1;
    indices[2] = key2;
    indices[3] = key3;
  }

  ENABLE_DEEP_COPY

  pdb::Vector<uint32_t> indices;

  TRABlockMeta getKey02() {
      TRABlockMeta meta02(indices[0], 0, indices[2]);
      return meta02;
  }

  uint32_t getIdx0() {
    return indices[0];
  }

  uint32_t getIdx1() {
    return indices[1];
  }

  uint32_t getIdx2() {
    return indices[2];
  }

  bool operator==(const TRABlockMeta &other) const {
    return indices[0] == other.indices[0] && indices[1] == other.indices[1];
  }

  size_t hash() const {
    return 1000000 * indices[0] + 1000 * indices[1];
  }
};

}