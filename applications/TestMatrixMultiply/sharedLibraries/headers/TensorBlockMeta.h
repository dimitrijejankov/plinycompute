#pragma once

#include <Object.h>
#include <PDBVector.h>

namespace pdb {

// the sub namespace
namespace matrix {

class TensorBlockMeta : public pdb::Object {
public:

  /**
   * The default constructor
   *
   */
  TensorBlockMeta() = default;

  TensorBlockMeta(uint32_t key0, uint32_t key1, uint32_t key2) : indices(3,3) {
    indices[0] = key0;
    indices[1] = key1;
    indices[2] = key2;
  }

  ENABLE_DEEP_COPY

  pdb::Vector<uint32_t> indices;

  TensorBlockMeta getKey02() {
      TensorBlockMeta meta02(indices[0], 0, indices[2]);
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

  bool operator==(const TensorBlockMeta &other) const {
    return indices[0] == other.indices[0] && indices[1] == other.indices[1] && indices[2] == other.indices[2];
  }

  size_t hash() const {
    return 1000000 * indices[0] + 1000 * indices[1] + indices[2];
  }
};

}
}