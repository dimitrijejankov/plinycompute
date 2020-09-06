#pragma once

#include <Object.h>

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

  TensorBlockMeta(uint32_t key0, uint32_t key1, uint32_t key2) :
          key0(key0), key1(key1), key2(key2) {}

  ENABLE_DEEP_COPY
  //The row position of the block if interpreted as matrix
  uint32_t key0 = 0;

  // The column position of the block if interpreted as matrix
  uint32_t key1 = 0;

  uint32_t key2 = 0;

  TensorBlockMeta getKey02() {
      TensorBlockMeta meta02(key0, 0, key2);
      return meta02;
  }

  bool operator==(const TensorBlockMeta &other) const {
    return key0 == other.key0 && key1 == other.key1 && key2 == other.key2;
  }

  size_t hash() const {
    return 1000000 * key0 + 1000*key1 + key2;
  }
};

}
}