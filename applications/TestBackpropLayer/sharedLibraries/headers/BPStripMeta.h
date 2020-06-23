#pragma once

#include <Object.h>

namespace pdb {

// the sub namespace
namespace bp {

class BPStripMeta : public pdb::Object {
 public:

  /**
   * The default constructor
   */
  BPStripMeta() = default;

  BPStripMeta(uint32_t batchID) : batchID(batchID){}

  ENABLE_DEEP_COPY

  /**
   * The id of the batch strip this block belongs to
   */
  uint32_t batchID = 0;

  bool operator==(const BPStripMeta &other) const {
    return batchID == other.batchID;
  }

  size_t hash() const {
    return batchID;
  }
};

}
}