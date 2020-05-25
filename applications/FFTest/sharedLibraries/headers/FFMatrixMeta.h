#pragma once

#include <Object.h>

namespace pdb {

// the sub namespace
namespace ff {

class FFMatrixMeta : public pdb::Object {
 public:

  /**
   * The default constructor
   */
  FFMatrixMeta() = default;

  FFMatrixMeta(uint32_t row_id, uint32_t col_id) : colID(col_id), rowID(row_id) {}

  ENABLE_DEEP_COPY

  /**
   * The column position of the block
   */
  uint32_t colID = 0;

  /**
   * The row position of the block
   */
  uint32_t rowID = 0;

  bool operator==(const FFMatrixMeta &other) const {
    return colID == other.colID && rowID == other.rowID;
  }

  size_t hash() const {
    return 10000 * rowID + colID;
  }
};

}

}