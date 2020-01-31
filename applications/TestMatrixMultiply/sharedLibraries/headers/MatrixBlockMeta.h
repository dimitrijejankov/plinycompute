#pragma once

#include <Object.h>

namespace pdb {

// the sub namespace
namespace matrix {

class MatrixBlockMeta : public pdb::Object {
 public:

  /**
   * The default constructor
   */
  MatrixBlockMeta() = default;

  MatrixBlockMeta(uint32_t row_id, uint32_t col_id) : rowID(row_id), colID(col_id) {}

  ENABLE_DEEP_COPY

  /**
   * The row position of the block
   */
  uint32_t rowID = 0;

  /**
   * The column position of the block
   */
  uint32_t colID = 0;

  bool operator==(const MatrixBlockMeta &other) const {
    return colID == other.colID && rowID == other.rowID;
  }

  size_t hash() const {
    return 10000 * rowID + colID;
  }
};

}
}
