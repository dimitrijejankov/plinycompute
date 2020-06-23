#pragma once

#include <Object.h>
#include <Handle.h>
#include "BPStripMeta.h"
#include "BPStripData.h"

namespace pdb {

// the sub namespace
namespace bp {

/**
 * This represents a block in a large matrix distributed matrix.
 * For example if the large matrix has the size of 10000x10000 and is split into 4 blocks of size 2500x2500
 * Then we would have the following blocks in the system
 *
 * |metaData.colID|metaData.rowID|data.numRows|data.numCols| data.block |
 * |       0      |       1      |    25k     |    25k     | 25k * 25k  |
 * |       1      |       1      |    25k     |    25k     | 25k * 25k  |
 * |       0      |       0      |    25k     |    25k     | 25k * 25k  |
 * |       1      |       0      |    25k     |    25k     | 25k * 25k  |
 */
class BPStrip : public pdb::Object {
 public:

  /**
   * The default constructor
   */
  BPStrip() = default;

  /**
   * The constructor for a block size
   * @param batchID - the value we want to initialize the batch id to
   * @param numRows - the number of rows the block has
   * @param numCols - the number of columns the block has
   */
  BPStrip(uint32_t batchID, uint32_t numRows, uint32_t numCols) {
    metaData = makeObject<BPStripMeta>(batchID),
    data = makeObject<BPStripData>(numRows, numCols);
  }

  ENABLE_DEEP_COPY

  /**
   * The metadata of the matrix
   */
  Handle<BPStripMeta> metaData;

  /**
   * The data of the matrix
   */
  Handle<BPStripData> data;

  /**
   *
   * @return
   */
  Handle<BPStripMeta>& getKey() {
    return metaData;
  }

  /**
   *
   * @return
   */
  BPStripMeta& getKeyRef(){
    return *metaData;
  }

  /**
   *
   * @return
   */
  Handle<BPStripData>& getValue() {
    return data;
  }

  BPStripData& getValueRef() {
    return *data;
  }

  uint32_t getRowID() {
    return metaData->batchID;
  }

  uint32_t getNumRows() {
    return data->numRows;
  }

  uint32_t getNumCols() {
    return data->numCols;
  }
};


}

}