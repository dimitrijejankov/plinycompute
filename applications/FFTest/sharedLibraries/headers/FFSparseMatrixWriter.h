#pragma once

#include "SetWriter.h"
#include "FFSparseBlock.h"

namespace pdb {

// the sub namespace
namespace ff {

/**
 * The matrix scanner
 */
class FFSparseMatrixWriter : public SetWriter<pdb::ff::FFSparseBlock> {
public:

  /**
   * The default constructor
   */
  FFSparseMatrixWriter() = default;

  FFSparseMatrixWriter(const String &dbName, const String &setName) : SetWriter(dbName, setName) {}

  ENABLE_DEEP_COPY
};

}

}