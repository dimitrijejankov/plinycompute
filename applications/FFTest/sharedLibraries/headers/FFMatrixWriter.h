#pragma once

#include "SetWriter.h"
#include "FFMatrixBlock.h"

namespace pdb {

// the sub namespace
namespace ff {

/**
 * The matrix scanner
 */
class FFMatrixWriter : public SetWriter<pdb::ff::FFMatrixBlock> {
public:

  /**
   * The default constructor
   */
  FFMatrixWriter() = default;

  FFMatrixWriter(const String &dbName, const String &setName) : SetWriter(dbName, setName) {}

  ENABLE_DEEP_COPY
};

}

}