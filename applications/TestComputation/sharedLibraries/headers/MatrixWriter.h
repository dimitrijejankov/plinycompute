#pragma once

#include "SetWriter.h"
#include "TRABlock.h"

namespace pdb {

// the sub namespace
namespace matrix {

/**
 * The matrix scanner
 */
class MatrixWriter : public SetWriter<TRABlock> {
public:

  /**
   * The default constructor
   */
  MatrixWriter() = default;

  MatrixWriter(const String &dbName, const String &setName) : SetWriter(dbName, setName) {}

  ENABLE_DEEP_COPY
};

}

}