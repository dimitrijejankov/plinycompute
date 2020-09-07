#pragma once

#include "SetWriter.h"
#include "TRABlock.h"

namespace pdb {

// the sub namespace
namespace matrix {

/**
 * The matrix scanner
 */
class TensorWriter : public SetWriter<pdb::TRABlock> {
public:

  /**
   * The default constructor
   */
  TensorWriter() = default;

  TensorWriter(const String &dbName, const String &setName) : SetWriter(dbName, setName) {}

  ENABLE_DEEP_COPY
};

}

}