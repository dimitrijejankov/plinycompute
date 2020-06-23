#pragma once

#include "SetWriter.h"
#include "BPStrip.h"

namespace pdb {

// the sub namespace
namespace bp {

/**
 * The matrix scanner
 */
class BPWriter : public SetWriter<pdb::bp::BPStrip> {
public:

  /**
   * The default constructor
   */
  BPWriter() = default;

  BPWriter(const String &dbName, const String &setName) : SetWriter(dbName, setName) {}

  ENABLE_DEEP_COPY
};

}

}