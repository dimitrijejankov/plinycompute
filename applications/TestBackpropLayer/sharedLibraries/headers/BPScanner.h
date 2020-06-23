#pragma once

#include "BPStrip.h"
#include "SetScanner.h"

namespace pdb {

// the sub namespace
namespace bp {

/**
 * The matrix scanner
 */
class BPScanner : public pdb::SetScanner<pdb::bp::BPStrip> {
public:

  /**
   * The default constructor
   */
  BPScanner() = default;

  BPScanner(const std::string &db, const std::string &set) : SetScanner(db, set) {}

  ENABLE_DEEP_COPY

};

}

}
