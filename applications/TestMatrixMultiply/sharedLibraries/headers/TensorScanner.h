#pragma once

#include "TRABlock.h"
#include "SetScanner.h"

namespace pdb {

// the sub namespace
namespace matrix {

/**
 * The matrix scanner
 */
class TensorScanner : public pdb::SetScanner<pdb::TRABlock> {
public:

  /**
   * The default constructor
   */
  TensorScanner() = default;

  TensorScanner(const std::string &db, const std::string &set) : SetScanner(db, set) {}

  ENABLE_DEEP_COPY

};

}

}
