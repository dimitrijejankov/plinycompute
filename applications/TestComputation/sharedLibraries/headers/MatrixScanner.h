#pragma once

#include "TRABlock.h"
#include "SetScanner.h"

namespace pdb {

// the sub namespace
namespace matrix {

/**
 * The matrix scanner
 */
class MatrixScanner : public pdb::SetScanner<TRABlock> {
public:

  /**
   * The default constructor
   */
  MatrixScanner() = default;

  MatrixScanner(const std::string &db, const std::string &set) : SetScanner(db, set) {}

  ENABLE_DEEP_COPY

};

}

}
