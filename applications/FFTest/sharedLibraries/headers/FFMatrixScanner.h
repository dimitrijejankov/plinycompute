#pragma once

#include "FFMatrixBlock.h"
#include "SetScanner.h"

namespace pdb {

// the sub namespace
namespace ff {

/**
 * The FFMatrix scanner
 */
class FFMatrixScanner : public pdb::SetScanner<pdb::ff::FFMatrixBlock> {
public:

  /**
   * The default constructor
   */
  FFMatrixScanner() = default;

  FFMatrixScanner(const std::string &db, const std::string &set) : SetScanner(db, set) {}

  ENABLE_DEEP_COPY

};

}

}
