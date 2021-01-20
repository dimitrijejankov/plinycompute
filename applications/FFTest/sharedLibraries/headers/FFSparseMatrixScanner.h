#pragma once

#include "FFMatrixBlock.h"
#include "FFSparseBlock.h"
#include "SetScanner.h"

namespace pdb {

// the sub namespace
namespace ff {

/**
 * The FFMatrix scanner
 */
class FFSparseMatrixScanner : public pdb::SetScanner<pdb::ff::FFSparseBlock> {
 public:

  /**
   * The default constructor
   */
  FFSparseMatrixScanner() = default;

  FFSparseMatrixScanner(const std::string &db, const std::string &set) : SetScanner(db, set) {}

  ENABLE_DEEP_COPY

};

}

}
