#pragma once

#include "MatrixBlock3D.h"
#include "SetScanner.h"

namespace pdb {

// the sub namespace
namespace matrix_3d {

/**
 * The matrix scanner
 */
class Matrix3DScanner : public pdb::SetScanner<pdb::matrix_3d::MatrixBlock3D> {
public:

  /**
   * The default constructor
   */
  Matrix3DScanner() = default;

  Matrix3DScanner(const std::string &db, const std::string &set) : SetScanner(db, set) {}

  ENABLE_DEEP_COPY

};

}

}
