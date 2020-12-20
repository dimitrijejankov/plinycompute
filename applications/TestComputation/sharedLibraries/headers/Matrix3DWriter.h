#pragma once

#include "SetWriter.h"
#include "MatrixConvResult.h"

namespace pdb {

// the sub namespace
namespace matrix_3d {

/**
 * The matrix scanner
 */
class Matrix3DWriter : public SetWriter<pdb::matrix_3d::MatrixConvResult> {
public:

  /**
   * The default constructor
   */
  Matrix3DWriter() = default;

  Matrix3DWriter(const String &dbName, const String &setName) : SetWriter(dbName, setName) {}

  ENABLE_DEEP_COPY
};

}

}