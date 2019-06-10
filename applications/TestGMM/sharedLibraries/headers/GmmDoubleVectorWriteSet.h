#pragma once

#include <DoubleVector.h>
#include "SetWriter.h"

using namespace pdb;
class GmmDoubleVectorWriteSet : public SetWriter<DoubleVector> {

 public:
  ENABLE_DEEP_COPY

  GmmDoubleVectorWriteSet() = default;

  GmmDoubleVectorWriteSet(std::string dbName, std::string setName) {
    this->setOutputSet(std::move(dbName), std::move(setName));
  }
};
