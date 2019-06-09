#pragma once

#include <DoubleVector.h>
#include "SetScanner.h"
#include "IntDoubleVectorPair.h"

using namespace pdb;
class GmmDoubleVectorScanSet : public SetScanner<DoubleVector> {
public:
  ENABLE_DEEP_COPY

  GmmDoubleVectorScanSet() = default;
  GmmDoubleVectorScanSet(const std::string& dbName, const std::string& setName) : SetScanner(dbName, setName) {}

};
