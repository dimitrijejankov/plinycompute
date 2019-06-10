#pragma once

#include "SetWriter.h"
#include "IntDoubleVectorPair.h"
#include "GmmAggregateOutputLazy.h"

using namespace pdb;
class GmmAggregateLazyWriteSet : public SetWriter<GmmAggregateOutputLazy> {

public:
  ENABLE_DEEP_COPY

  GmmAggregateLazyWriteSet() = default;

  GmmAggregateLazyWriteSet(std::string dbName, std::string setName) {
    this->setOutputSet(std::move(dbName), std::move(setName));
  }
};
