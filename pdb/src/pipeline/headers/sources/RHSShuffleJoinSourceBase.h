#pragma once

#include "TupleSet.h"
#include <memory>

class RHSShuffleJoinSourceBase;
using RHSShuffleJoinSourceBasePtr = std::shared_ptr<RHSShuffleJoinSourceBase>;

class RHSShuffleJoinSourceBase {
public:
  virtual ~RHSShuffleJoinSourceBase() = default;

  virtual std::pair<pdb::TupleSetPtr, std::vector<std::pair<size_t, size_t>>*> getNextTupleSet() = 0;

};