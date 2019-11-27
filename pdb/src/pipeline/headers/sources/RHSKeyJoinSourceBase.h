#pragma once

#include "TupleSet.h"
#include <memory>

class RHSKeyJoinSourceBase;
using RHSKeyJoinSourceBasePtr = std::shared_ptr<RHSKeyJoinSourceBase>;

class RHSKeyJoinSourceBase {
 public:
  virtual ~RHSKeyJoinSourceBase() = default;

  virtual std::tuple<pdb::TupleSetPtr, std::vector<std::pair<size_t, size_t>>*, std::vector<std::pair<uint32_t, int32_t>>> getNextTupleSet() = 0;

};