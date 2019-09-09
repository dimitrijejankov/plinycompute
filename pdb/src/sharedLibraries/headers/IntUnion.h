#pragma once

#include "UnionComp.h"
#include "IntUnion.h"

namespace pdb {

class IntUnion : public UnionComp <IntUnion, int> {
public:

  IntUnion() = default;

  ENABLE_DEEP_COPY
};

}