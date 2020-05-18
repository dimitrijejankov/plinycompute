#pragma once

#include "Computation.h"

namespace pdb {

  class UnionCompBase : public Computation {
  public:

    // returns the key extractor for the materialized result of this
    PDBKeyExtractorPtr getKeyExtractor() override {
      throw runtime_error("We don't extract keys from an union...");
    }
  };

}