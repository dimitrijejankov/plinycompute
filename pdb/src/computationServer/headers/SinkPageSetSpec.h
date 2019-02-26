//
// Created by vicram on 2/26/19.
//

#ifndef PDB_SINKPAGESETSPEC_H
#define PDB_SINKPAGESETSPEC_H

#include "PageSetSpec.h"
#include "PDBAbstractAlgorithm.h"

namespace pdb {
/**
 * This is a PageSetSpec which corresponds to a PDBSink.
 */
class SinkPageSetSpec : public PageSetSpec {
 private:
  PDBSinkType sinkType; // Should this be a Handle?
 public:
  ENABLE_DEEP_COPY
};
}

#endif //PDB_SINKPAGESETSPEC_H
