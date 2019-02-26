//
// Created by vicram on 2/26/19.
//

#ifndef PDB_SOURCEPAGESETSPEC_H
#define PDB_SOURCEPAGESETSPEC_H

#include "PageSetSpec.h"
#include "PDBAbstractAlgorithm.h"

namespace pdb {
/**
 * This is a PageSetSpec which corresponds to a PDBSource.
 */
class SourcePageSetSpec : public PageSetSpec {
 private:
  PDBSourceType sourceType; // Should this be a Handle?
 public:
  ENABLE_DEEP_COPY
};
}


#endif //PDB_SOURCEPAGESETSPEC_H
