#pragma once

#include "PDBPhysicalAlgorithm.h"

namespace pdb {

class PDBUnionAlgorithm : public PDBPhysicalAlgorithm {
 public:

  ENABLE_DEEP_COPY


  pdb::Handle<PDBSourcePageSetSpec> leftUnionSource;
  pdb::Handle<PDBSourcePageSetSpec> rightUnionSource;
  pdb::Handle<PDBSourcePageSetSpec> leftIntermediateSource;
  pdb::Handle<PDBSourcePageSetSpec> rightIntermediateSource;

  pdb::Handle<PDBSinkPageSetSpec> sink;


};

}