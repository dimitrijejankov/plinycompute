#pragma once

#include "PDBPhysicalAlgorithm.h"

namespace pdb {

class PDBBroadcastForJoinAlgorithm : public PDBPhysicalAlgorithm {
public:

  PDBBroadcastForJoinAlgorithm(const std::string &firstTupleSet,
                               const std::string &finalTupleSet,
                               const pdb::Handle<PDBSourcePageSetSpec> &source,
                               const pdb::Handle<PDBSinkPageSetSpec> &intermediate,
                               const pdb::Handle<PDBSinkPageSetSpec> &sink,
                               const pdb::Handle<pdb::Vector<pdb::Handle<PDBSourcePageSetSpec>>> &secondarySources);

  /**
   * Returns DistributedAggregation as the type
   * @return the type
   */
  PDBPhysicalAlgorithmType getAlgorithmType() override;

private:

  /**
   * The intermediate page set
   */
  pdb::Handle<PDBSinkPageSetSpec> intermediate;

};

}