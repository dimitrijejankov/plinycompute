//
// Created by dimitrije on 2/25/19.
//

#ifndef PDB_PDBJOBNODE_H
#define PDB_PDBJOBNODE_H

#include <cstdint>
#include <PDBString.h>
#include <PDBVector.h>
#include <Computation.h>
#include "PDBPhysicalAlgorithm.h"

// PRELOAD %ExJobNode%

namespace pdb {

class ExJobNode : public Object  {
public:

  ExJobNode() = default;

  ExJobNode(uint64_t port, const std::string &address) : port(port), address(address) {}

  ENABLE_DEEP_COPY

  /**
   * The port
   */
  uint64_t port;

  /**
   * The address
   */
  pdb::String address;
};

}


#endif //PDB_PDBJOB_H
