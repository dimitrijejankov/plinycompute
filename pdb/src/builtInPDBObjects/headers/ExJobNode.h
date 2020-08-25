#pragma once

#include <cstdint>
#include <PDBString.h>
#include <PDBVector.h>
#include <Computation.h>

// PRELOAD %ExJobNode%

namespace pdb {

class ExJobNode : public Object  {
public:

  ExJobNode() = default;

  ExJobNode(int32_t nodeID, int32_t port, const std::string &address) : nodeID(nodeID), port(port), address(address) {}

  ENABLE_DEEP_COPY

  /**
   * The id of the node
   */
  int32_t nodeID{};

  /**
   * The port
   */
  int32_t port{};

  /**
   * The address
   */
  pdb::String address;
};

}