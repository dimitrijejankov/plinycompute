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

  ExJobNode(int32_t port, const std::string &address) : port(port), address(address) {}

  ENABLE_DEEP_COPY

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