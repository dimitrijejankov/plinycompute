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

  ExJobNode(int32_t port, int32_t backendPort, const std::string &address) : port(port),
                                                                             backendPort(backendPort),
                                                                             address(address) {}

  ENABLE_DEEP_COPY

  /**
   * The port
   */
  int32_t port{};

  /**
   * The backend of the port
   */
  int32_t backendPort{};

  /**
   * The address
   */
  pdb::String address;
};

}