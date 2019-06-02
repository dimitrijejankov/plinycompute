//
// Created by dimitrije on 2/20/19.
//

#ifndef PDB_PDBCOMPUTATIONCLIENT_H
#define PDB_PDBCOMPUTATIONCLIENT_H

#include <ServerFunctionality.h>
#include <Computation.h>

namespace pdb {


class PDBComputationClient : public ServerFunctionality {

public:

  PDBComputationClient(const string &address, int port, const PDBLoggerPtr &myLogger);

  /**
   *
   * @param computations
   * @param tcap
   * @param error
   * @return
   */
  bool executeComputations(Handle<Vector<Handle<Computation>>> &computations, const pdb::String &tcap, std::string &error);

  /**
   *
   * @param forMe
   */
  void registerHandlers(PDBServer &forMe) override {};

 private:

  /* The IP address where this Catalog Client is connected to */
  std::string address;

  /* The port where this Catalog Client is connected to */
  int port = -1;

  /* Logger to debug information */
  PDBLoggerPtr myLogger;

  // what is the maximum computation size we support (bytes)
  static const uint64_t MAX_COMPUTATION_SIZE;

  // what is the increment we are going to increase the computation size if needed (bytes)
  static const uint64_t MAX_STEP_SIZE;
};

}
#endif //PDB_PDBCOMPUTATIONCLIENT_H
