//
// Created by dimitrije on 2/25/19.
//

#ifndef PDB_PDBJOB_H
#define PDB_PDBJOB_H

#include <cstdint>
#include <PDBString.h>
#include <PDBVector.h>
#include <Computation.h>
#include <PDBCatalogSet.h>
#include <PDBPhysicalAlgorithm.h>
#include <ExJobNode.h>

// PRELOAD %ExJob%

namespace pdb {

class ExJob : public Object  {
public:

  ~ExJob() = default;

  ENABLE_DEEP_COPY

  /**
   * The physical algorithm we want to run.
   */
  Handle<PDBPhysicalAlgorithm> physicalAlgorithm;

  /**
   * The computations we want to send
   */
  Handle<Vector<Handle<Computation>>> computations;

  /**
   * The tcap string of the computation
   */
  pdb::String tcap;

  /**
   * The id of the job
   */
  uint64_t jobID;

  /**
   * The id of the computation
   */
  uint64_t computationID;

  /**
   * The size of the computation
   */
  uint64_t computationSize;

  /**
   * The number of that are going to do the processing
   */
  uint64_t numberOfProcessingThreads;

  /**
   * The number of nodes
   */
  uint64_t numberOfNodes;

  /**
   * Nodes that are used for this job, just a bunch of IP
   */
  pdb::Vector<pdb::Handle<ExJobNode>> nodes;

  /**
   * the IP and port of the
   */
  pdb::Handle<ExJobNode> thisNode;

  /**
   * Returns all the sets that are going to be materialized after the job is executed
   * @return - a vector of pairs the frist value is the database name, the second value is the set name
   */
  std::vector<std::pair<std::string, std::string>> getSetsToMaterialize() {

    // get the sets to materialize
    const auto& sets = physicalAlgorithm->getSetsToMaterialize();

    // allocate the output container
    std::vector<std::pair<std::string, std::string>> out;
    out.reserve(sets->size());

    // copy the sets
    for(int i = 0; i < sets->size(); ++i) {
      out.emplace_back(make_pair((*sets)[i].database, (*sets)[i].set));
    }

    // return it
    return std::move(out);
  }

  /**
   * Returns the actual sets we are scanning, it assumes that we are doing that. Check that with @see isScanningSet
   * @return get the scanning set
   */
  vector<pair<string, string>> getScanningSets() {

    // return the scanning set
    return std::move(physicalAlgorithm->getSetsToScan());
  }

  /**
   * True if, the source is an actual set and not an intermediate set
   * @return true if it is, false otherwise
   */
  bool isScanningSet() {
    return !physicalAlgorithm->getSetsToScan().empty();
  }

  /**
   * Returns the type of the output container, that the materializing sets are going to have
   * @return the type
   */
  pdb::PDBCatalogSetContainerType getOutputSetContainer() {
    return physicalAlgorithm->getOutputContainerType();
  }

};

}


#endif //PDB_PDBJOB_H
