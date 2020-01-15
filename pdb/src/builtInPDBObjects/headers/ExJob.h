#pragma once

#include <cstdint>
#include <PDBString.h>
#include <PDBVector.h>
#include <Computation.h>
#include <PDBCatalogSet.h>
#include <ExJobNode.h>
#include <PDBPhysicalAlgorithm.h>

// PRELOAD %ExJob%

namespace pdb {

/**
 * Object we send to execute algorithm
 */
class ExJob : public Object  {
public:

  ~ExJob() = default;

  ENABLE_DEEP_COPY

  /**
   * The physical algorithm we want to run.
   */
  Handle<pdb::PDBPhysicalAlgorithm> physicalAlgorithm;

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
   * the index of the node in the nodes vector
   */
  int32_t thisNode;

  /**
   * Is it the lead node? Some algorithms require this to know if they are the leader of the group
   */
   bool isLeadNode = false;

  /**
   * Returns all the sets that are going to be materialized after the job is executed
   * @return - a vector of pairs the frist value is the database name, the second value is the set name
   */
  std::vector<std::pair<std::string, std::string>> getSetsToMaterialize();

  /**
   * Returns the actual sets we are scanning, it assumes that we are doing that. Check that with @see isScanningSet
   * @return get the scanning set
   */
  vector<pair<string, string>> getScanningSets();

  /**
   * True if, the source is an actual set and not an intermediate set
   * @return true if it is, false otherwise
   */
  bool isScanningSet();

  /**
   * Returns the type of the output container, that the materializing sets are going to have
   * @return the type
   */
  pdb::PDBCatalogSetContainerType getOutputSetContainer();

};

}