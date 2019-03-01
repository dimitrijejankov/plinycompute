//
// Created by dimitrije on 2/21/19.
//

#ifndef PDB_PDBABSTRACTPIPELINE_H
#define PDB_PDBABSTRACTPIPELINE_H

#include <list>

#include <AtomicComputation.h>
#include <AtomicComputationClasses.h>
#include <PDBPhysicalAlgorithm.h>
#include <Handle.h>

enum PDBPipelineType {

  PDB_STRAIGHT_PIPELINE,
  PDB_AGGREGATION_PIPELINE,
  PDB_JOIN_SIDE_PIPELINE
};

namespace pdb {

class PDBAbstractPhysicalNode;
using PDBAbstractPhysicalNodePtr = std::shared_ptr<PDBAbstractPhysicalNode>;
using PDBAbstractPhysicalNodeWeakPtr = std::weak_ptr<PDBAbstractPhysicalNode>;

class PDBAbstractPhysicalNode {

public:

  // TODO
  PDBAbstractPhysicalNode(const std::vector<AtomicComputationPtr>& pipeline, size_t id) : pipeline(pipeline), id(id) {};

  virtual ~PDBAbstractPhysicalNode() = default;

  /**
   * Returns a shared pointer handle to this node
   * @return the shared pointer handle
   */
  PDBAbstractPhysicalNodePtr getHandle() {

    std::shared_ptr<PDBAbstractPhysicalNode> tmp;

    // if we do not have a handle to this node already
    if(handle.expired()) {

      // make a handle and set the weak ptr handle
      tmp = std::shared_ptr<PDBAbstractPhysicalNode> (this);
      handle = tmp;
    }

    // return it
    return handle.lock();
  }

  /**
   * Removes a consumer of this node
   * @param consumer the consumer we want to remove
   */
  void removeConsumer(const PDBAbstractPhysicalNodePtr &consumer) {

    // detach them
    consumers.remove(consumer);
    consumer->producers.remove_if([&](PDBAbstractPhysicalNodeWeakPtr p){ return this->getHandle() == p.lock(); });
  }

  /**
  * Adds a consumer to the node
  * @param consumer the consumer
  */
  void addConsumer(const pdb::PDBAbstractPhysicalNodePtr &consumer) {
    consumers.push_back(consumer);
    consumer->producers.push_back(getWeakHandle());
  }

  const std::list<PDBAbstractPhysicalNodePtr> &getConsumers();

  const std::list<PDBAbstractPhysicalNodePtr> &getProducers();

  /**
   * Returns the cost of running this pipeline
   * @return the cost
   */
  virtual size_t getCost() {
    throw std::runtime_error("");
  }

  /**
   * Returns the type of the pipeline
   * @return the type
   */
  virtual PDBPipelineType getType() = 0;

  /**
   * Returns the identififer
   * @return
   */
  std::string getNodeIdentifier() { return std::string("node_") + std::to_string(id); };

  /**
   * Returns all the atomic computations the make up this pipeline
   * @return a vector with the atomic computations
   */
  const std::vector<AtomicComputationPtr>& getPipeComputations() { return pipeline; }

  /**
   * Checks whether this starts with a scan set in this case this means that it has a source set
   * @return true if it has one false otherwise.
   */
  bool hasScanSet() {

    // just to make sure the pipeline is not empty
    if(pipeline.empty()) {
      return false;
    }

    // check if the first computation is an atomic computation
    return pipeline.front()->getAtomicComputationTypeID() == ScanSetAtomicTypeID;
  };

  /**
   * Returns the source set, it assumes that it has one
   * @return (dbName, setName)
   */
  std::pair<std::string, std::string> getSourceSet() {

    // grab the scan set from the pipeline
    auto scanSet = std::dynamic_pointer_cast<ScanSet>(pipeline.front());
    return std::make_pair(scanSet->getDBName(), scanSet->getSetName());
  }

  /**
   * Returns the algorithm we chose to run this pipeline
   * @return the algorithm
   */
  virtual pdb::Handle<pdb::PDBPhysicalAlgorithm> generateAlgorithm() = 0;

private:

  /**
   * Returns a weak pointer handle to this node
   * @return the weak pointer handle
   */
  PDBAbstractPhysicalNodeWeakPtr getWeakHandle() {
    return handle;
  }

  /**
   * The identifier of this node
   */
  size_t id;

  /**
   * Where the pipeline begins
   */
  std::vector<AtomicComputationPtr> pipeline;

  /**
   * A list of consumers of this node
   */
  std::list<PDBAbstractPhysicalNodePtr> consumers;

  /**
   * A list of producers of this node
   */
  std::list<PDBAbstractPhysicalNodeWeakPtr> producers;

  /**
   * A shared pointer to an instance of this node
   */
  PDBAbstractPhysicalNodeWeakPtr handle;
};

}
#endif //PDB_PDBABSTRACTPIPELINE_H
