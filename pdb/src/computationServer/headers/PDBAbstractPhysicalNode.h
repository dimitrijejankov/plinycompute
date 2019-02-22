//
// Created by dimitrije on 2/21/19.
//

#ifndef PDB_PDBABSTRACTPIPELINE_H
#define PDB_PDBABSTRACTPIPELINE_H

#include <AtomicComputation.h>
#include <list>


enum PDBPipelineType {

  PDB_STRAIGHT_PIPELINE,
  PDB_AGGREGATION_PIPELINE

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
  virtual void removeConsumer(const PDBAbstractPhysicalNodePtr &consumer) {

    // detach them
    consumers.remove(consumer);
    consumer->producers.remove_if([&](PDBAbstractPhysicalNodeWeakPtr p){ return this->getHandle() == p.lock(); });
  }

  /**
  * Adds a consumer to the node
  * @param consumer the consumer
  */
  virtual void addConsumer(const pdb::PDBAbstractPhysicalNodePtr &consumer) {
    consumers.push_back(consumer);
    consumer->producers.push_back(getWeakHandle());
  }

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
   * // TODO remove this somehow
   * @return
   */
  bool hasScanSet() {

    // just to make sure the pipeline is not empty
    if(pipeline.empty()) {
      return false;
    }

    // check if the first computation is an atomic computation
    return pipeline.front()->getAtomicComputationTypeID() == ScanSetAtomicTypeID;
  };

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
