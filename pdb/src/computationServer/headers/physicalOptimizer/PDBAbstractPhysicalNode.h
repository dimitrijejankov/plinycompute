#include <utility>
#include <cassert>

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

using PDBPlanningResult = std::pair<Handle<PDBPhysicalAlgorithm>, std::list<pdb::PDBAbstractPhysicalNodePtr>>;

class PDBAbstractPhysicalNode {

public:

  // TODO
  PDBAbstractPhysicalNode(std::vector<AtomicComputationPtr> pipeline, size_t computationID, size_t id) : pipeline(std::move(pipeline)), id(id), computationID(computationID) {};

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

  /**
   * Returns the consumers of this node
   * @return - the list of consumers
   */
  const std::list<PDBAbstractPhysicalNodePtr> &getConsumers();

  /**
   * Returns the list of producers of this node
   * @return - the list of producers
   */
  const std::list<PDBAbstractPhysicalNodePtr> getProducers();

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
   * Check if we are doing a join, at the beginning of this pipeline
   * @return true if we are doing the join, false otherwise
   */
  bool isJoining() {

    // just to make sure the pipeline is not empty
    if(pipeline.empty()) {
      return false;
    }

    // check if it is a join
    return getPipeComputations().front()->getAtomicComputationTypeID() == ApplyJoinTypeID;
  }

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
   * Returns the source page set
   * @return returns a new instance of PDBSourcePageSetSpec, that describes the page set that is the source for this node
   */
  virtual pdb::Handle<PDBSourcePageSetSpec> getSourcePageSet() {

    pdb::Handle<PDBSourcePageSetSpec> source = pdb::makeObject<PDBSourcePageSetSpec>();

    // do we have a scan set here
    if(hasScanSet()) {

      source->sourceType = PDBSourceType::SetScanSource;
      source->pageSetIdentifier = std::make_pair(computationID, (String) pipeline.front()->getOutputName());

      return source;
    }

    // ok so this does not have and scan sets so this means it's source page set is produced by some other pipeline
    // since there can only be one producer, we just have to check him
    assert(producers.size() == 1);

    // grab the producer
    auto producer = producers.front().lock();

    // grab the sink from the producer
    auto sink = producer->getSinkPageSet();

    // this should never be null
    assert(sink != nullptr);

    // fill up the stuff
    source->sourceType = getSourceTypeForSinkType(sink->sinkType);
    source->pageSetIdentifier = sink->pageSetIdentifier;

    // return the source
    return source;
  }

  /**
   * Returns the sink page set for this node if any
   * @return the sinks set, null ptr otherwise
   */
  virtual pdb::Handle<PDBSinkPageSetSpec> getSinkPageSet() {

    // did actually produce the page set if we haven't return null
    if(!sinkPageSet.produced){
      return nullptr;
    }

    // create the sink object
    pdb::Handle<PDBSinkPageSetSpec> sink = pdb::makeObject<PDBSinkPageSetSpec>();

    // fill up the stuff
    sink->sinkType = sinkPageSet.sinkType;
    sink->pageSetIdentifier = sinkPageSet.pageSetIdentifier;

    return sink;
  }

  /**
   * Sets the sink page set so we know what this node produces
   * @param pageSink - the sink page set
   */
  virtual void setSinkPageSet(pdb::Handle<PDBSinkPageSetSpec> &pageSink) {

    // fill up the stuff
    sinkPageSet.sinkType = pageSink->sinkType;
    sinkPageSet.pageSetIdentifier = pageSink->pageSetIdentifier;
    sinkPageSet.produced = true;
  }

  /**
   * Returns the source type for a particular sink type. Basically it tells us which source we need
   * in order to use the result of a particular sink
   * @param sinkType - the type of the sink
   * @return - the type of the source
   */
  PDBSourceType getSourceTypeForSinkType(PDBSinkType sinkType) {

    switch (sinkType) {

      case SetSink: return SetScanSource;
      case AggregationSink: return AggregationSource;
      case AggShuffleSink: return ShuffledAggregatesSource;
      case JoinShuffleSink: return ShuffledJoinTuplesSource;
      case BroadcastJoinSink: return BroadcastJoinSource;
      default:break;
    }

    // this is not supposed to happen
    assert(false);
  }

  /**
   * Returns the algorithm we chose to run this pipeline
   * @return the planning result, a pair of the algorithm and the consumers of the result
   */
  virtual PDBPlanningResult generateAlgorithm() = 0;

  /**
   * Returns the algorithm we chose to run this pipeline, but assumes that we are pipelining stuff into it...
   * @return the planning result, a pair of the algorithm and the consumers of the result
   */
  virtual PDBPlanningResult generatePipelinedAlgorithm(const std::string &startTupleSet,
                                                       const pdb::Handle<PDBSourcePageSetSpec> &source,
                                                       pdb::Handle<pdb::Vector<PDBSourcePageSetSpec>> &additionalSources) = 0;

protected:

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

  /**
   * The computation this node belongs to
   */
  size_t computationID;

  /**
   * This contains the info about the page set produced by the algorithm
   */
  struct {

    /**
     * Has this dataset been produced
     */
    bool produced = false;

    /**
     * The type of the sink
     */
    PDBSinkType sinkType = None;

    /**
     * Each page set is identified by a integer and a string. Generally set to (computationID, tupleSetIdentifier)
     * but relying on that is considered bad practice
     */
    std::pair<size_t, std::string> pageSetIdentifier;

  } sinkPageSet;
};

}
#endif //PDB_PDBABSTRACTPIPELINE_H
