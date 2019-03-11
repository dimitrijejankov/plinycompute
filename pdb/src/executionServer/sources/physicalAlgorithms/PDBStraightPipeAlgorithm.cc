//
// Created by dimitrije on 2/25/19.
//

#include <PDBVector.h>
#include <ComputePlan.h>
#include "physicalAlgorithms/PDBStraightPipeAlgorithm.h"
#include "ExJob.h"

pdb::PDBStraightPipeAlgorithm::PDBStraightPipeAlgorithm(const pdb::Handle<PDBSourcePageSetSpec> &source,
                                                        const pdb::Handle<PDBSinkPageSetSpec> &sink,
                                                        const pdb::Handle<pdb::Vector<PDBSourcePageSetSpec>> &secondarySources)
                                                        : PDBPhysicalAlgorithm(source, sink, secondarySources) {}


bool pdb::PDBStraightPipeAlgorithm::setup(std::shared_ptr<pdb::PDBStorageManagerBackend> &storage, Handle<pdb::ExJob> &job, const std::string &error) {

  // init the plan
  ComputePlan plan(job->tcap, *job->computations);
  LogicalPlanPtr logicalPlan = plan.getPlan();

  /// 1. Figure out the source page set

  // get the source computation
  auto srcNode = logicalPlan->getComputations().getProducingAtomicComputation(source->tupleSetIdentifier);

  // if this is a scan set get the page set from a real set
  PDBAbstractPageSetPtr sourcePageSet;
  if(srcNode->getAtomicComputationTypeID() == ScanSetAtomicTypeID) {

    // cast it to a scan
    auto scanNode = std::dynamic_pointer_cast<ScanSet>(srcNode);

    std::cout << source->pageSetIdentifier.first << " : " << source->pageSetIdentifier.second;

    // get the page set
    sourcePageSet = storage->createPageSetFromPDBSet(scanNode->getDBName(),
                                                     scanNode->getSetName(),
                                                     std::make_pair(source->pageSetIdentifier.first, source->pageSetIdentifier.second));
  }
  else {

    // we are reading from an existing page set get it
    sourcePageSet = storage->getPageSet(std::make_pair(job->computationID, source->tupleSetIdentifier));
  }

  // did we manage to get a source page set? if not the setup failed
  if(sourcePageSet == nullptr) {
    return false;
  }

  /// 2. Figure out the sink tuple set

  // figure out the sink node
  auto sinkNode = logicalPlan->getComputations().getProducingAtomicComputation(sink->tupleSetIdentifier);

  // ok so are we writing to an output set if so store the name of the output set
  if(sinkNode->getAtomicComputationTypeID()  == WriteSetTypeID) {

    // cast the node to the output
    auto writerNode = std::dynamic_pointer_cast<WriteSet>(sinkNode);

    // set the output set
    outputSet = std::make_shared<std::pair<std::string, std::string>>(writerNode->getDBName(), writerNode->getSetName());

    // we should materialize this
    shouldMaterialize = true;
  }

  // get the sink page set
  auto sinkPageSet = storage->createAnonymousPageSet(std::make_pair(sink->pageSetIdentifier.first, sink->pageSetIdentifier.second));

  // did we manage to get a sink page set? if not the setup failed
  if(sinkPageSet == nullptr) {
    return false;
  }

  /// 3. Init the pipeline

  // empty computations parameters
  std::map<std::string, ComputeInfoPtr> params;

  // init the pipeline
  myPipeline = plan.buildPipeline(source->tupleSetIdentifier, /* this is the TupleSet the pipeline starts with */
                                  sink->tupleSetIdentifier,     /* this is the TupleSet the pipeline ends with */
                                  sourcePageSet,
                                  sinkPageSet,
                                  params,
                                  20,
                                  0);

  return true;
}

bool pdb::PDBStraightPipeAlgorithm::run(std::shared_ptr<pdb::PDBStorageManagerBackend> &storage) {

  // run the pipeline
  myPipeline->run();


  // should we materialize this to a set?
  if(shouldMaterialize) {

    // get the page set
    auto sinkPageSet = storage->getPageSet(std::make_pair(sink->pageSetIdentifier.first, sink->pageSetIdentifier.second));

    std::cout << sink->pageSetIdentifier.first << " : " << sink->pageSetIdentifier.second;

    // if the thing does not exist finish!
    if(sinkPageSet == nullptr) {
      return false;
    }

    // copy the anonymous page set to the real set
    return storage->materializePageSet(sinkPageSet, *outputSet);
  }

  return true;
}

pdb::PDBPhysicalAlgorithmType pdb::PDBStraightPipeAlgorithm::getAlgorithmType() {
  return StraightPipe;
}
