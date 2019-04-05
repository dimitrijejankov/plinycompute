//
// Created by dimitrije on 2/25/19.
//

#include <PDBVector.h>
#include <ComputePlan.h>
#include <GenericWork.h>
#include "physicalAlgorithms/PDBStraightPipeAlgorithm.h"
#include "ExJob.h"

pdb::PDBStraightPipeAlgorithm::PDBStraightPipeAlgorithm(const std::string &firstTupleSet,
                                                        const std::string &finalTupleSet,
                                                        const pdb::Handle<PDBSourcePageSetSpec> &source,
                                                        const pdb::Handle<PDBSinkPageSetSpec> &sink,
                                                        const pdb::Handle<pdb::Vector<PDBSourcePageSetSpec>> &secondarySources)
                                                        : PDBPhysicalAlgorithm(firstTupleSet, finalTupleSet, source, sink, secondarySources) {}


bool pdb::PDBStraightPipeAlgorithm::setup(std::shared_ptr<pdb::PDBStorageManagerBackend> &storage, Handle<pdb::ExJob> &job, const std::string &error) {

  // init the plan
  ComputePlan plan(job->tcap, *job->computations);
  LogicalPlanPtr logicalPlan = plan.getPlan();

  /// 1. Figure out the source page set

  // get the source computation
  auto srcNode = logicalPlan->getComputations().getProducingAtomicComputation(firstTupleSet);

  // if this is a scan set get the page set from a real set
  PDBAbstractPageSetPtr sourcePageSet;
  if(srcNode->getAtomicComputationTypeID() == ScanSetAtomicTypeID) {

    // cast it to a scan
    auto scanNode = std::dynamic_pointer_cast<ScanSet>(srcNode);

    // get the page set
    sourcePageSet = storage->createPageSetFromPDBSet(scanNode->getDBName(),
                                                     scanNode->getSetName(),
                                                     std::make_pair(source->pageSetIdentifier.first, source->pageSetIdentifier.second));
  }
  else {

    // we are reading from an existing page set get it
    sourcePageSet = storage->getPageSet(std::make_pair(source->pageSetIdentifier.first, source->pageSetIdentifier.second));
  }

  // did we manage to get a source page set? if not the setup failed
  if(sourcePageSet == nullptr) {
    return false;
  }

  /// 2. Figure out the sink tuple set

  // figure out the sink node
  auto sinkNode = logicalPlan->getComputations().getProducingAtomicComputation(finalTupleSet);

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

  /// 3. Init the pipelines

  // get the number of worker threads from this server's config
  int32_t numWorkers = storage->getConfiguration()->numThreads;

  // empty computations parameters
  std::map<ComputeInfoType, ComputeInfoPtr> params;

  // build a pipeline for each worker thread
  myPipelines = std::make_shared<std::vector<PipelinePtr>>();
  for (uint64_t i = 0; i < numWorkers; ++i) {
    auto pipeline = plan.buildPipeline(firstTupleSet, /* this is the TupleSet the pipeline starts with */
                                       finalTupleSet,     /* this is the TupleSet the pipeline ends with */
                                       sourcePageSet,
                                       sinkPageSet,
                                       params,
                                       job->numberOfNodes,
                                       job->numberOfProcessingThreads,
                                       20,
                                       i);
    myPipelines->push_back(pipeline);
  }

  return true;
}

bool pdb::PDBStraightPipeAlgorithm::run(std::shared_ptr<pdb::PDBStorageManagerBackend> &storage) {

  atomic_bool success;
  success = true;

  // create the buzzer
  atomic_int counter;
  counter = 0;
  PDBBuzzerPtr tempBuzzer = make_shared<PDBBuzzer>([&](PDBAlarm myAlarm, atomic_int &cnt) {

    // did we fail?
    if(myAlarm == PDBAlarm::GenericError) {
      success = false;
    }

    // increment the count
    cnt++;
  });

  // here we get a worker per pipeline and run all the preaggregationPipelines.

  for (int i = 0; i < myPipelines->size(); ++i) {
    // get a worker from the server
    PDBWorkerPtr worker = storage->getWorker();

    // make the work
    PDBWorkPtr myWork = std::make_shared<pdb::GenericWork>([&counter, i, this](PDBBuzzerPtr callerBuzzer) {
      (*myPipelines)[i]->run();
      // signal that the run was successful
      callerBuzzer->buzz(PDBAlarm::WorkAllDone, counter);
    });
    // run the work
    worker->execute(myWork, tempBuzzer);
  }

  // wait until all the preaggregationPipelines have completed
  while (counter < myPipelines->size()) {
    tempBuzzer->wait();
  }

  // should we materialize this to a set?
  if(shouldMaterialize) {

    // get the page set
    auto sinkPageSet = storage->getPageSet(std::make_pair(sink->pageSetIdentifier.first, sink->pageSetIdentifier.second));

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
