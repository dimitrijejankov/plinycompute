//
// Created by dimitrije on 2/25/19.
//

#include <PDBVector.h>
#include <ComputePlan.h>
#include <GenericWork.h>
#include <PDBCatalogClient.h>
#include <physicalAlgorithms/PDBStraightPipeAlgorithm.h>

#include "physicalAlgorithms/PDBStraightPipeAlgorithm.h"
#include "ExJob.h"

pdb::PDBStraightPipeAlgorithm::PDBStraightPipeAlgorithm(const std::vector<PDBPrimarySource> &primarySource,
                                                        const AtomicComputationPtr &finalAtomicComputation,
                                                        const pdb::Handle<PDBSinkPageSetSpec> &sink,
                                                        const std::vector<pdb::Handle<PDBSourcePageSetSpec>> &secondarySources,
                                                        const pdb::Handle<pdb::Vector<PDBSetObject>> &setsToMaterialize)
                                                        : PDBPhysicalAlgorithm(primarySource, finalAtomicComputation, sink, secondarySources, setsToMaterialize) {}


bool pdb::PDBStraightPipeAlgorithm::setup(std::shared_ptr<pdb::PDBStorageManagerBackend> &storage, Handle<pdb::ExJob> &job, const std::string &error) {

  // TODO remove this
  const int pipelineIndex = 0;
  bool swapLHSandRHS = sources[pipelineIndex].swapLHSandRHS;
  pdb::String firstTupleSet = sources[pipelineIndex].firstTupleSet;

  // init the plan
  ComputePlan plan(job->tcap, *job->computations);
  logicalPlan = plan.getPlan();

  /// 1. Figure out the source page set

  // get the source computation
  auto srcNode = logicalPlan->getComputations().getProducingAtomicComputation(firstTupleSet);

  // go grab the source page set
  PDBAbstractPageSetPtr sourcePageSet = getSourcePageSet(storage, pipelineIndex);

  // did we manage to get a source page set? if not the setup failed
  if(sourcePageSet == nullptr) {
    return false;
  }

  /// 2. Figure out the sink tuple set

  // get the sink page set
  auto sinkPageSet = storage->createAnonymousPageSet(std::make_pair(sink->pageSetIdentifier.first, sink->pageSetIdentifier.second));

  // did we manage to get a sink page set? if not the setup failed
  if(sinkPageSet == nullptr) {
    return false;
  }

  /// 3. Init the pipelines

  // get the number of worker threads from this server's config
  int32_t numWorkers = storage->getConfiguration()->numThreads;

  // figure out the join arguments
  auto joinArguments = getJoinArguments(storage);

  // if we could not create them we are out of here
  if(joinArguments == nullptr) {
    return false;
  }

  // get catalog client
  auto catalogClient = storage->getFunctionalityPtr<PDBCatalogClient>();

  // empty computations parameters
  std::map<ComputeInfoType, ComputeInfoPtr> params =  {{ComputeInfoType::PAGE_PROCESSOR, std::make_shared<NullProcessor>()},
                                                       {ComputeInfoType::JOIN_ARGS, joinArguments},
                                                       {ComputeInfoType::SHUFFLE_JOIN_ARG, std::make_shared<ShuffleJoinArg>(swapLHSandRHS)},
                                                       {ComputeInfoType::SOURCE_SET_INFO, getSourceSetArg(catalogClient, pipelineIndex)}};


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

  // if we failed finish
  if(!success) {
    return success;
  }

  // should we materialize this to a set?
  for(int j = 0; j < setsToMaterialize->size(); ++j) {

    // get the page set
    auto sinkPageSet = storage->getPageSet(std::make_pair(sink->pageSetIdentifier.first, sink->pageSetIdentifier.second));

    // if the thing does not exist finish!
    if(sinkPageSet == nullptr) {
      success = false;
      break;
    }

    // materialize the page set
    sinkPageSet->resetPageSet();
    success = storage->materializePageSet(sinkPageSet, std::make_pair<std::string, std::string>((*setsToMaterialize)[j].database, (*setsToMaterialize)[j].set)) && success;
  }

  return success;
}

pdb::PDBPhysicalAlgorithmType pdb::PDBStraightPipeAlgorithm::getAlgorithmType() {
  return StraightPipe;
}

void pdb::PDBStraightPipeAlgorithm::cleanup() {

  // invalidate everything
  myPipelines = nullptr;
  logicalPlan = nullptr;
}

pdb::PDBCatalogSetContainerType pdb::PDBStraightPipeAlgorithm::getOutputContainerType() {
  return PDBCatalogSetContainerType::PDB_CATALOG_SET_VECTOR_CONTAINER;
}
