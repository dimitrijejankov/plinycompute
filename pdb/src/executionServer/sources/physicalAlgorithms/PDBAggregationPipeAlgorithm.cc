//
// Created by dimitrije on 3/20/19.
//

#include <SourceSetArg.h>
#include <PDBCatalogClient.h>
#include "ComputePlan.h"
#include "ExJob.h"
#include "PDBAggregationPipeAlgorithm.h"
#include "PDBStorageManagerBackend.h"
#include "GenericWork.h"

pdb::PDBAggregationPipeAlgorithm::PDBAggregationPipeAlgorithm(const std::vector<PDBPrimarySource> &primarySource,
                                                              const AtomicComputationPtr &finalAtomicComputation,
                                                              const Handle<PDBSinkPageSetSpec> &hashedToSend,
                                                              const Handle<PDBSourcePageSetSpec> &hashedToRecv,
                                                              const Handle<PDBSinkPageSetSpec> &sink,
                                                              const std::vector<pdb::Handle<PDBSourcePageSetSpec>> &secondarySources,
                                                              const pdb::Handle<pdb::Vector<PDBSetObject>> &setsToMaterialize)

    : PDBPhysicalAlgorithm(primarySource, finalAtomicComputation, sink, secondarySources, setsToMaterialize), hashedToSend(hashedToSend), hashedToRecv(hashedToRecv) {}

bool pdb::PDBAggregationPipeAlgorithm::setup(std::shared_ptr<pdb::PDBStorageManagerBackend> &storage, Handle<pdb::ExJob> &job, const std::string &error) {

  // init the plan
  ComputePlan plan(job->tcap, *job->computations);
  logicalPlan = plan.getPlan();

  // get the buffer manager
  auto myMgr = storage->getFunctionalityPtr<PDBBufferManagerInterface>();

  // init the logger
  logger = make_shared<PDBLogger>("aggregationPipeAlgorithm" + std::to_string(job->computationID));

  /// 1. Figure out the sink tuple set for the preaggregation (this will provide empty pages to the pipeline but we will
  /// discard them since they will be processed by the PreaggregationPageProcessor and they won't stay around).

  // get the sink page set
  auto intermediatePageSet = storage->createAnonymousPageSet(hashedToSend->pageSetIdentifier);

  // did we manage to get a sink page set? if not the setup failed
  if (intermediatePageSet == nullptr) {
    return false;
  }

  /// 2. Init the preaggregation queues

  pageQueues = std::make_shared<std::vector<PDBPageQueuePtr>>();
  for(int i = 0; i < job->numberOfNodes; ++i) { pageQueues->emplace_back(std::make_shared<PDBPageQueue>()); }


  /// 3. Initialize the sources

  // we put them here
  std::vector<PDBAbstractPageSetPtr> sourcePageSets;
  sourcePageSets.reserve(sources.size());

  // initialize them
  for(int i = 0; i < sources.size(); i++) {
    sourcePageSets.emplace_back(getSourcePageSet(storage, i));
  }

  /// 4. Initialize all the pipelines

  // get the number of worker threads from this server's config
  int32_t numWorkers = storage->getConfiguration()->numThreads;

  // check that we have at least one worker per primary source
  if(numWorkers < sources.size()) {
    return false;
  }

  // fill uo the vector for each thread
  preaggregationPipelines = std::make_shared<std::vector<PipelinePtr>>();
  for (uint64_t pipelineIndex = 0; pipelineIndex < job->numberOfProcessingThreads; ++pipelineIndex) {

    /// 4.1. Figure out the source page set

    // figure out what pipeline
    auto pipelineSource = pipelineIndex % sources.size();

    // grab these thins from the source we need them
    bool swapLHSandRHS = sources[pipelineSource].swapLHSandRHS;
    const pdb::String &firstTupleSet = sources[pipelineSource].firstTupleSet;

    // get the source computation
    auto srcNode = logicalPlan->getComputations().getProducingAtomicComputation(firstTupleSet);

    // go grab the source page set
    PDBAbstractPageSetPtr sourcePageSet = sourcePageSets[pipelineSource];

    // did we manage to get a source page set? if not the setup failed
    if (sourcePageSet == nullptr) {
      return false;
    }

    /// 4.2. Figure out the parameters of the pipeline

    // figure out the join arguments
    auto joinArguments = getJoinArguments (storage);

    // if we could not create them we are out of here
    if(joinArguments == nullptr) {
      return false;
    }

    // get catalog client
    auto catalogClient = storage->getFunctionalityPtr<PDBCatalogClient>();

    // initialize the parameters
    std::map<ComputeInfoType, ComputeInfoPtr> params = { { ComputeInfoType::PAGE_PROCESSOR,  std::make_shared<PreaggregationPageProcessor>(job->numberOfNodes,
                                                                                                                                           job->numberOfProcessingThreads,
                                                                                                                                           *pageQueues,
                                                                                                                                           myMgr) },
                                                         { ComputeInfoType::JOIN_ARGS, joinArguments },
                                                         { ComputeInfoType::SHUFFLE_JOIN_ARG, std::make_shared<ShuffleJoinArg>(swapLHSandRHS) },
                                                         { ComputeInfoType::SOURCE_SET_INFO, getSourceSetArg(catalogClient, pipelineSource)}} ;

    /// 4.3. Build the pipeline

    auto pipeline = plan.buildPipeline(firstTupleSet, /* this is the TupleSet the pipeline starts with */
                                       finalTupleSet,     /* this is the TupleSet the pipeline ends with */
                                       sourcePageSet,
                                       intermediatePageSet,
                                       params,
                                       job->numberOfNodes,
                                       job->numberOfProcessingThreads,
                                       20,
                                       pipelineIndex);

    preaggregationPipelines->push_back(pipeline);
  }

  /// 5. Create the sink

  // get the sink page set
  auto sinkPageSet = storage->createAnonymousPageSet(std::make_pair(sink->pageSetIdentifier.first, sink->pageSetIdentifier.second));

  // did we manage to get a sink page set? if not the setup failed
  if(sinkPageSet == nullptr) {
    return false;
  }

  /// 6. Create the page set that contains the preaggregated pages for this node

  // get the receive page set
  auto recvPageSet = storage->createFeedingAnonymousPageSet(std::make_pair(hashedToRecv->pageSetIdentifier.first, hashedToRecv->pageSetIdentifier.second),
                                                            job->numberOfProcessingThreads,
                                                            job->numberOfNodes);

  // did we manage to get a page set where we receive this? if not the setup failed
  if(recvPageSet == nullptr) {
    return false;
  }

  /// 7. Create the self receiver to forward pages that are created on this node and the network senders to forward pages for the other nodes

  senders = std::make_shared<std::vector<PDBPageNetworkSenderPtr>>();
  for(unsigned i = 0; i < job->nodes.size(); ++i) {

    // check if it is this node or another node
    if(job->nodes[i]->port == job->thisNode->port && job->nodes[i]->address == job->thisNode->address) {

      // make the self receiver
      selfReceiver = std::make_shared<pdb::PDBPageSelfReceiver>(pageQueues->at(i), recvPageSet, myMgr);
    }
    else {

      // make the sender
      auto sender = std::make_shared<PDBPageNetworkSender>(job->nodes[i]->address,
                                                           job->nodes[i]->port,
                                                           job->numberOfProcessingThreads,
                                                           job->numberOfNodes,
                                                           storage->getConfiguration()->maxRetries,
                                                           logger,
                                                           std::make_pair(hashedToRecv->pageSetIdentifier.first, hashedToRecv->pageSetIdentifier.second),
                                                           pageQueues->at(i));

      // setup the sender, if we fail return false
      if(!sender->setup()) {
        return false;
      }

      // make the sender
      senders->emplace_back(sender);
    }
  }

  /// 8. Create the aggregation pipeline

  aggregationPipelines = std::make_shared<std::vector<PipelinePtr>>();
  for (uint64_t workerID = 0; workerID < job->numberOfProcessingThreads; ++workerID) {

    // build the aggregation pipeline
    auto aggPipeline = plan.buildAggregationPipeline(finalTupleSet, recvPageSet, sinkPageSet, workerID);

    // store the aggregation pipeline
    aggregationPipelines->push_back(aggPipeline);
  }

  return true;
}

bool pdb::PDBAggregationPipeAlgorithm::run(std::shared_ptr<pdb::PDBStorageManagerBackend> &storage) {

  // success indicator
  atomic_bool success;
  success = true;

  /// 1. Run the aggregation pipeline, this runs after the preaggregation pipeline, but is started first.

  // create the buzzer
  atomic_int aggCounter;
  aggCounter = 0;
  PDBBuzzerPtr aggBuzzer = make_shared<PDBBuzzer>([&](PDBAlarm myAlarm, atomic_int &cnt) {

    // did we fail?
    if (myAlarm == PDBAlarm::GenericError) {
      success = false;
    }

    // increment the count
    cnt++;
  });

  // here we get a worker per pipeline and run all the preaggregation Pipelines.
  for (int workerID = 0; workerID < aggregationPipelines->size(); ++workerID) {

    // get a worker from the server
    PDBWorkerPtr worker = storage->getWorker();

    // make the work
    PDBWorkPtr myWork = std::make_shared<pdb::GenericWork>([&aggCounter, workerID, this](PDBBuzzerPtr callerBuzzer) {

      // run the pipeline
      (*aggregationPipelines)[workerID]->run();

      // signal that the run was successful
      callerBuzzer->buzz(PDBAlarm::WorkAllDone, aggCounter);
    });

    // run the work
    worker->execute(myWork, aggBuzzer);
  }

  /// 2. Run the self receiver so it can server pages to the aggregation pipeline

  // create the buzzer
  atomic_int selfRecDone;
  selfRecDone = 0;
  PDBBuzzerPtr selfRefBuzzer = make_shared<PDBBuzzer>([&](PDBAlarm myAlarm, atomic_int &cnt) {

    // did we fail?
    if (myAlarm == PDBAlarm::GenericError) {
      success = false;
    }

    // we are done here
    cnt = 1;
  });

  // run the work
  {

    // make the work
    PDBWorkPtr myWork = std::make_shared<pdb::GenericWork>([&selfRecDone, this](PDBBuzzerPtr callerBuzzer) {

      // run the receiver
      if(selfReceiver->run()) {

        // signal that the run was successful
        callerBuzzer->buzz(PDBAlarm::WorkAllDone, selfRecDone);
      }
      else {

        // signal that the run was unsuccessful
        callerBuzzer->buzz(PDBAlarm::GenericError, selfRecDone);
      }
    });

    // run the work
    storage->getWorker()->execute(myWork, selfRefBuzzer);
  }

  /// 3. Run the senders

  // create the buzzer
  atomic_int sendersDone;
  sendersDone = senders->size();
  PDBBuzzerPtr sendersBuzzer = make_shared<PDBBuzzer>([&](PDBAlarm myAlarm, atomic_int &cnt) {

    // did we fail?
    if (myAlarm == PDBAlarm::GenericError) {
      success = false;
    }

    // we are done here
    cnt++;
  });

  // go through each sender and run them
  for(auto &sender : *senders) {

    // make the work
    PDBWorkPtr myWork = std::make_shared<pdb::GenericWork>([&sendersDone, sender, this](PDBBuzzerPtr callerBuzzer) {

      // run the sender
      if(sender->run()) {

        // signal that the run was successful
        callerBuzzer->buzz(PDBAlarm::WorkAllDone, sendersDone);
      }
      else {

        // signal that the run was unsuccessful
        callerBuzzer->buzz(PDBAlarm::GenericError, sendersDone);
      }
    });

    // run the work
    storage->getWorker()->execute(myWork, sendersBuzzer);
  }

  /// 4. Run the preaggregation, this step comes before the aggregation step

  // create the buzzer
  atomic_int preaggCounter;
  preaggCounter = 0;
  PDBBuzzerPtr preaggBuzzer = make_shared<PDBBuzzer>([&](PDBAlarm myAlarm, atomic_int &cnt) {

    // did we fail?
    if (myAlarm == PDBAlarm::GenericError) {
      success = false;
    }

    // increment the count
    cnt++;
  });

  // here we get a worker per pipeline and run all the preaggregationPipelines.
  for (int workerID = 0; workerID < preaggregationPipelines->size(); ++workerID) {

    // get a worker from the server
    PDBWorkerPtr worker = storage->getWorker();

    // make the work
    PDBWorkPtr myWork = std::make_shared<pdb::GenericWork>([&preaggCounter, workerID, this](PDBBuzzerPtr callerBuzzer) {

      // run the pipeline
      (*preaggregationPipelines)[workerID]->run();

      // signal that the run was successful
      callerBuzzer->buzz(PDBAlarm::WorkAllDone, preaggCounter);
    });

    // run the work
    worker->execute(myWork, preaggBuzzer);
  }

  /// 5. Do the waiting

  // wait until all the preaggregationPipelines have completed
  while (preaggCounter < preaggregationPipelines->size()) {
    preaggBuzzer->wait();
  }

  // ok they have finished now push a null page to each of the preagg queues
  for(auto &queue : *pageQueues) { queue->enqueue(nullptr); }

  // wait while we are running the receiver
  while(selfRecDone == 0) {
    selfRefBuzzer->wait();
  }

  // wait while we are running the senders
  while(sendersDone < senders->size()) {
    sendersBuzzer->wait();
  }

  // wait until all the aggregation pipelines have completed
  while (aggCounter < aggregationPipelines->size()) {
    aggBuzzer->wait();
  }

  /// 6. Should we materialize

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

  return true;
}


void pdb::PDBAggregationPipeAlgorithm::cleanup() {

  // invalidate all the ptrs this should destroy everything
  hashedToSend = nullptr;
  hashedToRecv = nullptr;
  selfReceiver = nullptr;
  senders = nullptr;
  logger = nullptr;
  preaggregationPipelines = nullptr;
  aggregationPipelines = nullptr;
  pageQueues = nullptr;
}

pdb::PDBPhysicalAlgorithmType pdb::PDBAggregationPipeAlgorithm::getAlgorithmType() {
  return DistributedAggregation;
}

pdb::PDBCatalogSetContainerType pdb::PDBAggregationPipeAlgorithm::getOutputContainerType() {
  return PDBCatalogSetContainerType::PDB_CATALOG_SET_MAP_CONTAINER;
}
