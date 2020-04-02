#include <PDBPageSelfReceiver.h>
#include "physicalAlgorithms/PDBJoinAggregationAggregationStage.h"
#include "PDBJoinAggregationState.h"
#include "GenericWork.h"
#include "ExJob.h"
#include "ComputePlan.h"
#include "AtomicComputationClasses.h"
#include "PreaggregationPageProcessor.h"

pdb::PDBJoinAggregationAggregationStage::PDBJoinAggregationAggregationStage(const pdb::PDBSinkPageSetSpec &sink,
                                                                            const pdb::PDBSinkPageSetSpec &preaggIntermediate,
                                                                            const pdb::Vector<pdb::PDBSourceSpec> &sources,
                                                                            const pdb::String &final_tuple_set,
                                                                            const pdb::Vector<pdb::Handle<PDBSourcePageSetSpec>> &secondary_sources,
                                                                            const pdb::Vector<pdb::PDBSetObject> &sets_to_materialize,
                                                                            const pdb::String &join_tuple_set)
    : PDBPhysicalAlgorithmStage(sink, sources, final_tuple_set, secondary_sources, sets_to_materialize),
      joinTupleSet(join_tuple_set), preaggIntermediate(preaggIntermediate) {}

bool pdb::PDBJoinAggregationAggregationStage::setup(const pdb::Handle<pdb::ExJob> &job,
                                                         const pdb::PDBPhysicalAlgorithmStatePtr &state,
                                                         const std::shared_ptr<pdb::PDBStorageManagerBackend> &storage,
                                                         const std::string &error) {

  // cast the state
  auto s = dynamic_pointer_cast<PDBJoinAggregationState>(state);

  // cast the buffer manager
  auto myMgr = storage->getFunctionalityPtr<PDBBufferManagerInterface>();

  /// 15. Setup the aggregation pipeline

  // init the plan
  auto plan = ComputePlan(std::make_shared<LogicalPlan>(job->tcap, *job->computations));
  auto logicalPlan = plan.getPlan();

  // get the join computation
  auto joinAtomicComp =
      dynamic_pointer_cast<ApplyJoin>(logicalPlan->getComputations().getProducingAtomicComputation(joinTupleSet));

  // the join arguments
  auto joinArguments = std::make_shared<JoinArguments>(JoinArgumentsInit{{joinAtomicComp->getRightInput().getSetName(),
                                                                          std::make_shared<JoinArg>(s->rightShuffledPageSet)}});

  // mark that this is the join aggregation algorithm
  joinArguments->isJoinAggAggregation = true;
  joinArguments->isLocalJoinAggAggregation = false;

  // set the page that contains the mapping from aggregation key to tid
  joinArguments->aggKeyPage = s->aggKeyPage;

  // set the left and right mappings
  joinArguments->leftTIDToRecordMapping = &s->leftTIDToRecordMapping;
  joinArguments->rightTIDToRecordMapping = &s->rightTIDToRecordMapping;

  // set the plan page
  joinArguments->planPage = s->planPage;

  /// 15.1 Init the preaggregation queues

  s->pageQueues = std::make_shared<std::vector<PDBPageQueuePtr>>();
  for(int i = 0; i < job->numberOfNodes; ++i) { s->pageQueues->emplace_back(std::make_shared<PDBPageQueue>()); }

  // fill uo the vector for each thread
  std::map<ComputeInfoType, ComputeInfoPtr> params;
  s->preaggregationPipelines = std::make_shared<std::vector<PipelinePtr>>();
  for (uint64_t pipelineIndex = 0; pipelineIndex < job->numberOfProcessingThreads; ++pipelineIndex) {

    /// 15.2. Figure out the parameters of the pipeline

    // initialize the parameters
    params = {{ComputeInfoType::PAGE_PROCESSOR,
               std::make_shared<PreaggregationPageProcessor>(job->numberOfNodes, // we use one since this pipeline is completely local.
                                                             job->numberOfProcessingThreads,
                                                             *s->pageQueues,
                                                             myMgr)},
              {ComputeInfoType::JOIN_ARGS, joinArguments},
              {ComputeInfoType::SHUFFLE_JOIN_ARG, std::make_shared<ShuffleJoinArg>(false)},
              {ComputeInfoType::SOURCE_SET_INFO, nullptr}};

    /// 15.3. Build the pipeline

    auto pipeline = plan.buildPipeline(joinTupleSet, /* this is the TupleSet the pipeline starts with */
                                       finalTupleSet,     /* this is the TupleSet the pipeline ends with */
                                       s->leftShuffledPageSet,
                                       s->intermediatePageSet,
                                       params,
                                       job->thisNode,
                                       job->numberOfNodes, // we use one since this pipeline is completely local.
                                       job->numberOfProcessingThreads,
                                       pipelineIndex);

    s->preaggregationPipelines->push_back(pipeline);
  }

  /// 8. Create the aggregation pipeline

  // get the sink page set
  auto sinkPageSet =
      storage->createAnonymousPageSet(std::make_pair(sink.pageSetIdentifier.first, sink.pageSetIdentifier.second));

  // did we manage to get a sink page set? if not the setup failed
  if (sinkPageSet == nullptr) {
    return false;
  }

  // we are putting the pages from the queues here
  auto recvPageSet = storage->createFeedingAnonymousPageSet(std::make_pair(preaggIntermediate.pageSetIdentifier.first, (std::string) preaggIntermediate.pageSetIdentifier.second),
                                                            job->numberOfProcessingThreads,
                                                            job->numberOfNodes);

  // did we manage to get a page set where we receive this? if not the setup failed
  if(recvPageSet == nullptr) {
    return false;
  }

  /// 7. Create the self receiver to forward pages that are created on this node and the network senders to forward pages for the other nodes

  int32_t currNode = job->thisNode;
  s->senders = std::make_shared<std::vector<PDBPageNetworkSenderPtr>>();
  for(unsigned i = 0; i < job->nodes.size(); ++i) {

    // check if it is this node or another node
    if(job->nodes[i]->port == job->nodes[currNode]->port && job->nodes[i]->address == job->nodes[currNode]->address) {

      // make the self receiver
      s->selfReceiver = std::make_shared<pdb::PDBPageSelfReceiver>(s->pageQueues->at(i), recvPageSet, myMgr);
    }
    else {

      // make the sender
      auto sender = std::make_shared<PDBPageNetworkSender>(job->nodes[i]->address,
                                                           job->nodes[i]->port,
                                                           job->numberOfProcessingThreads,
                                                           job->numberOfNodes,
                                                           storage->getConfiguration()->maxRetries,
                                                           s->logger,
                                                           std::make_pair(preaggIntermediate.pageSetIdentifier.first, (std::string) preaggIntermediate.pageSetIdentifier.second),
                                                           s->pageQueues->at(i));

      // setup the sender, if we fail return false
      if(!sender->setup()) {
        return false;
      }

      // make the sender
      s->senders->emplace_back(sender);
    }
  }

  /// 8. Create the aggregation pipeline

  s->aggregationPipelines = std::make_shared<std::vector<PipelinePtr>>();
  for (uint64_t workerID = 0; workerID < job->numberOfProcessingThreads; ++workerID) {

    // build the aggregation pipeline
    auto aggPipeline = plan.buildAggregationPipeline(finalTupleSet, recvPageSet, sinkPageSet, workerID);

    // store the aggregation pipeline
    s->aggregationPipelines->push_back(aggPipeline);
  }

  return true;
}

bool pdb::PDBJoinAggregationAggregationStage::run(const pdb::Handle<pdb::ExJob> &job,
                                                       const pdb::PDBPhysicalAlgorithmStatePtr &state,
                                                       const std::shared_ptr<pdb::PDBStorageManagerBackend> &storage,
                                                       const std::string &error) {

  // cast the state
  auto s = dynamic_pointer_cast<PDBJoinAggregationState>(state);

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
  for (int workerID = 0; workerID < s->aggregationPipelines->size(); ++workerID) {

    // get a worker from the server
    PDBWorkerPtr worker = storage->getWorker();

    // make the work
    PDBWorkPtr myWork = std::make_shared<pdb::GenericWork>([&aggCounter, &success, workerID, &s](const PDBBuzzerPtr& callerBuzzer) {

      try {

        // run the pipeline
        (*s->aggregationPipelines)[workerID]->run();
      }
      catch (std::exception &e) {

        // log the error
        s->logger->error(e.what());

        // we failed mark that we have
        success = false;
      }

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
    PDBWorkPtr myWork = std::make_shared<pdb::GenericWork>([&selfRecDone, &s](const PDBBuzzerPtr& callerBuzzer) {

      // run the receiver
      if(s->selfReceiver->run()) {

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
  sendersDone = s->senders->size();
  PDBBuzzerPtr sendersBuzzer = make_shared<PDBBuzzer>([&](PDBAlarm myAlarm, atomic_int &cnt) {

    // did we fail?
    if (myAlarm == PDBAlarm::GenericError) {
      success = false;
    }

    // we are done here
    cnt++;
  });

  // go through each sender and run them
  for(auto &sender : *s->senders) {

    // make the work
    PDBWorkPtr myWork = std::make_shared<pdb::GenericWork>([&sendersDone, sender, this](const PDBBuzzerPtr& callerBuzzer) {

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
  for (int workerID = 0; workerID < s->preaggregationPipelines->size(); ++workerID) {

    // get a worker from the server
    PDBWorkerPtr worker = storage->getWorker();

    // make the work
    PDBWorkPtr myWork = std::make_shared<pdb::GenericWork>([&preaggCounter, &success, workerID, &s](const PDBBuzzerPtr& callerBuzzer) {

      try {

        // run the pipeline
        (*s->preaggregationPipelines)[workerID]->run();
      }
      catch (std::exception &e) {

        // log the error
        s->logger->error(e.what());

        // we failed mark that we have
        success = false;
      }

      // signal that the run was successful
      callerBuzzer->buzz(PDBAlarm::WorkAllDone, preaggCounter);
    });

    // run the work
    worker->execute(myWork, preaggBuzzer);
  }

  /// 5. Do the waiting

  // wait until all the preaggregationPipelines have completed
  while (preaggCounter < s->preaggregationPipelines->size()) {
    preaggBuzzer->wait();
  }

  // ok they have finished now push a null page to each of the preagg queues
  for(auto &queue : *s->pageQueues) { queue->enqueue(nullptr); }

  // wait while we are running the receiver
  while(selfRecDone == 0) {
    selfRefBuzzer->wait();
  }

  // wait while we are running the senders
  while(sendersDone < s->senders->size()) {
    sendersBuzzer->wait();
  }

  // wait until all the aggregation pipelines have completed
  while (aggCounter < s->aggregationPipelines->size()) {
    aggBuzzer->wait();
  }

  /// 6. Should we materialize

  // should we materialize this to a set?
  for(int j = 0; j < setsToMaterialize.size(); ++j) {

    // get the page set
    auto sinkPageSet = storage->getPageSet(std::make_pair(sink.pageSetIdentifier.first, sink.pageSetIdentifier.second));

    // if the thing does not exist finish!
    if(sinkPageSet == nullptr) {
      success = false;
      break;
    }

    // materialize the page set
    sinkPageSet->resetPageSet();
    success = storage->materializePageSet(sinkPageSet, std::make_pair<std::string, std::string>(setsToMaterialize[j].database, setsToMaterialize[j].set)) && success;
  }

  return success;
}

void pdb::PDBJoinAggregationAggregationStage::cleanup(const pdb::PDBPhysicalAlgorithmStatePtr &state) {}
