#include <GenericWork.h>
#include <JoinAggPlanner.h>
#include <PDBPhysicalAlgorithm.h>
#include "AggregateCompBase.h"
#include "PDBJoinAggregationKeyStage.h"
#include "ExJob.h"
#include "PDBPhysicalAlgorithmState.h"
#include "PDBStorageManagerBackend.h"
#include "PDBJoinAggregationState.h"
#include "LogicalPlanTransformer.h"
#include "KeyComputePlan.h"

pdb::PDBJoinAggregationKeyStage::PDBJoinAggregationKeyStage(const pdb::PDBSinkPageSetSpec &sink,
                                                            const pdb::Vector<pdb::PDBSourceSpec> &sources,
                                                            const pdb::String &final_tuple_set,
                                                            const pdb::Vector<pdb::Handle<PDBSourcePageSetSpec>> &secondary_sources,
                                                            const pdb::Vector<pdb::PDBSetObject> &sets_to_materialize,
                                                            const pdb::String &left_input_tuple_set,
                                                            const pdb::String &right_input_tuple_set,
                                                            const pdb::String &join_tuple_set,
                                                            const pdb::PDBSinkPageSetSpec &lhs_key_sink,
                                                            const pdb::PDBSinkPageSetSpec &rhs_key_sink,
                                                            const pdb::PDBSinkPageSetSpec &join_agg_key_sink,
                                                            const pdb::Vector<PDBSourceSpec> &right_sources,
                                                            const PDBSourcePageSetSpec &left_key_source,
                                                            const PDBSourcePageSetSpec &right_key_source,
                                                            const PDBSourcePageSetSpec &plan_source)
    : PDBPhysicalAlgorithmStage(sink, sources, final_tuple_set, secondary_sources, sets_to_materialize),
      leftInputTupleSet(left_input_tuple_set),
      rightInputTupleSet(right_input_tuple_set),
      joinTupleSet(join_tuple_set),
      lhsKeySink(lhs_key_sink),
      rhsKeySink(rhs_key_sink),
      joinAggKeySink(join_agg_key_sink),
      rightSources(right_sources),
      leftKeySource(left_key_source),
      rightKeySource(right_key_source),
      planSource(plan_source) {}

bool pdb::PDBJoinAggregationKeyStage::setup(const Handle<ExJob> &job,
                                            const PDBPhysicalAlgorithmStatePtr &state,
                                            const std::shared_ptr<pdb::PDBStorageManagerBackend> &storage,
                                            const std::string &error) {

  // cast the state
  auto s = dynamic_pointer_cast<PDBJoinAggregationState>(state);

  // make a logical plan
  s->logicalPlan = std::make_shared<LogicalPlan>(job->tcap, *job->computations);

  /// 1. Find the aggregation comp

  // figure out the aggregation
  if(!s->logicalPlan->getComputations().hasConsumer(finalTupleSet)) {
    throw runtime_error("Aggregation key has to have a consumer!");
  }

  // figure out the aggregation computation
  auto consumers = s->logicalPlan->getComputations().getConsumingAtomicComputations(finalTupleSet);
  if(consumers.size() != 1) {
    throw runtime_error("None or multiple aggregation computations!");
  }
  auto aggComp = consumers.front();

  /// 2. Find the key extraction

  // try to find the key extraction of the key
  auto comps = s->logicalPlan->getComputations().findByPredicate([&aggComp](AtomicComputationPtr &c) {

    if(c->getComputationName() == aggComp->getComputationName() &&
        c->getAtomicComputationTypeID() == ApplyLambdaTypeID) {

      // check if it is a key extraction lambda
      auto it = c->getKeyValuePairs()->find("lambdaType");
      return it != c->getKeyValuePairs()->end() && it->second == "key";
    }

    // check if this is a join
    return false;
  });

  // check if we have exactly one key extraction
  if(comps.size() != 1) {
    throw runtime_error("None or multiple key extractions!");
  }
  auto aggKey = *comps.begin();

  /// 3. Run the transformations

  // make the transformer
  auto transformer = std::make_shared<LogicalPlanTransformer>(s->logicalPlan);

  // add the transformation
  transformer->addTransformation(std::make_shared<InsertKeyScanSetsTransformation>(leftInputTupleSet));
  transformer->addTransformation(std::make_shared<InsertKeyScanSetsTransformation>(rightInputTupleSet));
  transformer->addTransformation(std::make_shared<JoinKeySideTransformation>(leftInputTupleSet));
  transformer->addTransformation(std::make_shared<JoinKeySideTransformation>(rightInputTupleSet));
  transformer->addTransformation(std::make_shared<JoinKeyTransformation>(joinTupleSet));
  transformer->addTransformation(std::make_shared<DropDependents>(aggComp->getOutputName()));
  transformer->addTransformation(std::make_shared<AggKeyTransformation>(aggKey->getOutputName()));
  transformer->addTransformation(std::make_shared<AddJoinTID>(joinTupleSet));

  // apply all the transformations
  s->logicalPlan = transformer->applyTransformations();
  std::cout << "Key only plan : \n" << *s->logicalPlan << '\n';

  // make the compute plan
  auto computePlan = std::make_shared<KeyComputePlan>(s->logicalPlan);

  // get catalog client
  auto catalogClient = storage->getFunctionalityPtr<PDBCatalogClient>();

  /// 4. Figure out the left and right key sink tuple set tuple set

  // get the left key sink page set
  auto leftPageSet = storage->createAnonymousPageSet(std::make_pair(lhsKeySink.pageSetIdentifier.first, lhsKeySink.pageSetIdentifier.second));

  // did we manage to get a left key sink page set? if not the setup failed
  if(leftPageSet == nullptr) {
    return false;
  }

  // create a labeled page set from it
  s->labeledLeftPageSet = std::make_shared<PDBLabeledPageSet>(leftPageSet);

  // get the right key sink page set
  auto rightPageSet = storage->createAnonymousPageSet(std::make_pair(rhsKeySink.pageSetIdentifier.first, rhsKeySink.pageSetIdentifier.second));

  // did we manage to get a right key sink page set? if not the setup failed
  if(rightPageSet == nullptr) {
    return false;
  }

  // create a labeled page set from it
  s->labeledRightPageSet = std::make_shared<PDBLabeledPageSet>(rightPageSet);

  // get sink for the join aggregation pipeline
  s->joinAggPageSet = storage->createAnonymousPageSet(std::make_pair(joinAggKeySink.pageSetIdentifier.first, joinAggKeySink.pageSetIdentifier.second));

  // did we manage to get the sink for the join aggregation pipeline? if not the setup failed
  if(s->joinAggPageSet == nullptr) {
    return false;
  }

  /// 5. Initialize the left key pipelines

  // the join key pipelines
  int32_t currNode = job->thisNode;
  s->joinKeyPipelines = make_shared<std::vector<PipelinePtr>>();

  // for each node we create a pipeline
  for(int n = 0; n < job->numberOfNodes; n++) {

    // for each lhs source create a pipeline
    for(uint64_t i = 0; i < sources.size(); i++) {

      // make the parameters for the first set
      std::map<ComputeInfoType, ComputeInfoPtr> params = {{ ComputeInfoType::SOURCE_SET_INFO,
                                                            getKeySourceSetArg(catalogClient, sources, i)} };

      // grab the left page set
      auto lps = s->labeledLeftPageSet->getLabeledView(n);

      // go grab the source page set
      bool isThisNode = job->nodes[currNode]->address == job->nodes[n]->address && job->nodes[currNode]->port == job->nodes[n]->port;
      PDBAbstractPageSetPtr sourcePageSet = isThisNode ? getKeySourcePageSet(storage, i, sources) :
                                            getFetchingPageSet(storage, i,sources, job->nodes[n]->address, job->nodes[n]->port);

      // build the pipeline
      auto leftPipeline = computePlan->buildHashPipeline(leftInputTupleSet, sourcePageSet, lps, params);

      // store the pipeline
      s->joinKeyPipelines->emplace_back(leftPipeline);
    }
  }

  /// 6. Initialize the right key pipelines

  // for each node we create a pipeline
  for(int n = 0; n < job->numberOfNodes; n++) {

    // for each lhs source create a pipeline
    for (uint64_t i = 0; i < rightSources.size(); i++) {

      // make the parameters for the first set
      std::map<ComputeInfoType, ComputeInfoPtr> params = {{ComputeInfoType::SOURCE_SET_INFO,
                                                           getKeySourceSetArg(catalogClient, rightSources, i)}};

      // grab the left page set
      auto rps = s->labeledRightPageSet->getLabeledView(n);

      // go grab the source page set
      bool isThisNode = job->nodes[currNode]->address == job->nodes[n]->address && job->nodes[currNode]->port == job->nodes[n]->port;
      PDBAbstractPageSetPtr sourcePageSet = isThisNode ? getKeySourcePageSet(storage, i, rightSources) :
                                            getFetchingPageSet(storage, i, rightSources, job->nodes[n]->address, job->nodes[n]->port);

      // build the pipeline
      auto rightPipeline = computePlan->buildHashPipeline(rightInputTupleSet, sourcePageSet, rps, params);

      // store the pipeline
      s->joinKeyPipelines->emplace_back(rightPipeline);
    }
  }

  /// 7. Initialize the join-agg key pipeline

  auto &computations = s->logicalPlan->getComputations();

  // try to find the sink
  auto sinkList = computations.findByPredicate([&computations](AtomicComputationPtr &c){

    // check if this is a sink
    return computations.getConsumingAtomicComputations(c->getOutputName()).empty();
  });

  // make sure we have only one join
  if(sinkList.size() != 1){
    throw runtime_error("Could not find an aggregation!");
  }

  // get the join computation
  auto &sinkComp = *sinkList.begin();

  // try to find the join
  auto joinList = computations.findByPredicate([](AtomicComputationPtr &c){

    // check if this is a join
    return c->getAtomicComputationTypeID() == ApplyJoinTypeID;
  });

  // make sure we have only one join
  if(joinList.size() != 1){
    throw runtime_error("Could not find a join!");
  }

  auto &joinComp = *joinList.begin();

  // the key aggregation processor
  auto aggComputation = ((AggregateCompBase*)(&computePlan->getPlan()->getNode(sinkComp->getComputationName()).getComputation()));

  // get the buffer manager
  auto myMgr = storage->getFunctionalityPtr<PDBBufferManagerInterface>();

  // get the left and right key page, basically a map of pdb::Map<Key, TID>
  s->leftKeyPage = myMgr->getPage();
  s->rightKeyPage = myMgr->getPage();

  // this page will contain the plan
  s->planPage = myMgr->getPage();

  // the parameters
  std::map<ComputeInfoType, ComputeInfoPtr> params;
  params = {{ComputeInfoType::PAGE_PROCESSOR, aggComputation->getAggregationKeyProcessor()},
            {ComputeInfoType::JOIN_ARGS, std::make_shared<JoinArguments>(JoinArgumentsInit{{joinComp->getRightInput().getSetName(), std::make_shared<JoinArg>(s->labeledRightPageSet)}})},
            {ComputeInfoType::KEY_JOIN_SOURCE_ARGS, std::make_shared<KeyJoinSourceArgs>(std::vector<PDBPageHandle>({ s->leftKeyPage, s->rightKeyPage }))},
            {ComputeInfoType::SHUFFLE_JOIN_ARG, std::make_shared<ShuffleJoinArg>(false)}};

  s->joinKeyAggPipeline = computePlan->buildJoinAggPipeline(joinComp->getOutputName(),
                                                            sinkComp->getOutputName(),     /* this is the TupleSet the pipeline ends with */
                                                            s->labeledLeftPageSet,
                                                            s->joinAggPageSet,
                                                            params,
                                                            1,
                                                            1,
                                                            20,
                                                            0);

  // if this is the lead node
  if(job->isLeadNode) {

    /// 8. Init the sending queues

    // this is to transfer left key TIDs
    s->leftKeyPageQueues = std::make_shared<std::vector<PDBPageQueuePtr>>();

    // this is to transfer right key TIDs
    s->rightKeyPageQueues = std::make_shared<std::vector<PDBPageQueuePtr>>();

    // this is to transfer the plan
    s->planPageQueues = std::make_shared<std::vector<PDBPageQueuePtr>>();


    /// 9. Init the senders

    // setup the senders for the left key page
    if(!setupSenders(job, s, leftKeySource, storage, s->leftKeyPageQueues, s->leftKeySenders, nullptr)) {

      // log the error
      s->logger->error("Failed to setup the senders for the left keys");

      // return false
      return false;
    }

    // setup the senders for the right key page
    if(!setupSenders(job, s, rightKeySource, storage, s->rightKeyPageQueues, s->rightKeySenders, nullptr)) {

      // log the error
      s->logger->error("Failed to setup the senders for the right keys");

      // return false
      return false;
    }

    // setup the senders for the plan
    if(!setupSenders(job, s, planSource, storage, s->planPageQueues, s->planSenders, nullptr)) {

      // log the error
      s->logger->error("Failed to setup the senders for the plan");

      // return false
      return false;
    }
  }
  else {

      /// 1. Wait to get the plan

      // create to receive the left key page
      s->leftKeyToNodePageSet = storage->createFeedingAnonymousPageSet(std::make_pair(leftKeySource.pageSetIdentifier.first,
                                                                                                leftKeySource.pageSetIdentifier.second),
                                                                                        1,
                                                                                                   job->numberOfNodes);

      // create to receive the right key page
      s->rightKeyToNodePageSet = storage->createFeedingAnonymousPageSet(std::make_pair(rightKeySource.pageSetIdentifier.first,
                                                                                                 rightKeySource.pageSetIdentifier.second),
                                                                                         1,
                                                                                                    job->numberOfNodes);

      // create to receive the plan
      s->planPageSet = storage->createFeedingAnonymousPageSet(std::make_pair(planSource.pageSetIdentifier.first,
                                                                                       planSource.pageSetIdentifier.second),
                                                                               1,
                                                                                          job->numberOfNodes);
  }


  return true;
}

bool pdb::PDBJoinAggregationKeyStage::run(const Handle<ExJob> &job,
                                          const PDBPhysicalAlgorithmStatePtr &state,
                                          const std::shared_ptr<pdb::PDBStorageManagerBackend> &storage,
                                          const std::string &error) {

  // check if it is the lead node
  if(job->isLeadNode) {

    // run the lead node
    return runLead(job, state, storage, error);
  }

  return runFollower(job, state, storage, error);
}

void pdb::PDBJoinAggregationKeyStage::cleanup(const pdb::PDBPhysicalAlgorithmStatePtr &state) {

  // cast the state
  auto s = dynamic_pointer_cast<PDBJoinAggregationState>(state);

  // clear the left senders
  if(s->leftKeySenders != nullptr) {
    s->leftKeySenders->clear();
  }

  // clear the right senders
  if(s->rightKeySenders != nullptr) {
    s->rightKeySenders->clear();
  }

  // clear the plan senders
  if(s->planSenders != nullptr) {
    s->planSenders->clear();
  }

}

pdb::SourceSetArgPtr pdb::PDBJoinAggregationKeyStage::getKeySourceSetArg(std::shared_ptr<pdb::PDBCatalogClient> &catalogClient,
                                                                         const pdb::Vector<PDBSourceSpec> &sources,
                                                                         size_t idx) {

  // grab the source set from the sources
  auto &sourceSet = sources[idx].sourceSet;

  // check if we actually have a set
  if(sourceSet == nullptr) {
    return nullptr;
  }

  // grab the set
  std::string error;
  auto set = catalogClient->getSet(sourceSet->database, sourceSet->set, error);
  if(set == nullptr || !set->isStoringKeys) {
    return nullptr;
  }

  // update the set so it is keyed
  set->name = PDBCatalog::toKeySetName(sourceSet->set);
  set->containerType = PDB_CATALOG_SET_VECTOR_CONTAINER;

  // return the argument
  return std::make_shared<pdb::SourceSetArg>(set);
}

pdb::PDBAbstractPageSetPtr pdb::PDBJoinAggregationKeyStage::getKeySourcePageSet(const std::shared_ptr<pdb::PDBStorageManagerBackend> &storage,
                                                                                size_t idx,
                                                                                const pdb::Vector<PDBSourceSpec> &srcs) {

  // grab the source set from the sources
  auto &sourceSet = srcs[idx].sourceSet;

  // if this is a scan set get the page set from a real set
  PDBAbstractPageSetPtr sourcePageSet;
  if (sourceSet != nullptr) {

    // get the page set
    sourcePageSet = storage->createPageSetFromPDBSet(sourceSet->database, sourceSet->set, true);
    sourcePageSet->resetPageSet();

  } else {

    // we are reading from an existing page set get it
    sourcePageSet = storage->getPageSet(PDBAbstractPageSet::toKeyPageSetIdentifier(srcs[idx].pageSet->pageSetIdentifier));
    sourcePageSet->resetPageSet();
  }

  // return the page set
  return sourcePageSet;
}


pdb::PDBAbstractPageSetPtr pdb::PDBJoinAggregationKeyStage::getFetchingPageSet(const std::shared_ptr<pdb::PDBStorageManagerBackend> &storage,
                                                                               size_t idx,
                                                                               const pdb::Vector<PDBSourceSpec> &srcs,
                                                                               const std::string &ip,
                                                                               int32_t port) {

  // grab the source set from the sources
  auto &sourceSet = srcs[idx].sourceSet;

  // if this is a scan set get the page set from a real set
  PDBAbstractPageSetPtr sourcePageSet;
  if (sourceSet != nullptr) {

    // get the page set
    sourcePageSet = storage->fetchPDBSet(sourceSet->database, sourceSet->set, true, ip, port);
    sourcePageSet->resetPageSet();

  } else {

    // we are reading from an existing page set get it
    sourcePageSet = storage->fetchPageSet(*srcs[idx].pageSet, true, ip, port);
    sourcePageSet->resetPageSet();
  }

  // return the page set
  return sourcePageSet;
}


bool pdb::PDBJoinAggregationKeyStage::setupSenders(const Handle<pdb::ExJob> &job,
                                                   const std::shared_ptr<PDBJoinAggregationState> &state,
                                                   const PDBSourcePageSetSpec &recvPageSet,
                                                   const std::shared_ptr<pdb::PDBStorageManagerBackend> &storage,
                                                   std::shared_ptr<std::vector<PDBPageQueuePtr>> &pageQueues,
                                                   std::shared_ptr<std::vector<PDBPageNetworkSenderPtr>> &senders,
                                                   PDBPageSelfReceiverPtr *selfReceiver) {

  // get the buffer manager
  auto myMgr = storage->getFunctionalityPtr<PDBBufferManagerInterface>();

  // go through the nodes and create the page sets
  int32_t currNode = job->thisNode;
  senders = std::make_shared<std::vector<PDBPageNetworkSenderPtr>>();
  for(unsigned i = 0; i < job->nodes.size(); ++i) {

    // check if it is this node or another node
    if(job->nodes[i]->port == job->nodes[currNode]->port &&
        job->nodes[i]->address == job->nodes[currNode]->address &&
        selfReceiver != nullptr) {

      // create a new queue
      pageQueues->push_back(std::make_shared<PDBPageQueue>());

      // get the receive page set
      auto pageSet = storage->createFeedingAnonymousPageSet(std::make_pair(recvPageSet.pageSetIdentifier.first,
                                                                           recvPageSet.pageSetIdentifier.second),
                                                            job->numberOfProcessingThreads,
                                                            job->numberOfNodes);

      // make sure we can use them all at the same time
      pageSet->setUsagePolicy(PDBFeedingPageSetUsagePolicy::KEEP_AFTER_USED);

      // did we manage to get a page set where we receive this? if not the setup failed
      if(pageSet == nullptr) {
        return false;
      }

      // make the self receiver
      *selfReceiver = std::make_shared<pdb::PDBPageSelfReceiver>(pageQueues->back(), pageSet, myMgr);
    }
    else {

      // create a new queue
      pageQueues->push_back(std::make_shared<PDBPageQueue>());

      // make the sender
      auto sender = std::make_shared<PDBPageNetworkSender>(job->nodes[i]->address,
                                                           job->nodes[i]->port,
                                                           job->numberOfProcessingThreads,
                                                           job->numberOfNodes,
                                                           storage->getConfiguration()->maxRetries,
                                                           state->logger,
                                                           std::make_pair(recvPageSet.pageSetIdentifier.first, recvPageSet.pageSetIdentifier.second),
                                                           pageQueues->back());

      // setup the sender, if we fail return false
      if(!sender->setup()) {
        return false;
      }

      // make the sender
      senders->emplace_back(sender);
    }
  }

  return true;
}

bool pdb::PDBJoinAggregationKeyStage::runLead(const Handle<pdb::ExJob> &job,
                                              const PDBPhysicalAlgorithmStatePtr &state,
                                              const std::shared_ptr<pdb::PDBStorageManagerBackend> &storage,
                                              const std::string &error) {

  // cast the state
  auto s = dynamic_pointer_cast<PDBJoinAggregationState>(state);

  /**
   * 1. Process the left and right key side
   */

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

  // here we get a worker per pipeline and run them all.
  for (int i = 0; i < s->joinKeyPipelines->size(); ++i) {

    // get a worker from the server
    PDBWorkerPtr worker = storage->getWorker();

    // make the work
    PDBWorkPtr myWork = std::make_shared<pdb::GenericWork>([&counter, &success, i, s](const PDBBuzzerPtr& callerBuzzer) {

      try {

        // run the pipeline
        (*s->joinKeyPipelines)[i]->run();
      }
      catch (std::exception &e) {

        // log the error
        s->logger->error(e.what());

        // we failed mark that we have
        success = false;
      }

      // signal that the run was successful
      callerBuzzer->buzz(PDBAlarm::WorkAllDone, counter);
    });

    // run the work
    worker->execute(myWork, tempBuzzer);
  }

  // wait until all the preaggregationPipelines have completed
  while (counter < s->joinKeyPipelines->size()) {
    tempBuzzer->wait();
  }

  // clear the pipeline
  s->joinKeyPipelines->clear();

  // reset the left and the right page set
  s->labeledLeftPageSet->resetPageSet();
  s->labeledRightPageSet->resetPageSet();

  /**
   * 2. Run the join aggregation pipeline
   */

  try {

    // run the pipeline
    s->joinKeyAggPipeline->run();
  }
  catch (std::exception &e) {

    // log the error
    s->logger->error(e.what());

    // we failed mark that we have
    return false;
  }

  /**
   * 3. Run the planner
   */

  // make the planner
  JoinAggPlanner planner(s->joinAggPageSet, job->numberOfNodes, job->numberOfProcessingThreads, s->planPage);

  // get a page to store the planning result onto
  auto bufferManager = storage->getFunctionalityPtr<PDBBufferManagerInterface>();

  // do the planning
  planner.doPlanning();

  /**
   * 4. Broadcast the plan (left key page, right key page)
   */

  /// 4.1 Start the senders

  counter = 0;
  PDBBuzzerPtr broadcastKeyPagesBuzzer = make_shared<PDBBuzzer>([&](PDBAlarm myAlarm, atomic_int &cnt) {

    // did we fail?
    if(myAlarm == PDBAlarm::GenericError) {
      success = false;
    }

    // increment the count
    cnt++;
  });


  /// 4.2 broadcast the left key page

  for(const auto& q : *s->leftKeyPageQueues) {
    q->enqueue(s->leftKeyPage);
    q->enqueue(nullptr);
  }

  /// 4.3 broadcast the right key page

  for(const auto& q : *s->rightKeyPageQueues) {
    q->enqueue(s->rightKeyPage);
    q->enqueue(nullptr);
  }

  /// 4.4 run the senders for the left side

  for(const auto &sender : *s->leftKeySenders) {

    // get a worker from the server
    PDBWorkerPtr worker = storage->getWorker();

    // make the work
    PDBWorkPtr myWork = std::make_shared<pdb::GenericWork>([&sender, &success, &counter, s](const PDBBuzzerPtr& callerBuzzer) {

      try {

        // run the pipeline
        sender->run();
      }
      catch (std::exception &e) {

        // log the error
        s->logger->error(e.what());

        // we failed mark that we have
        success = false;
      }

      // signal that the run was successful
      callerBuzzer->buzz(PDBAlarm::WorkAllDone, counter);
    });

    // run the work
    worker->execute(myWork, tempBuzzer);
  }

  /// 4.5 run the senders for the right side

  for(const auto &sender : *s->rightKeySenders) {

    // get a worker from the server
    PDBWorkerPtr worker = storage->getWorker();

    // make the work
    PDBWorkPtr myWork = std::make_shared<pdb::GenericWork>([&sender, &success, &counter, s](const PDBBuzzerPtr& callerBuzzer) {

      try {

        // run the pipeline
        sender->run();
      }
      catch (std::exception &e) {

        // log the error
        s->logger->error(e.what());

        // we failed mark that we have
        success = false;
      }

      // signal that the run was successful
      callerBuzzer->buzz(PDBAlarm::WorkAllDone, counter);
    });

    // run the work
    worker->execute(myWork, tempBuzzer);
  }

  /**
   * 5. Broadcast to each node the plan except for this one
   */

  /// 5.1 Fill up the plan queues

  for(const auto& q : *s->planPageQueues) {
    q->enqueue(s->planPage);
    q->enqueue(nullptr);
  }

  /// 5.2 Run the senders

  for(const auto &sender : *s->planSenders) {

    // get a worker from the server
    PDBWorkerPtr worker = storage->getWorker();

    // make the work
    PDBWorkPtr myWork = std::make_shared<pdb::GenericWork>([&sender, &success, &counter, s](const PDBBuzzerPtr& callerBuzzer) {

      try {

        // run the pipeline
        sender->run();
      }
      catch (std::exception &e) {

        // log the error
        s->logger->error(e.what());

        // we failed mark that we have
        success = false;
      }

      // signal that the run was successful
      callerBuzzer->buzz(PDBAlarm::WorkAllDone, counter);
    });

    // run the work
    worker->execute(myWork, tempBuzzer);
  }

  /**
   * 6. Wait for everything to finish
   */

  // wait for all the left senders to finish
  // for all the right senders to finish
  // the plan senders
  while (counter < s->leftKeySenders->size() + s->rightKeySenders->size() + s->planSenders->size()) {
    tempBuzzer->wait();
  }

  return true;
}

bool pdb::PDBJoinAggregationKeyStage::runFollower(const Handle<pdb::ExJob> &job,
                                                  const pdb::PDBPhysicalAlgorithmStatePtr &state,
                                                  const std::shared_ptr<pdb::PDBStorageManagerBackend> &storage,
                                                  const std::string &error) {

  // cast the state
  auto s = dynamic_pointer_cast<PDBJoinAggregationState>(state);

  /**
   * 1. Wait to left and right join map
   */
  auto myMgr = storage->getFunctionalityPtr<PDBBufferManagerInterface>();

  /// TODO this needs to be rewritten using the new methods for direct communication
  auto tmp = s->leftKeyToNodePageSet->getNextPage(0);
  memcpy(s->leftKeyPage->getBytes(), tmp->getBytes(), tmp->getSize());

  /// TODO this needs to be rewritten using the new methods for direct communication
  tmp = s->rightKeyToNodePageSet->getNextPage(0);
  memcpy(s->rightKeyPage->getBytes(), tmp->getBytes(), tmp->getSize());

  /**
   * 2. Wait to receive the plan
   */
  /// TODO this needs to be rewritten using the new methods for direct communication
  tmp = s->planPageSet->getNextPage(0);
  memcpy(s->planPage->getBytes(), tmp->getBytes(), tmp->getSize());

  return true;
}