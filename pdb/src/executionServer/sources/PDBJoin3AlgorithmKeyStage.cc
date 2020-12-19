#include <PDBJoin3AlgorithmKeyStage.h>
#include <ExJob.h>
#include <GenericWork.h>
#include <JoinPlanner.h>
#include <JoinPlannerResult.h>

bool pdb::PDBJoin3AlgorithmKeyStage::setup(const pdb::Handle<pdb::ExJob> &job,
                                           const pdb::PDBPhysicalAlgorithmStatePtr &state,
                                           const std::shared_ptr<pdb::PDBStorageManagerBackend> &storage,
                                           const std::string &error) {

  // cast the state
  auto s = dynamic_pointer_cast<PDBJoin3AlgorithmState>(state);

  // get a page to store the planning result onto
  auto bufferManager = storage->getFunctionalityPtr<PDBBufferManagerInterface>();

  // make a logical plan
  s->logicalPlan = std::make_shared<LogicalPlan>(job->tcap, *job->computations);

  // get catalog client
  auto catalogClient = storage->getFunctionalityPtr<PDBCatalogClient>();

  // this page will contain the plan
  s->planPage = bufferManager->getPage();

  // the current node
  int32_t currNode = job->thisNode;

  PDBSourcePageSetSpec planSource;
  planSource.sourceType = SinglePageSource;
  planSource.pageSetIdentifier.first = 0;
  planSource.pageSetIdentifier.second = "";

  if(job->isLeadNode) {

    // go through each node for set A
    for(int n = 0; n < job->numberOfNodes; n++) {

        // make the parameters for the first set
        std::map<ComputeInfoType, ComputeInfoPtr> params = {{ ComputeInfoType::SOURCE_SET_INFO,
                                                              getKeySourceSetArg(catalogClient)} };

        // go grab the source page set
        bool isThisNode = job->nodes[currNode]->address == job->nodes[n]->address && job->nodes[currNode]->port == job->nodes[n]->port;
        PDBAbstractPageSetPtr sourcePageSet = isThisNode ? getKeySourcePageSet(storage, sources, 0) :
                                                           getFetchingPageSet(storage, sources, job->nodes[n]->address, job->nodes[n]->port, 0);

        // store the pipeline
        s->keySourcePageSets0[n] = sourcePageSet;
    }

    // go through each node for set B
    for(int n = 0; n < job->numberOfNodes; n++) {

      // make the parameters for the first set
      std::map<ComputeInfoType, ComputeInfoPtr> params = {{ ComputeInfoType::SOURCE_SET_INFO,
                                                            getKeySourceSetArg(catalogClient)} };

      // go grab the source page set C
      bool isThisNode = job->nodes[currNode]->address == job->nodes[n]->address && job->nodes[currNode]->port == job->nodes[n]->port;
      PDBAbstractPageSetPtr sourcePageSet = isThisNode ? getKeySourcePageSet(storage, sources, 0) :
                                            getFetchingPageSet(storage, sources, job->nodes[n]->address, job->nodes[n]->port, 0);

      // store the pipeline
      s->keySourcePageSets1[n] = sourcePageSet;
    }

    // go through each node
    for(int n = 0; n < job->numberOfNodes; n++) {

      // make the parameters for the first set
      std::map<ComputeInfoType, ComputeInfoPtr> params = {{ ComputeInfoType::SOURCE_SET_INFO,
                                                            getKeySourceSetArg(catalogClient)} };

      // go grab the source page set
      bool isThisNode = job->nodes[currNode]->address == job->nodes[n]->address && job->nodes[currNode]->port == job->nodes[n]->port;
      PDBAbstractPageSetPtr sourcePageSet = isThisNode ? getKeySourcePageSet(storage, sources, 0) :
                                            getFetchingPageSet(storage, sources, job->nodes[n]->address, job->nodes[n]->port, 0);

      // store the pipeline
      s->keySourcePageSets2[n] = sourcePageSet;
    }

    // this is the pipeline that runs the key join
    s->keyPipeline = std::make_shared<Join3KeyPipeline>(s->keySourcePageSets0, s->keySourcePageSets1, s->keySourcePageSets2);

    // this is to transfer the plan
    s->planPageQueues = std::make_shared<std::vector<PDBPageQueuePtr>>();

    // setup the senders for the plan
    if(!setupSenders(job, s, planSource, storage, s->planPageQueues, s->planSenders, nullptr)) {

      // log the error
      s->logger->error("Failed to setup the senders for the plan");

      // return false
      return false;
    }
  }
  else {

    // create to receive the plan
    s->planPageSet = storage->createFeedingAnonymousPageSet(std::make_pair(planSource.pageSetIdentifier.first,
                                                                           planSource.pageSetIdentifier.second),
                                                            1,
                                                            job->numberOfNodes);
  }
  return true;
}

bool pdb::PDBJoin3AlgorithmKeyStage::setupSenders(const Handle<pdb::ExJob> &job,
                                                  const std::shared_ptr<PDBJoin3AlgorithmState> &state,
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

pdb::SourceSetArgPtr pdb::PDBJoin3AlgorithmKeyStage::getKeySourceSetArg(std::shared_ptr<pdb::PDBCatalogClient> &catalogClient) {

  // grab the set
  std::string error;
  auto set = catalogClient->getSet(sourceSet0.database, sourceSet0.set, error);
  if(set == nullptr || !set->isStoringKeys) {
    return nullptr;
  }

  // update the set so it is keyed
  set->name = PDBCatalog::toKeySetName(sourceSet0.set);
  set->containerType = PDB_CATALOG_SET_VECTOR_CONTAINER;

  // return the argument
  return std::make_shared<pdb::SourceSetArg>(set);
}

bool pdb::PDBJoin3AlgorithmKeyStage::run(const pdb::Handle<pdb::ExJob> &job,
                                         const pdb::PDBPhysicalAlgorithmStatePtr &state,
                                         const std::shared_ptr<pdb::PDBStorageManagerBackend> &storage,
                                         const std::string &error) {
  // check if it is the lead node
  if(job->isLeadNode) {

    // run the lead node
    return runLead(job, state, storage, error);
  }

  return runFollower(job, state, storage, error);
}

void pdb::PDBJoin3AlgorithmKeyStage::cleanup(const pdb::PDBPhysicalAlgorithmStatePtr &state) {

  // cast the state
  auto s = dynamic_pointer_cast<PDBJoin3AlgorithmState>(state);

  // remove the key pipelines
  s->keyPipeline = nullptr;
}

bool pdb::PDBJoin3AlgorithmKeyStage::runLead(const pdb::Handle<pdb::ExJob> &job,
                                             const pdb::PDBPhysicalAlgorithmStatePtr &state,
                                             const std::shared_ptr<pdb::PDBStorageManagerBackend> &storage,
                                             const std::string &error) {
  // cast the state
  auto s = dynamic_pointer_cast<PDBJoin3AlgorithmState>(state);

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

  // go through the nodes and execute the key pipelines
  for(int set = 0; set < 3; ++set) {
    for(int n = 0; n < job->numberOfNodes; n++) {

      // get a worker from the server
      PDBWorkerPtr worker = storage->getWorker();

      // make the work
      PDBWorkPtr myWork = std::make_shared<pdb::GenericWork>([&s, n, set, &success, &counter] (const PDBBuzzerPtr& callerBuzzer) {

        try {

          // run the pipeline
          s->keyPipeline->runSide(n, set);
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
  }


  // wait for it to finish
  while (counter < job->numberOfNodes * 3) {
    tempBuzzer->wait();
  }

  // run the join
  s->keyPipeline->runJoin();

  // make the join planner
  JoinPlanner planner(job->numberOfNodes,
                      job->numberOfProcessingThreads,
                      s->keyPipeline->nodeRecords0,
                      s->keyPipeline->nodeRecords1,
                      s->keyPipeline->nodeRecords2,
                      s->keyPipeline->joined,
                      s->keyPipeline->aggGroups);

  // do the planning
  planner.doPlanning(s->planPage);

  /**
   * 5. Broadcast to each node the plan except for this one
   */

  /// 5.1 Fill up the plan queues

  for(const auto& q : *s->planPageQueues) {
    q->enqueue(s->planPage);
    q->enqueue(nullptr);
  }

  /// 5.2 Run the senders

  counter = 0;
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
  while (counter < s->planSenders->size()) {
    tempBuzzer->wait();
  }

  return true;
}

bool pdb::PDBJoin3AlgorithmKeyStage::runFollower(const pdb::Handle<pdb::ExJob> &job,
                                                 const pdb::PDBPhysicalAlgorithmStatePtr &state,
                                                 const std::shared_ptr<pdb::PDBStorageManagerBackend> &storage,
                                                 const std::string &error) {
  // cast the state
  auto s = dynamic_pointer_cast<PDBJoin3AlgorithmState>(state);

  /**
   * 2. Wait to receive the plan
   */
  /// TODO this needs to be rewritten using the new methods for direct communication
  auto tmp = s->planPageSet->getNextPage(0);
  memcpy(s->planPage->getBytes(), tmp->getBytes(), tmp->getSize());

  return true;
}

pdb::PDBAbstractPageSetPtr pdb::PDBJoin3AlgorithmKeyStage::getKeySourcePageSet(const std::shared_ptr<pdb::PDBStorageManagerBackend> &storage,
                                                                               const pdb::Vector<PDBSourceSpec> &srcs, int32_t idx) {

  const PDBSetObject *_set;
  if(idx == 0) {
    _set = &sourceSet0;
  }
  else if(idx == 1) {
    _set = &sourceSet1;
  }
  else if(idx == 2) {
    _set = &sourceSet2;
  }

  // if this is a scan set get the page set from a real set
  PDBAbstractPageSetPtr sourcePageSet = storage->createPageSetFromPDBSet(_set->database, _set->set, true);
  sourcePageSet->resetPageSet();

  // return the page set
  return sourcePageSet;
}

pdb::PDBAbstractPageSetPtr pdb::PDBJoin3AlgorithmKeyStage::getFetchingPageSet(const std::shared_ptr<pdb::PDBStorageManagerBackend> &storage,
                                                                              const pdb::Vector<PDBSourceSpec> &srcs,
                                                                              const std::string &ip,
                                                                              int32_t port,
                                                                              int32_t idx) {
  // get the page set
  PDBAbstractPageSetPtr sourcePageSet = storage->fetchPDBSet(sourceSet0.database, sourceSet0.set, true, ip, port);
  sourcePageSet->resetPageSet();

  // return the page set
  return sourcePageSet;
}
