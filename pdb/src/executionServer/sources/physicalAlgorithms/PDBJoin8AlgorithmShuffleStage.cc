#include <PDBJoin8AlgorithmShuffleStage.h>
#include <PDBJoin8AlgorithmState.h>
#include <PDBJoinAggregationState.h>
#include <ExJob.h>
#include <ComputePlan.h>
#include <AtomicComputationClasses.h>
#include <Join8SideSender.h>
#include <GenericWork.h>

bool pdb::PDBJoin8AlgorithmShuffleStage::setup(const pdb::Handle<pdb::ExJob> &job,
                                               const pdb::PDBPhysicalAlgorithmStatePtr &state,
                                               const std::shared_ptr<pdb::PDBStorageManagerBackend> &storage,
                                               const std::string &error) {


  // cast the state
  auto s = dynamic_pointer_cast<PDBJoin8AlgorithmState>(state);

  // get catalog client
  auto catalogClient = storage->getFunctionalityPtr<PDBCatalogClient>();
  auto myMgr = storage->getFunctionalityPtr<PDBBufferManagerInterface>();

  // init the plan
  ComputePlan plan(std::make_shared<LogicalPlan>(job->tcap, *job->computations));
  s->logicalPlan = plan.getPlan();

  /// 1. Get the incoming connections to this node.

  // make the object
  UseTemporaryAllocationBlock tmp{1024};
  pdb::Handle<SerConnectToRequest> connectionID = pdb::makeObject<SerConnectToRequest>(job->computationID,
                                                                                       job->jobID,
                                                                                       job->thisNode,
                                                                                       PDBJoinAggregationState::LEFT_JOIN_SIDE_TASK);

  // init the vector for the join sides
  s->joinSideCommunicatorsOut = std::make_shared<std::vector<PDBCommunicatorPtr>>();
  for (int n = 0; n < job->numberOfNodes; n++) {

    // connect to the node
    s->joinSideCommunicatorsOut->push_back(myMgr->connectTo(job->nodes[n]->address,
                                                            job->nodes[n]->backendPort,
                                                            connectionID));
  }

  /// 2. Setup the join side senders

  // wait for left side connections
  connectionID->taskID = PDBJoinAggregationState::LEFT_JOIN_SIDE_TASK;
  s->joinSideCommunicatorsIn = std::make_shared<std::vector<PDBCommunicatorPtr>>();
  for (int n = 0; n < job->numberOfNodes; n++) {

    // set the node id
    connectionID->nodeID = n;

    // wait for the connection
    s->joinSideCommunicatorsIn->push_back(myMgr->waitForConnection(connectionID));

    // check if the socket is open
    if (s->joinSideCommunicatorsIn->back()->isSocketClosed()) {

      // log the error
      s->logger->error("Socket for the left side is closed");

      return false;
    }
  }

  /// 3. Setup the join side readers

  // setup the left senders
  s->joinSideSenders = std::make_shared<std::vector<Join8SideSenderPtr>>();

  // init the senders
  for (auto &comm : *s->joinSideCommunicatorsIn) {

    // init the right senders
    s->joinSideSenders->push_back(std::make_shared<Join8SideSender>(myMgr->getPage(), comm));
  }

  s->joinSideReader = std::make_shared<std::vector<Join8SideReaderPtr>>();

  // get the source page set
  auto sourcePageSet = storage->createPageSetFromPDBSet(sourceSet.database, sourceSet.set, false);
  sourcePageSet->resetPageSet();

  // setup the processing threads
  for(int i = 0; i < job->numberOfProcessingThreads; ++i) {
    s->joinSideReader->push_back(std::make_shared<Join8SideReader>(sourcePageSet, i, job->numberOfNodes, s->joinSideSenders, s->planPage));
  }

  /// 4. Setup the join map creators
  s->joinMapCreators = std::make_shared<std::vector<Join8MapCreatorPtr>>();

  // init the join side creators
  s->shuffledPageSet = storage->createRandomAccessPageSet({0, "intermediate"});
  for (auto &comm : *s->joinSideCommunicatorsOut) {

    // make the creators
    s->joinMapCreators->emplace_back(std::make_shared<Join8MapCreator>(s->shuffledPageSet, comm, s->logger, s->planPage));
  }

  return true;
}

bool pdb::PDBJoin8AlgorithmShuffleStage::run(const pdb::Handle<pdb::ExJob> &job,
                                             const pdb::PDBPhysicalAlgorithmStatePtr &state,
                                             const std::shared_ptr<pdb::PDBStorageManagerBackend> &storage,
                                             const std::string &error) {

  // cast the state
  auto s = dynamic_pointer_cast<PDBJoin8AlgorithmState>(state);

  // stats
  atomic_bool success;
  success = true;

  // create the buzzer
  atomic_int counter;
  counter = 0;
  PDBBuzzerPtr tempBuzzer = make_shared<PDBBuzzer>([&](PDBAlarm myAlarm, atomic_int &cnt) {

    // did we fail?
    if (myAlarm == PDBAlarm::GenericError) {
      success = false;
    }

    // increment the count
    cnt++;
  });

  /// 1. Run the readers

  // run on of the join pipelines
  counter = 0;
  for (int i = 0; i < s->joinSideReader->size(); ++i) {

    // get a worker from the server
    PDBWorkerPtr worker = storage->getWorker();

    // make the work
    PDBWorkPtr myWork = std::make_shared<pdb::GenericWork>([&counter, &success, i, &s](const PDBBuzzerPtr &callerBuzzer) {

      try {

        // run the pipeline
        (*s->joinSideReader)[i]->run();
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

  /// 2. Run the senders

  // create the buzzer
  atomic_int sendCnt;
  sendCnt = 0;

  for (int i = 0; i < s->joinSideSenders->size(); ++i) {

    // get a worker from the server
    PDBWorkerPtr worker = storage->getWorker();

    // make the work
    PDBWorkPtr myWork = std::make_shared<pdb::GenericWork>([&sendCnt, &success, i, s](const PDBBuzzerPtr &callerBuzzer) {

      try {

        // run the pipeline
        (*s->joinSideSenders)[i]->run();
      }
      catch (std::exception &e) {

        // log the error
        s->logger->error(e.what());

        // we failed mark that we have
        success = false;
      }

      // signal that the run was successful
      callerBuzzer->buzz(PDBAlarm::WorkAllDone, sendCnt);
    });

    // run the work
    worker->execute(myWork, tempBuzzer);
  }

  /// 3. Run the join map creators

  // create the buzzer
  atomic_int commCnt;
  commCnt = 0;

  for (const auto& joinMapCreator : *s->joinMapCreators) {

    // get a worker from the server
    PDBWorkerPtr worker = storage->getWorker();

    // make the work
    PDBWorkPtr myWork = std::make_shared<pdb::GenericWork>([&commCnt, &success, joinMapCreator, &s](const PDBBuzzerPtr &callerBuzzer) {

      // run the join map creator
      try {
        // run the join map creator
        joinMapCreator->run();
      }
      catch (std::exception &e) {

        // log the error
        s->logger->error(e.what());

        // we failed mark that we have
        success = false;
      }

      // check if the creator succeeded
      if (!joinMapCreator->getSuccess()) {

        // log the error
        s->logger->error(joinMapCreator->getError());

        // we failed mark that we have
        success = false;
      }

      std::cout << "Ended...\n";

      // signal that the run was successful
      callerBuzzer->buzz(PDBAlarm::WorkAllDone, commCnt);
    });

    // run the work
    worker->execute(myWork, tempBuzzer);
  }


  // wait to finish the pipelines
  while (counter < s->joinSideReader->size()) {
    tempBuzzer->wait();
  }

  // shutdown the senders since the pipelines are done
  for (auto &se : *s->joinSideSenders) {
    se->shutdown();
  }

  // wait for senders to finish
  while (sendCnt < s->joinSideSenders->size()) {
    tempBuzzer->wait();
  }

  // wait until the senders finish
  while (commCnt < s->joinMapCreators->size()) {
    tempBuzzer->wait();
  }

  // since this finished copy all the indices to the state
  for(auto &it : *s->joinMapCreators) {
    s->TIDToRecordMapping.emplace_back(it->extractTIDMap());
  }

  return true;
}

void pdb::PDBJoin8AlgorithmShuffleStage::cleanup(const pdb::PDBPhysicalAlgorithmStatePtr &state) {

  // cast the state
  auto s = dynamic_pointer_cast<PDBJoin8AlgorithmState>(state);

  s->joinMapCreators = nullptr;
  s->joinSideSenders = nullptr;
  s->joinSideReader = nullptr;
  s->planPageQueues = nullptr;
}
