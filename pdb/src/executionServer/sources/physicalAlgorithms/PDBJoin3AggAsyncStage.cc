#include <PDBJoin3AggAsyncStage.h>
#include <PDBJoin3AlgorithmState.h>
#include <PDBJoinAggregationState.h>
#include <ExJob.h>
#include <ComputePlan.h>
#include <AtomicComputationClasses.h>
#include <Join8SideSender.h>
#include <GenericWork.h>


bool pdb::PDBJoin3AggAsyncStage::setup(const pdb::Handle<pdb::ExJob> &job,
                                       const pdb::PDBPhysicalAlgorithmStatePtr &state,
                                       const std::shared_ptr<pdb::PDBStorageManagerBackend> &storage,
                                       const std::string &error) {


  // cast the state
  auto s = dynamic_pointer_cast<PDBJoin3AlgorithmState>(state);

  /// 1. Communication for set A

  auto myMgr = storage->getFunctionalityPtr<PDBBufferManagerInterface>();

  // make the object
  UseTemporaryAllocationBlock tmp{1024};
  pdb::Handle<SerConnectToRequest> connectionID = pdb::makeObject<SerConnectToRequest>(job->computationID,
                                                                                       job->jobID,
                                                                                       job->thisNode,
                                                                                       PDBJoinAggregationState::A_TASK);


  // init the vector for the left sides
  s->aJoinSideCommunicatorsOut = std::make_shared<std::vector<PDBCommunicatorPtr>>();
  for (int n = 0; n < job->numberOfNodes; n++) {

    // skip com to itself
    if(job->thisNode == n) {
      s->aJoinSideCommunicatorsOut->push_back(nullptr);
      continue;
    }

    // connect to the node
    s->aJoinSideCommunicatorsOut->push_back(myMgr->connectTo(job->nodes[n]->address,
                                                             job->nodes[n]->backendPort,
                                                             connectionID));
  }

  // init the vector for the right sides
  connectionID->taskID = PDBJoinAggregationState::A_TASK;
  s->aJoinSideCommunicatorsIn = std::make_shared<std::vector<PDBCommunicatorPtr>>();
  for (int n = 0; n < job->numberOfNodes; n++) {

    // skip com to itself
    if(job->thisNode == n) {
      s->aJoinSideCommunicatorsIn->push_back(nullptr);
      continue;
    }

    // connect to the node
    s->aJoinSideCommunicatorsIn->push_back(myMgr->connectTo(job->nodes[n]->address,
                                                            job->nodes[n]->backendPort,
                                                            connectionID));
  }

  /// 2. Communication for set B

  // init the vector for the left sides
  connectionID->compID = PDBJoinAggregationState::B_TASK;
  s->bJoinSideCommunicatorsOut = std::make_shared<std::vector<PDBCommunicatorPtr>>();
  for (int n = 0; n < job->numberOfNodes; n++) {

    // skip com to itself
    if(job->thisNode == n) {
      s->bJoinSideCommunicatorsOut->push_back(nullptr);
      continue;
    }

    // connect to the node
    s->bJoinSideCommunicatorsOut->push_back(myMgr->connectTo(job->nodes[n]->address,
                                                             job->nodes[n]->backendPort,
                                                             connectionID));
  }

  // init the vector for the right sides
  connectionID->taskID = PDBJoinAggregationState::B_TASK;
  s->bJoinSideCommunicatorsIn = std::make_shared<std::vector<PDBCommunicatorPtr>>();
  for (int n = 0; n < job->numberOfNodes; n++) {

    // skip com to itself
    if(job->thisNode == n) {
      s->bJoinSideCommunicatorsIn->push_back(nullptr);
      continue;
    }

    // connect to the node
    s->bJoinSideCommunicatorsIn->push_back(myMgr->connectTo(job->nodes[n]->address,
                                                            job->nodes[n]->backendPort,
                                                            connectionID));
  }

  /// 2. Communication for set C

  // init the vector for the left sides
  connectionID->compID = PDBJoinAggregationState::C_TASK;
  s->cJoinSideCommunicatorsOut = std::make_shared<std::vector<PDBCommunicatorPtr>>();
  for (int n = 0; n < job->numberOfNodes; n++) {

    // skip com to itself
    if(job->thisNode == n) {
      s->cJoinSideCommunicatorsOut->push_back(nullptr);
      continue;
    }

    // connect to the node
    s->cJoinSideCommunicatorsOut->push_back(myMgr->connectTo(job->nodes[n]->address,
                                                             job->nodes[n]->backendPort,
                                                             connectionID));
  }

  // init the vector for the right sides
  connectionID->taskID = PDBJoinAggregationState::C_TASK;
  s->cJoinSideCommunicatorsIn = std::make_shared<std::vector<PDBCommunicatorPtr>>();
  for (int n = 0; n < job->numberOfNodes; n++) {

    // skip com to itself
    if(job->thisNode == n) {
      s->cJoinSideCommunicatorsIn->push_back(nullptr);
      continue;
    }

    // connect to the node
    s->cJoinSideCommunicatorsIn->push_back(myMgr->connectTo(job->nodes[n]->address,
                                                            job->nodes[n]->backendPort,
                                                            connectionID));
  }

  return true;
}

bool pdb::PDBJoin3AggAsyncStage::run(const pdb::Handle<pdb::ExJob> &job,
                                     const pdb::PDBPhysicalAlgorithmStatePtr &state,
                                     const std::shared_ptr<pdb::PDBStorageManagerBackend> &storage,
                                     const std::string &error) {

  // cast the state
  auto s = dynamic_pointer_cast<PDBJoin3AlgorithmState>(state);

  // repin the plan page
  s->planPage->repin();

  // get the the plan
  pdb::Handle<JoinPlannerResult> plan = ((Record<JoinPlannerResult>*) s->planPage->getBytes())->getRootObject();

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

  // get index for set A
  auto setAIdx = storage->getIndex({0, "myData:A"});

  // senders A
  for(int n = 0; n < job->numberOfNodes; ++n) {

    // get a worker from the server
    PDBWorkerPtr worker = storage->getWorker();

    // make the work
    PDBWorkPtr myWork = std::make_shared<pdb::GenericWork>([&job, &counter, &s, &setAIdx, n, &plan](const PDBBuzzerPtr& callerBuzzer) {

      auto num_nodes = job->numberOfNodes;
      if (job->thisNode != n) {

          // go through all the records for set A we have on this node
          auto &seq = setAIdx->sequential;

          for(auto t : seq) {

              // unpack this
              auto [rowID, colID, page, idx] = t;

              // get the tid for this record
              auto tid = (*plan->records0)[TRABlockMeta(colID, rowID)];

              // check if we need to send this
              if((*(*plan).record_mapping)[tid * num_nodes + n]) {
                  std::cout << "sending " << rowID << colID << '\n';
              }
          }
      }
      else {

      }

      // signal that the run was successful
      callerBuzzer->buzz(PDBAlarm::WorkAllDone, counter);
    });

    // run the work
    worker->execute(myWork, tempBuzzer);
  }

  // receivers
  for(int n = 0; n < job->numberOfNodes; ++n) {

    // get a worker from the server
    PDBWorkerPtr worker = storage->getWorker();

    // make the work
    PDBWorkPtr myWork = std::make_shared<pdb::GenericWork>([&job, &counter, &s, n](const PDBBuzzerPtr& callerBuzzer) {

      if (job->thisNode == n) {

      }
      else {

      }

      // signal that the run was successful
      callerBuzzer->buzz(PDBAlarm::WorkAllDone, counter);
    });

    // run the work
    worker->execute(myWork, tempBuzzer);
  }

  // wait until all the preaggregationPipelines have completed
  while (counter < job->numberOfNodes * 2) {
    tempBuzzer->wait();
  }

  return true;
}

void pdb::PDBJoin3AggAsyncStage::cleanup(const pdb::PDBPhysicalAlgorithmStatePtr &state, const std::shared_ptr<pdb::PDBStorageManagerBackend> &storage) {

  // cast the state
  auto s = dynamic_pointer_cast<pdb::PDBJoin3AlgorithmState>(state);

}
