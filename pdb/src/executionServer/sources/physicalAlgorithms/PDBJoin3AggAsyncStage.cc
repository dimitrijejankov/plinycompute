#include <PDBJoin3AggAsyncStage.h>
#include <PDBJoin3AlgorithmState.h>
#include <PDBJoinAggregationState.h>
#include <ExJob.h>
#include <ComputePlan.h>
#include <mkl.h>
#include <AtomicComputationClasses.h>
#include <Join8SideSender.h>
#include <JoinPlannerResult.h>
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


  // init the vector for connections for A
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

  // init the vector connections for A
  connectionID->taskID = PDBJoinAggregationState::A_TASK;
  s->aJoinSideCommunicatorsOut = std::make_shared<std::vector<PDBCommunicatorPtr>>();
  for (int n = 0; n < job->numberOfNodes; n++) {

    // set the node id
    connectionID->nodeID = n;

    // skip com to itself
    if(job->thisNode == n) {
      s->aJoinSideCommunicatorsOut->push_back(nullptr);
      continue;
    }

    // wait for the connection
    s->aJoinSideCommunicatorsOut->push_back(myMgr->waitForConnection(connectionID));

    // check if the socket is open
    if (s->aJoinSideCommunicatorsOut->back()->isSocketClosed()) {

      // log the error
      s->logger->error("Socket for the left side is closed");

      return false;
    }
  }

  /// 2. Communication for set B

  connectionID->nodeID = job->thisNode;
  connectionID->taskID = PDBJoinAggregationState::B_TASK;

  // init the vector for connections for B
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

  // init the vector connections for B
  connectionID->taskID = PDBJoinAggregationState::B_TASK;
  s->bJoinSideCommunicatorsOut = std::make_shared<std::vector<PDBCommunicatorPtr>>();
  for (int n = 0; n < job->numberOfNodes; n++) {

    // set the node id
    connectionID->nodeID = n;

    // skip com to itself
    if(job->thisNode == n) {
      s->bJoinSideCommunicatorsOut->push_back(nullptr);
      continue;
    }

    // wait for the connection
    s->bJoinSideCommunicatorsOut->push_back(myMgr->waitForConnection(connectionID));

    // check if the socket is open
    if (s->bJoinSideCommunicatorsOut->back()->isSocketClosed()) {

      // log the error
      s->logger->error("Socket for the left side is closed");

      return false;
    }
  }

  /// 3. Communication for set C

  connectionID->nodeID = job->thisNode;
  connectionID->taskID = PDBJoinAggregationState::C_TASK;

  // init the vector for connections for C
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

  // init the vector connections for C
  connectionID->taskID = PDBJoinAggregationState::C_TASK;
  s->cJoinSideCommunicatorsOut = std::make_shared<std::vector<PDBCommunicatorPtr>>();
  for (int n = 0; n < job->numberOfNodes; n++) {

    // set the node id
    connectionID->nodeID = n;

    // skip com to itself
    if(job->thisNode == n) {
      s->cJoinSideCommunicatorsOut->push_back(nullptr);
      continue;
    }

    // wait for the connection
    s->cJoinSideCommunicatorsOut->push_back(myMgr->waitForConnection(connectionID));

    // check if the socket is open
    if (s->cJoinSideCommunicatorsOut->back()->isSocketClosed()) {

      // log the error
      s->logger->error("Socket for the left side is closed");

      return false;
    }
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

  // the records we need to join
  std::mutex m;
  std::condition_variable cv;
  std::vector<emitter_row_t> to_join(plan->joinedRecords->size());

  // the records to join group
  std::vector<std::vector<int32_t>> records_to_join(plan->records0->size() +
      plan->records1->size() +
      plan->records2->size());

  // fill the records
  auto &jr = *plan->joinedRecords;
  for(int32_t i = 0; i < jr.size(); ++i) {

    // store what we need to
    records_to_join[jr[i].first].push_back(i);
    records_to_join[jr[i].second].push_back(i);
    records_to_join[jr[i].third].push_back(i);
  }

  // the joined records
  std::vector<int32_t> joined;

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

  // run the senders for each set
  setup_set_comm("A", m, cv, counter, joined, records_to_join, to_join, plan, job, storage, state, tempBuzzer);
  setup_set_comm("B", m, cv, counter, joined, records_to_join, to_join, plan, job, storage, state, tempBuzzer);
  setup_set_comm("C", m, cv, counter, joined, records_to_join, to_join, plan, job, storage, state, tempBuzzer);

  // wait until all the preaggregationPipelines have completed
  while (counter < job->numberOfNodes * 6) {
    tempBuzzer->wait();
  }

  return true;
}

void pdb::PDBJoin3AggAsyncStage::cleanup(const pdb::PDBPhysicalAlgorithmStatePtr &state, const std::shared_ptr<pdb::PDBStorageManagerBackend> &storage) {

  // cast the state
  auto s = dynamic_pointer_cast<pdb::PDBJoin3AlgorithmState>(state);

}

void pdb::PDBJoin3AggAsyncStage::setup_set_comm(const std::string &set,
                                                std::mutex &m,
                                                std::condition_variable &cv,
                                                atomic_int &counter,
                                                std::vector<int32_t> &joined,
                                                std::vector<std::vector<int32_t>> &records_to_join,
                                                std::vector<emitter_row_t> &to_join,
                                                pdb::Handle<JoinPlannerResult> &plan,
                                                const pdb::Handle<pdb::ExJob> &job,
                                                const std::shared_ptr<pdb::PDBStorageManagerBackend> &storage,
                                                const pdb::PDBPhysicalAlgorithmStatePtr &state,
                                                PDBBuzzerPtr &tempBuzzer) {

  // cast the state
  auto s = dynamic_pointer_cast<PDBJoin3AlgorithmState>(state);

  // get index for set A
  auto setIdx = storage->getIndex({0, string("myData:") + set });

  // get the a page set
  auto pageSet = std::dynamic_pointer_cast<PDBRandomAccessPageSet>(storage->getPageSet({0, string("myData:") + set }));
  pageSet->repinAll();

  // grab all the vectors
  std:shared_ptr<std::vector<Handle<Vector<Handle<TRABlock>>>>> inputVectors = std::make_shared<std::vector<Handle<Vector<Handle<TRABlock>>>>>();
  for(int i = 0; i < pageSet->getNumPages(); ++i) {

    // get the vector from the page
    auto vec = ((Record<Vector<Handle<TRABlock>>> *) (*pageSet)[i]->getBytes())->getRootObject();
    (*inputVectors).push_back(vec);
  }

  std::shared_ptr<std::vector<PDBCommunicatorPtr>> comIN;
  std::shared_ptr<std::vector<PDBCommunicatorPtr>> comOUT;

  std::function<void(int32_t tid,
                     void *data,
                     std::vector<std::vector<int32_t>> &records_to_join,
                     std::vector<emitter_row_t> &to_join,
                     std::vector<int32_t> &joined)> update_to_join;

  // figure out what records
  pdb::Map<TRABlockMeta, int32_t> *records;
  if(set == "A") {

    // set the record structure for this set
    records = &(*plan->records0);

    // set the update function for when the tid arrives
    update_to_join = [](int32_t tid,
                        void *data,
                        std::vector<std::vector<int32_t>> &records_to_join,
                        std::vector<emitter_row_t> &to_join,
                        std::vector<int32_t> &joined) {
      // go through all join records with this tid
      for(auto j : records_to_join[tid]) {

        to_join[j].a = data;
        if(to_join[j].b != nullptr && to_join[j].c != nullptr){
          joined.push_back(j);
        }
      }
    };

    // set the communicators
    comIN = s->aJoinSideCommunicatorsIn;
    comOUT = s->aJoinSideCommunicatorsOut;
  }
  else if(set == "B") {

    // set the record structure for this set
    records = &(*plan->records1);

    // set the update function for when the tid arrives
    update_to_join = [](int32_t tid,
                        void *data,
                        std::vector<std::vector<int32_t>> &records_to_join,
                        std::vector<emitter_row_t> &to_join,
                        std::vector<int32_t> &joined) {
      // go through all join records with this tid
      for(auto j : records_to_join[tid]) {

        to_join[j].b = data;
        if(to_join[j].a != nullptr && to_join[j].c != nullptr){
          joined.push_back(j);
        }
      }
    };

    // set the communicators
    comIN = s->bJoinSideCommunicatorsIn;
    comOUT = s->bJoinSideCommunicatorsOut;
  }
  else if(set == "C") {

    // set the record structure for this set
    records = &(*plan->records2);

    // set the update function for when the tid arrives
    update_to_join = [](int32_t tid,
                        void *data,
                        std::vector<std::vector<int32_t>> &records_to_join,
                        std::vector<emitter_row_t> &to_join,
                        std::vector<int32_t> &joined) {

      // go through all join records with this tid
      for(auto j : records_to_join[tid]) {

        to_join[j].c = data;
        if(to_join[j].a != nullptr && to_join[j].b != nullptr){
          joined.push_back(j);
        }
      }
    };

    // set the communicators
    comIN = s->cJoinSideCommunicatorsIn;
    comOUT = s->cJoinSideCommunicatorsOut;
  }

  // senders A
  for(int32_t n = 0; n < job->numberOfNodes; ++n) {

    // get a worker from the server
    PDBWorkerPtr worker = storage->getWorker();

    // make the work
    PDBWorkPtr myWork = std::make_shared<pdb::GenericWork>([&job, &counter, &s,
                                                                       &setIdx, n, &plan,
                                                                       records,
                                                                       inputVectors,
                                                                       &records_to_join,
                                                                       update_to_join,
                                                                       comOUT,
                                                                       &to_join, &joined,
                                                                       &m, &cv](const PDBBuzzerPtr& callerBuzzer) {

      auto num_nodes = job->numberOfNodes;
      auto _idx = TRABlockMeta(0, 0);

      meta_t _meta{};
      std::string error;

      if (job->thisNode != n) {

        std::cout << "Records has : " << records->size() << '\n';

        // go through all the records for set A we have on this node
        auto &seq = setIdx->sequential;
        for(auto t : seq) {

          // unpack this
          auto [rowID, colID, page, idx] = t;

          // get the tid for this record
          _idx.indices[0] = rowID;
          _idx.indices[1] = colID;
          //std::cout << "Checking - record0 | " << rowID << " - " << colID << '\n';
          if((*records).count(_idx)) {

            // get the tensor id
            auto tid = (*records)[_idx];

            // check if we need to send this
            if((*(*plan).record_mapping)[tid * num_nodes + n]) {

              // set the meta
              _meta.rowID = rowID;
              _meta.colID = colID;
              _meta.numRows = colID;
              _meta.numCols = colID;
              _meta.hasMore = true;

              // send the meta
              auto com = (*comOUT)[n];

              //std::cout << "Sending - records | " << rowID << " - " << colID << '\n';
              com->sendBytes(&_meta, sizeof(_meta), error);

              // get the block
              auto data = (*(*inputVectors)[page])[idx]->data->data->c_ptr();

              // send the data
              com->sendBytes(data,sizeof(float) * _meta.numRows * _meta.numCols,error);
            }
          }
        }

        // set the meta
        _meta.hasMore = false;

        // send the meta
        (*comOUT)[n]->sendBytes(&_meta, sizeof(_meta), error);
      }
      else {


        // go through all the records for set A we have on this node
        auto &seq = setIdx->sequential;
        for(auto t : seq) {

          // unpack this
          auto [rowID, colID, page, idx] = t;

          // get the tid for this record
          _idx.indices[0] = rowID;
          _idx.indices[1] = colID;

          //std::cout << "Checking - record0 | " << rowID << " - " << colID << '\n';
          if ((*records).count(_idx)) {

            // get the block
            auto data = (*(*inputVectors)[page])[idx]->data->data->c_ptr();

            // get the tensor id
            auto tid = (*records)[_idx];

            // lock to notify all the joins
            std::unique_lock<std::mutex> lck(m);

            // update the join
            update_to_join(tid, data, records_to_join, to_join, joined);

            // notify that we have something
            cv.notify_all();
          }
        }
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
    PDBWorkPtr myWork = std::make_shared<pdb::GenericWork>([&job, &counter, s, comIN,
                                                                       n, &plan, &m, records,
                                                                       &cv, &records_to_join,
                                                                       &to_join, update_to_join,
                                                                       &joined](const PDBBuzzerPtr& callerBuzzer) {

      auto _idx = TRABlockMeta(0, 0);
      meta_t _meta{};
      std::string error;

      if (job->thisNode != n) {

        while(true) {

          // receive the meta
          (*comIN)[n]->receiveBytes(&_meta, error);

          //
          std::cout << "received - records | " << _meta.rowID << " colID " << _meta.colID << '\n';

          // check if we are done receiving
          if(!_meta.hasMore){
            break;
          }

          // allocate the memory
          auto data = (float*) malloc(_meta.numRows * _meta.numCols * sizeof(float));

          // get the data
          (*comIN)[n]->receiveBytes(data, error);

          // set the index
          _idx.indices[0] = _meta.numRows;
          _idx.indices[1] = _meta.numCols;

          // get the tid
          auto tid = (*records)[_idx];

          // lock to notify all the joins
          std::unique_lock<std::mutex> lck(m);

          // update the join
          update_to_join(tid, data, records_to_join, to_join, joined);

          // notify that we have something
          cv.notify_all();
        }
      }

      // signal that the run was successful
      callerBuzzer->buzz(PDBAlarm::WorkAllDone, counter);
    });

    // run the work
    worker->execute(myWork, tempBuzzer);
  }
}
