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


  // init the vector for connections for A
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

  // init the vector connections for A
  connectionID->taskID = PDBJoinAggregationState::A_TASK;
  s->aJoinSideCommunicatorsIn = std::make_shared<std::vector<PDBCommunicatorPtr>>();
  for (int n = 0; n < job->numberOfNodes; n++) {

    // set the node id
    connectionID->nodeID = n;

    // skip com to itself
    if(job->thisNode == n) {
      s->aJoinSideCommunicatorsIn->push_back(nullptr);
      continue;
    }

    // wait for the connection
    s->aJoinSideCommunicatorsIn->push_back(myMgr->waitForConnection(connectionID));

    // check if the socket is open
    if (s->aJoinSideCommunicatorsIn->back()->isSocketClosed()) {

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

  // get the a page set
  auto aPageSet = std::dynamic_pointer_cast<PDBRandomAccessPageSet>(storage->getPageSet({0, "myData:A"}));

  // grab all the vectors
  std::vector<Handle<Vector<Handle<TRABlock>>>> aInputVectors;
  for(int i = 0; i < aPageSet->getNumPages(); ++i) {

    // get the vector from the page
    auto vec = ((Record<Vector<Handle<TRABlock>>> *) (*aPageSet)[i]->getBytes())->getRootObject();
    aInputVectors.push_back(vec);
  }

  // senders A
  for(int n = 0; n < job->numberOfNodes; ++n) {

    // get a worker from the server
    PDBWorkerPtr worker = storage->getWorker();

    // make the work
    PDBWorkPtr myWork = std::make_shared<pdb::GenericWork>([&job, &counter, &s, &setAIdx, n, &plan, &aInputVectors](const PDBBuzzerPtr& callerBuzzer) {

      auto num_nodes = job->numberOfNodes;
      auto _idx = TRABlockMeta(0, 0);

      meta_t _meta{};
      std::string error;

      if (job->thisNode != n) {

        std::cout << "Records0 has : " << plan->records0->size() << '\n';
        // go through all the records for set A we have on this node
        auto &seq = setAIdx->sequential;

        for(auto t : seq) {

          // unpack this
          auto [rowID, colID, page, idx] = t;

          // get the tid for this record
          _idx.indices[0] = rowID;
          _idx.indices[1] = colID;
          std::cout << "Checking - record0 | " << rowID << " - " << colID << '\n';
          if((*plan->records0).count(_idx)) {

            // get the tensor id
            auto tid = (*plan->records0)[_idx];

            // check if we need to send this
            if((*(*plan).record_mapping)[tid * num_nodes + n]) {

              // set the meta
              _meta.rowID = rowID;
              _meta.colID = colID;
              _meta.numRows = colID;
              _meta.numCols = colID;
              _meta.hasMore = true;

              // send the meta
              auto com = (*s->aJoinSideCommunicatorsOut)[n];

              std::cout << "Sending - records0 | " << rowID << " - " << colID << '\n';
              com->sendBytes(&_meta, sizeof(_meta), error);

              // get the block
              auto data = (*aInputVectors[page])[idx]->data->data->c_ptr();

              // send the data
              com->sendBytes(data,sizeof(float) * _meta.numRows * _meta.numCols,error);
            }
          }
        }

        // set the meta
        _meta.hasMore = false;

        // send the meta
        (*s->aJoinSideCommunicatorsOut)[n]->sendBytes(&_meta, sizeof(_meta), error);
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

      meta_t _meta{};
      std::string error;

      if (job->thisNode != n) {

        while(true) {

          // receive the meta
          (*s->aJoinSideCommunicatorsIn)[n]->receiveBytes(&_meta, error);

          //
          std::cout << "received - records0 | " << _meta.rowID << " colID " << _meta.colID << '\n';

          // check if we are done receiving
          if(!_meta.hasMore){
            break;
          }

          // just skip for now
          (*s->aJoinSideCommunicatorsIn)[n]->skipBytes(error);
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
