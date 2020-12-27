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
  std::vector<std::vector<int32_t>> records_to_join(plan->records0->size() + plan->records1->size() + plan->records2->size());

  // fill the records
  auto &jr = *plan->joinedRecords;
  for(int32_t i = 0; i < jr.size(); ++i) {

    // store what we need to
    records_to_join[jr[i].first].push_back(i);
    records_to_join[jr[i].second].push_back(i);
    records_to_join[jr[i].third].push_back(i);
  }

  // map each join group to an aggregation group
  auto &agg = *plan->aggRecords;
  std::vector<int32_t> aggInto(plan->joinedRecords->size());
  for(auto a = 0; a < agg.size(); ++a) {
    for(auto j = 0; j < (*agg[a]).size(); ++j) {
      auto tj = (*agg[a])[j];
      aggInto[tj] = a;
    }
  }

  // the joined records
  std::vector<int32_t> joined;

  atomic_bool success;
  success = true;

  // create the buzzer
  atomic_int rippleCounter;
  rippleCounter = 0;
  PDBBuzzerPtr rippleBuzzer = make_shared<PDBBuzzer>([&](PDBAlarm myAlarm, atomic_int &cnt) {

    // did we fail?
    if(myAlarm == PDBAlarm::GenericError) {
      success = false;
    }

    // increment the count
    cnt++;
  });

  // run the senders for each set
  setup_set_comm("A", m, cv, rippleCounter, joined, records_to_join, to_join, plan, job, storage, state, rippleBuzzer);
  setup_set_comm("B", m, cv, rippleCounter, joined, records_to_join, to_join, plan, job, storage, state, rippleBuzzer);
  setup_set_comm("C", m, cv, rippleCounter, joined, records_to_join, to_join, plan, job, storage, state, rippleBuzzer);

  // the buzzer for the counter
  atomic_int joinCounter;
  joinCounter = 0;
  PDBBuzzerPtr joinBuzzer = make_shared<PDBBuzzer>([&](PDBAlarm myAlarm, atomic_int &cnt) {

    // did we fail?
    if(myAlarm == PDBAlarm::GenericError) {
      success = false;
    }

    // increment the count
    cnt++;
  });

  // cache for the sums
  std::mutex cache_m;
  std::condition_variable cache_cv;
  std::map<tuple<uint64_t, uint64_t>, void*> sum_cache;

  // the aggregation sync
  std::mutex agg_m;
  std::condition_variable agg_cv;

  //                blob,   join_tid,  rowID,   colID,   rowID,   colID
  std::vector<tuple<float*, int32_t, int32_t, int32_t, int32_t, int32_t>> joinToAgg;

  for(int32_t n = 0; n < job->numberOfProcessingThreads; ++n) {

    // get a worker from the server
    PDBWorkerPtr worker = storage->getWorker();

    // make the work
    PDBWorkPtr myWork = std::make_shared<pdb::GenericWork>([&m, &cv, &job, &joined, &joinCounter,
                                                                       &rippleCounter, &to_join, &sum_cache,
                                                                       &cache_m, &cache_cv, &agg_m, &agg_cv, &joinToAgg](const PDBBuzzerPtr& callerBuzzer) {

      while(true) {

        int32_t to_process;
        {

          // wait until we have something here
          std::unique_lock<std::mutex> lk(m);

          // wait to get some joined records
          cv.wait(lk, [&]{ return !joined.empty() || rippleCounter == job->numberOfNodes * 2 * 3; });

          // check if the ripple is done
          if(rippleCounter == job->numberOfNodes * 2 * 3 && joined.empty()) {
            break;
          }

          // get the info for processing
          to_process = joined.back(); joined.pop_back();
        }

        // grab what we need to do
        auto &info = to_join[to_process];

        assert(info.a_numRows == info.b_numRows);
        assert(info.a_numCols == info.b_numCols);
        assert(info.b_numCols == info.c_numRows);

        auto *a = (float*) info.a;
        auto *b = (float*) info.b;
        auto *c = (float*) info.c;

        // check the sum
        float* sum = nullptr;
        {
          // wait until we have something here
          std::unique_lock<std::mutex> lk(cache_m);

          // check if we have it in store
          tuple<uint64_t, uint64_t> cache_key = { (uint64_t) a, (uint64_t) b };

          // wait if it is currently being processed
          decltype(sum_cache.begin()) it;
          cache_cv.wait(lk, [&]{

            // try to find it
            it = sum_cache.find(cache_key);
            return (it != sum_cache.end() && it->second != nullptr) ||
                   (it == sum_cache.end());
          });

          // mark that we are currently processing this sum
          if(it == sum_cache.end()) {
            sum_cache[cache_key] = nullptr;
          }
        }

        // ok we don't have it we need to create it
        if(sum == nullptr) {

          // allocate the memory and perform the sum
          sum = (float*) mkl_malloc(info.a_numRows * info.a_numCols * sizeof(float), 32);
          for(int32_t i = 0; i < info.a_numRows * info.a_numCols; ++i) {
            sum[i] = a[i] + b[i];
          }

          // notify other threads that we are honky dory
          {
            // wait until we have something here
            std::unique_lock<std::mutex> lk(cache_m);

            // store it and notify
            sum_cache[{ (uint64_t) a, (uint64_t) b }] = (void*) sum;
            cache_cv.notify_all();
          }
        }

        // the output of the multiplication
        auto mult = (float*) mkl_malloc(info.a_numRows * info.a_numCols * sizeof(float), 32);

        // get the sizes
        uint32_t I = info.a_numRows;
        uint32_t J = info.c_numCols;
        uint32_t K = info.a_numCols;

        // did the multiply
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, I, J, K, 1.0f, sum, K, c, J, 0.0f, mult, J);

        // store it for aggregation
        {
          // wait until we have something here
          std::unique_lock<std::mutex> lk(agg_m);

          // store these to aggregate
          joinToAgg.emplace_back(mult, to_process, info.rowID, info.colID, I, J);
          agg_cv.notify_all();

          // std::cout << "Multiply rowID : " << info.rowID << ", " << info.colID << "\n" << std::flush;
        }
      }

      //std::cout << "Mult done finished\n" << std::flush;
      // signal that the run was successful
      callerBuzzer->buzz(PDBAlarm::WorkAllDone, joinCounter);
    });

    // run the work
    worker->execute(myWork, joinBuzzer);
  }


  // create the buzzer
  atomic_int aggCounter;
  aggCounter = 0;
  PDBBuzzerPtr aggBuzzer = make_shared<PDBBuzzer>([&](PDBAlarm myAlarm, atomic_int &cnt) {

    // did we fail?
    if(myAlarm == PDBAlarm::GenericError) {
      success = false;
    }

    // increment the count
    cnt++;
  });

  // we are aggregating here                      data,   rowID,   colID, numRows, numCols
  std::vector<std::tuple<std::mutex, std::tuple<float*, int32_t, int32_t, int32_t, int32_t>  >> aggregated(plan->aggRecords->size());
  for(auto &t : aggregated) {
    get<0>(get<1>(t)) = nullptr;
  }

  // start the aggregation threads
  for(int32_t n = 0; n < job->numberOfProcessingThreads; ++n) {

    // get a worker from the server
    PDBWorkerPtr worker = storage->getWorker();

    // make the work
    PDBWorkPtr myWork = std::make_shared<pdb::GenericWork>([&aggCounter, &agg_m, &agg_cv, &joinCounter, &job,
                                                                       &joinToAgg, &aggregated, &aggInto](const PDBBuzzerPtr& callerBuzzer) {


      while(true) {

        tuple<float*, int32_t, int32_t, int32_t, int32_t, int32_t> tuple;
        {
          // wait until we have something here
          std::unique_lock<std::mutex> lk(agg_m);

          // wait to get some joined records
          agg_cv.wait(lk, [&]{ return !joinToAgg.empty() || joinCounter == job->numberOfProcessingThreads; });

          // see if we have finished
          if(joinToAgg.empty() && joinCounter == job->numberOfProcessingThreads) {
            break;
          }

          // the get join tuple to aggregate
          tuple = joinToAgg.back(); joinToAgg.pop_back();
        }

        // get the input
        auto [in, j, rowID, colID, numRows, numCols] = tuple;
        {
          // lock the aggregated struct
          auto a = aggInto[j];
          std::unique_lock<std::mutex> lck(std::get<0>(aggregated[a]));

          // if we don't have anything we cool
          auto t = std::get<1>(aggregated[a]);
          if(std::get<0>(t) == nullptr) {
            std::get<1>(aggregated[a]) = {in, rowID, colID, numRows, numCols};
          }
          else {

            // sum it all up
            for(int32_t i = 0; i < numRows * numCols; ++i) {
              std::get<0>(t)[i] += in[i];
            }
          }
        }

      }

      // signal that the run was successful
      callerBuzzer->buzz(PDBAlarm::WorkAllDone, aggCounter);
    });

    // run the work
    worker->execute(myWork, aggBuzzer);
  }

  // wait until all the preaggregationPipelines have completed
  while (rippleCounter < job->numberOfNodes * 2 * 3) {
    rippleBuzzer->wait();
  }
  std::cout << "Ripple finished\n";

  // mark that the ripple is done
  cv.notify_all();

  // wait for the join threads to finish
  while (joinCounter < job->numberOfProcessingThreads) {
    joinBuzzer->wait();
  }

  // mark that the join threads are done
  agg_cv.notify_all();

  //
  while (aggCounter < job->numberOfProcessingThreads) {
    aggBuzzer->wait();
  }

  // make a random access page set
  auto outTmp = std::dynamic_pointer_cast<pdb::PDBRandomAccessPageSet>(storage->createRandomAccessPageSet({0, "outTmp"}));

  // create the buzzer
  atomic_int matCounter;
  matCounter = 0;
  PDBBuzzerPtr matBuzzer = make_shared<PDBBuzzer>([&](PDBAlarm myAlarm, atomic_int &cnt) {

    // did we fail?
    if(myAlarm == PDBAlarm::GenericError) {
      success = false;
    }

    // increment the count
    cnt++;
  });


  // start the aggregation threads
  atomic_int32_t toProcessGroup = 0;
  std::mutex sync;

  auto myMgr = storage->getFunctionalityPtr<PDBBufferManagerInterface>();
  for(int32_t n = 0; n < job->numberOfProcessingThreads; ++n) {

    // get a worker from the server
    PDBWorkerPtr worker = storage->getWorker();

    // make the work
    PDBWorkPtr myWork = std::make_shared<pdb::GenericWork>([&outTmp, &matCounter, &myMgr, &aggregated, &toProcessGroup, &sync](const PDBBuzzerPtr& callerBuzzer) {

      // get a new page
      auto currentPage = myMgr->getPage();
      makeObjectAllocatorBlock(currentPage->getBytes(), currentPage->getSize(), true);

      // is there stuff on the page
      bool stuffOnPage = false;

      // make the vector we write to
      int32_t curGroup = -1;
      Handle<Vector<Handle<pdb::TRABlock>>> writeMe = makeObject<Vector<Handle<pdb::TRABlock>>>();
      while (true) {

        // grab a new group to process if needed
        if(curGroup == -1) {

          // grab the group we want to process
          curGroup = toProcessGroup++;
        }

        // break if we are done
        if(curGroup >= aggregated.size()) {
          break;
        }

        // get the info about the group
        auto [data, rowID, colID, numRows, numCols] = std::get<1>(aggregated[curGroup]);

        if(data == nullptr) {
          curGroup = -1;
          continue;
        }

        sync.lock();
        std::cout << "Materializing " << rowID << " " << colID << " " << numRows << " " << numCols << " " << curGroup << '\n';
        sync.unlock();

        try {

          // allocate a matrix
          Handle<TRABlock> myInt = makeObject<TRABlock>(rowID, colID, numRows, numCols);

          // store it
          writeMe->push_back(myInt);

          // copy the data
          memmove(myInt->data->data->c_ptr(), data, sizeof(float) * numRows * numCols);

          // mark that there is stuff on the page
          stuffOnPage = true;

          // set that we need to fetch a new aggregation group
          curGroup = -1;

        } catch (pdb::NotEnoughSpace &n) {

          // make this the root object
          getRecord(writeMe);

          // store the page
          outTmp->pushPage(currentPage);

          // grab a new page
          stuffOnPage = false;
          currentPage = myMgr->getPage();
          makeObjectAllocatorBlock(currentPage->getBytes(), currentPage->getSize(), true);

          // make a new vector!
          writeMe = makeObject<Vector<Handle<pdb::TRABlock>>>();
        }
      }

      // is there some stuff on the page
      if(stuffOnPage) {

        // make this the root object
        getRecord(writeMe);

        // insert into page into the page set
        outTmp->pushPage(currentPage);
      }

      // invalidate the block
      makeObjectAllocatorBlock(1024, true);

      // signal that the run was successful
      callerBuzzer->buzz(PDBAlarm::WorkAllDone, matCounter);
    });

    // run the work
    worker->execute(myWork, matBuzzer);
  }

  while (matCounter < job->numberOfProcessingThreads) {
    matBuzzer->wait();
  }

  // materialize the page set
  outTmp->resetPageSet();
  storage->materializePageSet(outTmp, std::make_pair<std::string, std::string>("myData", "D"));

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
                     int32_t node,
                     int32_t rowID,
                     int32_t colID,
                     int32_t num_rows,
                     int32_t num_cols,
                     pdb::Vector<int32_t> &join_group_mapping,
                     void *data,
                     std::vector<std::vector<int32_t>> &records_to_join,
                     std::vector<emitter_row_t> &to_join,
                     std::vector<int32_t> &joined)> update_to_join;

  // figure out what records
  pdb::Map<TRABlockMeta, int32_t> *records;
  if(set == "A") {

    // set the record structure for this set
    records = &(*plan->records0);

//    std::cout << "A : \n";
//    auto _idx = TRABlockMeta(0, 0);
//    std::cout << "(0, 0) " << (*records)[_idx] << '\n';
//
//    _idx = TRABlockMeta(1, 0);
//    std::cout << "(1, 0) " << (*records)[_idx] << '\n';
//
//    _idx = TRABlockMeta(0, 1);
//    std::cout << "(0, 1) " << (*records)[_idx] << '\n';
//
//    _idx = TRABlockMeta(1, 1);
//    std::cout << "(1, 1) " << (*records)[_idx] << '\n';

    // set the update function for when the tid arrives
    update_to_join = [](int32_t tid,
                        int32_t node,
                        int32_t rowID,
                        int32_t colID,
                        int32_t num_rows,
                        int32_t num_cols,
                        pdb::Vector<int32_t> &join_group_mapping,
                        void *data,
                        std::vector<std::vector<int32_t>> &records_to_join,
                        std::vector<emitter_row_t> &to_join,
                        std::vector<int32_t> &joined) {

      assert(num_rows != 0);
      assert(num_cols != 0);

      // go through all join records with this tid
      for(auto j : records_to_join[tid]) {

        // is this join group on our node
        if(join_group_mapping[j] != node) {
          continue;
        }

        assert(data != nullptr);
        //std::cout << "Adding A : " << tid << " for " << j << '\n';

        // ok this is on our node update
        to_join[j].a = data;
        to_join[j].a_numRows = num_rows;
        to_join[j].a_numCols = num_cols;
        to_join[j].rowID = rowID;

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
//
//    std::cout << "B : \n";
//    auto _idx = TRABlockMeta(0, 0);
//    std::cout << "(0, 0) " << (*records)[_idx] << '\n';
//
//    _idx = TRABlockMeta(1, 0);
//    std::cout << "(1, 0) " << (*records)[_idx] << '\n';
//
//    _idx = TRABlockMeta(0, 1);
//    std::cout << "(0, 1) " << (*records)[_idx] << '\n';
//
//    _idx = TRABlockMeta(1, 1);
//    std::cout << "(1, 1) " << (*records)[_idx] << '\n';
//


    // set the update function for when the tid arrives
    update_to_join = [](int32_t tid,
                        int32_t node,
                        int32_t rowID,
                        int32_t colID,
                        int32_t num_rows,
                        int32_t num_cols,
                        pdb::Vector<int32_t> &join_group_mapping,
                        void *data,
                        std::vector<std::vector<int32_t>> &records_to_join,
                        std::vector<emitter_row_t> &to_join,
                        std::vector<int32_t> &joined) {

      // go through all join records with this tid
      for(auto j : records_to_join[tid]) {

        // is this join group on our node
        if(join_group_mapping[j] != node) {
          continue;
        }

        //std::cout << "Adding B : " << tid << " for " << j << '\n';
        to_join[j].b = data;
        to_join[j].b_numRows = num_rows;
        to_join[j].b_numCols = num_cols;

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

//    std::cout << "C : \n";
//    auto _idx = TRABlockMeta(0, 0);
//    std::cout << "(0, 0) " << (*records)[_idx] << '\n';
//
//    _idx = TRABlockMeta(1, 0);
//    std::cout << "(1, 0) " << (*records)[_idx] << '\n';
//
//    _idx = TRABlockMeta(0, 1);
//    std::cout << "(0, 1) " << (*records)[_idx] << '\n';
//
//    _idx = TRABlockMeta(1, 1);
//    std::cout << "(1, 1) " << (*records)[_idx] << '\n';

    // set the update function for when the tid arrives
    update_to_join = [](int32_t tid,
                        int32_t node,
                        int32_t rowID,
                        int32_t colID,
                        int32_t num_rows,
                        int32_t num_cols,
                        pdb::Vector<int32_t> &join_group_mapping,
                        void *data,
                        std::vector<std::vector<int32_t>> &records_to_join,
                        std::vector<emitter_row_t> &to_join,
                        std::vector<int32_t> &joined) {

      // go through all join records with this tid
      for(auto j : records_to_join[tid]) {

        // is this join group on our node
        if(join_group_mapping[j] != node) {
          continue;
        }

        //std::cout << "Adding C : " << tid << " for " << j << '\n';
        to_join[j].c = data;
        to_join[j].c_numRows = num_rows;
        to_join[j].c_numCols = num_cols;
        to_join[j].colID = colID;

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
    PDBWorkPtr myWork = std::make_shared<pdb::GenericWork>([&job, &counter, &s, set,
                                                                       setIdx, n, &plan,
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

              // get the block
              auto &block_data = *((*(*inputVectors)[page])[idx]->data);

              // set the meta
              _meta.rowID = rowID;
              _meta.colID = colID;
              _meta.numRows = block_data.dim0;
              _meta.numCols = block_data.dim1;
              _meta.hasMore = true;

              // send the meta
              auto com = (*comOUT)[n];

              //std::cout << "Sending - records | " << rowID << " - " << colID << '\n';
              com->sendBytes(&_meta, sizeof(_meta), error);

              // get the block
              auto data = block_data.data->c_ptr();

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
            auto &block_data = *((*(*inputVectors)[page])[idx]->data);

            // get the block
            auto data = block_data.data->c_ptr();

            // get the tensor id
            auto tid = (*records)[_idx];

            //
            _meta.numRows = block_data.dim0;
            _meta.numCols = block_data.dim1;

            // lock to notify all the joins
            std::unique_lock<std::mutex> lck(m);

            // update the join
            //std::cout << "self for set " << set << " | rowID " << rowID << " colID " << colID << " tid : " << tid << '\n' << std::flush;
            update_to_join(tid, job->thisNode, rowID, colID, _meta.numRows, _meta.numCols, *plan->join_group_mapping, data, records_to_join, to_join, joined);

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
    PDBWorkPtr myWork = std::make_shared<pdb::GenericWork>([&job, &counter, s, comIN, set,
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

          // check if we are done receiving
          if(!_meta.hasMore){
            break;
          }

          // allocate the memory
          auto data = (float*) mkl_malloc(_meta.numRows * _meta.numCols * sizeof(float), 32);

          // get the data
          (*comIN)[n]->receiveBytes(data, error);

          // set the index
          _idx.indices[0] = _meta.rowID;
          _idx.indices[1] = _meta.colID;

          // get the tid
          auto tid = (*records)[_idx];
          //std::cout << "received for set " << set << " | rowID : " << _meta.rowID << " colID : " << _meta.colID << " tid : " << tid << '\n' << std::flush;

          // lock to notify all the joins
          std::unique_lock<std::mutex> lck(m);

          // update the join
          update_to_join(tid, job->thisNode, _meta.rowID, _meta.colID, _meta.numRows, _meta.numCols, *plan->join_group_mapping, data, records_to_join, to_join, joined);

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
