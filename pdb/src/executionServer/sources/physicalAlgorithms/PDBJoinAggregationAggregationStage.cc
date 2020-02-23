#include "../../headers/physicalAlgorithms/PDBJoinAggregationAggregationStage.h"
#include "PDBJoinAggregationState.h"
#include "GenericWork.h"
#include "ExJob.h"
#include "ComputePlan.h"
#include "AtomicComputationClasses.h"
#include "PreaggregationPageProcessor.h"

pdb::PDBJoinAggregationAggregationStage::PDBJoinAggregationAggregationStage(const pdb::PDBSinkPageSetSpec &sink,
                                                                            const pdb::Vector<pdb::PDBSourceSpec> &sources,
                                                                            const pdb::String &final_tuple_set,
                                                                            const pdb::Vector<pdb::Handle<
                                                                                PDBSourcePageSetSpec>> &secondary_sources,
                                                                            const pdb::Vector<pdb::PDBSetObject> &sets_to_materialize,
                                                                            const pdb::String &join_tuple_set)
    : PDBPhysicalAlgorithmStage(sink, sources, final_tuple_set, secondary_sources, sets_to_materialize),
      joinTupleSet(join_tuple_set) {}

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

  // set the left and right mappings
  joinArguments->leftTIDToRecordMapping = &s->leftTIDToRecordMapping;
  joinArguments->rightTIDToRecordMapping = &s->rightTIDToRecordMapping;

  // set the plan page
  joinArguments->planPage = s->planPage;

  /// 15.1 Init the preaggregation queues

  s->pageQueues = std::make_shared<std::vector<PDBPageQueuePtr>>();
  for (int i = 0; i < job->numberOfProcessingThreads;
       ++i) { s->pageQueues->emplace_back(std::make_shared<PDBPageQueue>()); }

  // fill uo the vector for each thread
  std::map<ComputeInfoType, ComputeInfoPtr> params;
  s->preaggregationPipelines = std::make_shared<std::vector<PipelinePtr>>();
  for (uint64_t pipelineIndex = 0; pipelineIndex < job->numberOfProcessingThreads; ++pipelineIndex) {

    /// 15.2. Figure out the parameters of the pipeline

    // initialize the parameters
    params = {{ComputeInfoType::PAGE_PROCESSOR,
               std::make_shared<PreaggregationPageProcessor>(1, // we use one since this pipeline is completely local.
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
                                       1, // we use one since this pipeline is completely local.
                                       job->numberOfProcessingThreads,
                                       pipelineIndex);

    s->preaggregationPipelines->push_back(pipeline);
  }

  /// 8. Create the aggregation pipeline

  // we are putting the pages from the queues here
  s->preaggPageSet =
      std::make_shared<PDBFeedingPageSet>(job->numberOfProcessingThreads, job->numberOfProcessingThreads);

  // get the sink page set
  auto sinkPageSet =
      storage->createAnonymousPageSet(std::make_pair(sink.pageSetIdentifier.first, sink.pageSetIdentifier.second));

  // did we manage to get a sink page set? if not the setup failed
  if (sinkPageSet == nullptr) {
    return false;
  }

  s->aggregationPipelines = std::make_shared<std::vector<PipelinePtr>>();
  for (uint64_t workerID = 0; workerID < job->numberOfProcessingThreads; ++workerID) {

    // build the aggregation pipeline
    auto aggPipeline = plan.buildAggregationPipeline(finalTupleSet, s->preaggPageSet, sinkPageSet, workerID);

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

  std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();

  // run the aggregation pipelines
  atomic_int preaggCnt;
  preaggCnt = 0;

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

  for (int i = 0; i < s->preaggregationPipelines->size(); ++i) {

    // get a worker from the server
    PDBWorkerPtr worker = storage->getWorker();

    // make the work
    PDBWorkPtr
        myWork = std::make_shared<pdb::GenericWork>([&preaggCnt, &success, i, s](const PDBBuzzerPtr &callerBuzzer) {

      try {

        // run the pipeline
        (*s->preaggregationPipelines)[i]->run();
      }
      catch (std::exception &e) {

        // log the error
        s->logger->error(e.what());

        // we failed mark that we have
        success = false;
      }

      // signal that the run was successful
      callerBuzzer->buzz(PDBAlarm::WorkAllDone, preaggCnt);
    });

    // run the work
    worker->execute(myWork, tempBuzzer);
  }

  // make the threads that feed into the feed page set
  counter = 0;
  for (int i = 0; i < s->preaggregationPipelines->size(); ++i) {

    // get a worker from the server
    PDBWorkerPtr worker = storage->getWorker();

    // make the work
    PDBWorkPtr
        myWork = std::make_shared<pdb::GenericWork>([&counter, &success, i, &s](const PDBBuzzerPtr &callerBuzzer) {

      // do this until we get a null
      PDBPageHandle tmp;
      while (true) {

        // get the page from the queue
        (*s->pageQueues)[i]->wait_dequeue(tmp);

        // get out of loop
        if (tmp == nullptr) {
          s->preaggPageSet->finishFeeding();
          break;
        }

        // feed the page into the page set
        s->preaggPageSet->feedPage(tmp);
      }

      // signal that the run was successful
      callerBuzzer->buzz(PDBAlarm::WorkAllDone, counter);
    });

    // run the work
    worker->execute(myWork, tempBuzzer);
  }

  // run the aggregation pipelines
  for (int i = 0; i < s->aggregationPipelines->size(); ++i) {

    // get a worker from the server
    PDBWorkerPtr worker = storage->getWorker();

    // make the work
    PDBWorkPtr
        myWork = std::make_shared<pdb::GenericWork>([&counter, &success, i, &s](const PDBBuzzerPtr &callerBuzzer) {

      try {

        // run the pipeline
        (*s->aggregationPipelines)[i]->run();
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

  // wait for the preaggregation to finish
  while (preaggCnt < s->preaggregationPipelines->size()) {
    tempBuzzer->wait();
  }

  // insert to the page queues
  for (const auto &q : *s->pageQueues) {
    q->enqueue(nullptr);
  }

  // wait until the feeding is finished and the aggregation pipelines are finished
  while (counter < s->preaggregationPipelines->size() + s->aggregationPipelines->size()) {
    tempBuzzer->wait();
  }


  std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
  std::cout << "AggregationStage run for " << std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count() << "[ns]" << '\n';
  begin = end;

  // should we materialize this to a set?
  for (int j = 0; j < setsToMaterialize.size(); ++j) {

    // get the page set
    auto sinkPageSet = storage->getPageSet(std::make_pair(sink.pageSetIdentifier.first, sink.pageSetIdentifier.second));

    // if the thing does not exist finish!
    if (sinkPageSet == nullptr) {
      success = false;
      break;
    }

    // materialize the page set
    sinkPageSet->resetPageSet();
    success = storage->materializePageSet(sinkPageSet,
                                          std::make_pair<std::string, std::string>(setsToMaterialize[j].database,
                                                                                   setsToMaterialize[j].set))
        && success;
  }

  end = std::chrono::steady_clock::now();
  std::cout << "Materialization run for " << std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count() << "[ns]" << '\n';
  return success;
}

void pdb::PDBJoinAggregationAggregationStage::cleanup(const pdb::PDBPhysicalAlgorithmStatePtr &state) {}
