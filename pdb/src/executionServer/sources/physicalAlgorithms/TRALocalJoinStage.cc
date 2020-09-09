#include <PipelineInterface.h>
#include <ComputePlan.h>
#include <physicalAlgorithms/TRALocalJoinState.h>
#include <processors/NullProcessor.h>
#include <AtomicComputationClasses.h>
#include <GenericWork.h>
#include "TRALocalJoinStage.h"
#include "TRALocalJoinEmitter.h"
#include "ExJob.h"

namespace pdb {

bool TRALocalJoinStage::setup(const Handle<pdb::ExJob> &job,
                              const PDBPhysicalAlgorithmStatePtr &state,
                              const std::shared_ptr<pdb::PDBStorageManagerBackend> &storage,
                              const std::string &error) {

  // cast the state
  auto s = dynamic_pointer_cast<TRALocalJoinState>(state);

  // init the plan
  ComputePlan plan(std::make_shared<LogicalPlan>(job->tcap, *job->computations));
  s->logicalPlan = plan.getPlan();

  // get the join comp
  auto joinAtomicComp =
      dynamic_pointer_cast<ApplyJoin>(s->logicalPlan->getComputations().getProducingAtomicComputation(firstTupleSet));

  // the the input page set
  s->inputPageSet = storage->createPageSetFromPDBSet(rhsDB, rhsSet, false);

  // the emmitter will put set pageser here
  s->rightPageSet = storage->createRandomAccessPageSet({0, "right_intermediate"});

  // get index of the left page set
  s->index = storage->getIndex({0, lhsPageSet});

  // get the lhs page set
  s->leftPageSet = std::dynamic_pointer_cast<pdb::PDBRandomAccessPageSet>(storage->getPageSet({0, lhsPageSet}));

  // make the output intermediate set
  s->outputIntermediate = storage->createAnonymousPageSet({0, "intermediate"});

  // make the output sink
  s->output = storage->createRandomAccessPageSet({0, sink});

  // create a new index for the output set
  s->outputIndex = storage->createIndex({0, sink});

  std::map<ComputeInfoType, ComputeInfoPtr> params;

  // the join arguments
  auto joinArguments = std::make_shared<JoinArguments>(JoinArgumentsInit{{joinAtomicComp->getRightInput().getSetName(),
                                                                          std::make_shared<JoinArg>(s->rightPageSet)}});

  // make the emitter
  s->emitter = std::make_shared<TRALocalJoinEmitter>(job->numberOfProcessingThreads,
                                                     s->inputPageSet,
                                                     s->leftPageSet,
                                                     s->rightPageSet,
                                                     lhsIndices,
                                                     rhsIndices,
                                                     s->index);

  // mark that this is the join aggregation algorithm
  joinArguments->isTRALocalJoin = true;
  joinArguments->emitter = s->emitter;

  /// 1.1 init the join pipelines

  // fill uo the vector for each thread
  s->joinPipelines = std::make_shared<std::vector<PipelinePtr>>();
  for (uint64_t pipelineIndex = 0; pipelineIndex < job->numberOfProcessingThreads; ++pipelineIndex) {

    /// 1.2. Figure out the parameters of the pipeline

    // initialize the parameters
    params = {{ComputeInfoType::PAGE_PROCESSOR, std::make_shared<NullProcessor>()},
              {ComputeInfoType::JOIN_ARGS, joinArguments},
              {ComputeInfoType::SHUFFLE_JOIN_ARG, std::make_shared<ShuffleJoinArg>(false)},
              {ComputeInfoType::SOURCE_SET_INFO, nullptr}};

    /// 1.3. Build the pipeline

    auto pipeline = plan.buildPipeline(firstTupleSet, /* this is the TupleSet the pipeline starts with */
                                       finalTupleSet,     /* this is the TupleSet the pipeline ends with */
                                       s->leftPageSet,
                                       s->outputIntermediate,
                                       params,
                                       job->thisNode,
                                       job->numberOfNodes, // we use one since this pipeline is completely local.
                                       job->numberOfProcessingThreads,
                                       pipelineIndex);

    s->joinPipelines->push_back(pipeline);
  }

  std::cout << "setup\n";
  return true;
}

bool TRALocalJoinStage::run(const Handle<pdb::ExJob> &job,
                            const PDBPhysicalAlgorithmStatePtr &state,
                            const std::shared_ptr<pdb::PDBStorageManagerBackend> &storage,
                            const std::string &error) {

  // cast the state
  auto s = dynamic_pointer_cast<TRALocalJoinState>(state);

  // success indicator
  atomic_bool success;
  success = true;

  // create the buzzer
  atomic_int emitterDone;
  emitterDone = 0;
  PDBBuzzerPtr emitterBuzzer = make_shared<PDBBuzzer>([&](PDBAlarm myAlarm, atomic_int &cnt) {

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
    PDBWorkPtr myWork = std::make_shared<pdb::GenericWork>([&emitterDone, s](const PDBBuzzerPtr& callerBuzzer) {

      // run the receiver
      s->emitter->run();

      // signal that the run was successful
      callerBuzzer->buzz(PDBAlarm::WorkAllDone, emitterDone);
    });

    // run the work
    storage->getWorker()->execute(myWork, emitterBuzzer);
  }

  ///

  // create the buzzer
  atomic_int joinCounter;
  joinCounter = 0;
  PDBBuzzerPtr joinBuzzer = make_shared<PDBBuzzer>([&](PDBAlarm myAlarm, atomic_int &cnt) {

    // did we fail?
    if (myAlarm == PDBAlarm::GenericError) {
      success = false;
    }

    // increment the count
    cnt++;
  });

  // here we get a worker per pipeline and run all the preaggregationPipelines.
  for (int workerID = 0; workerID < s->joinPipelines->size(); ++workerID) {

    // get a worker from the server
    PDBWorkerPtr worker = storage->getWorker();

    // make the work
    PDBWorkPtr myWork = std::make_shared<pdb::GenericWork>([&joinCounter, &success, workerID, &s](const PDBBuzzerPtr& callerBuzzer) {

      try {

        // run the pipeline
        (*s->joinPipelines)[workerID]->run();
      }
      catch (std::exception &e) {

        // log the error
        s->logger->error(e.what());

        // we failed mark that we have
        success = false;
      }

      // signal that the run was successful
      callerBuzzer->buzz(PDBAlarm::WorkAllDone, joinCounter);
    });

    // run the work
    worker->execute(myWork, joinBuzzer);
  }

  while (emitterDone != 1) {
    emitterBuzzer->wait();
  }

  while (joinCounter != s->joinPipelines->size()) {
    joinBuzzer->wait();
  }

  // I will kick off only one thread to make the index, if this happens to be an overhead we need to make it parallel.
  // create the buzzer
  atomic_int indexerDone;
  indexerDone = 0;
  PDBBuzzerPtr indexerBuzzer = make_shared<PDBBuzzer>([&](PDBAlarm myAlarm, atomic_int &cnt) {

    // did we fail?
    if (myAlarm == PDBAlarm::GenericError) {
      success = false;
    }

    // we are done here
    cnt = 1;
  });


  auto indexer_start = std::chrono::steady_clock::now();
  {

    // reset the output page set
    s->outputIntermediate->resetPageSet();

    // make the work
    PDBWorkPtr myWork = std::make_shared<pdb::GenericWork>([&indexerDone, s](const PDBBuzzerPtr& callerBuzzer) {

      PDBPageHandle page;
      while((page = s->outputIntermediate->getNextPage(0)) != nullptr) {

        // repin the page
        page->repin();

        // store the page in the indexed page set
        auto loc = s->output->pushPage(page);

        // get the vector from the page
        auto &vec = *(((Record<Vector<Handle<TRABlock>>> *) page->getBytes())->getRootObject());

        // generate the index
        for(int i = 0; i < vec.size(); ++i) {
          s->outputIndex->insert(*vec[i]->metaData, { loc,  i});
        }

        // unpin the page
        page->unpin();
      }

      // signal that the run was successful
      callerBuzzer->buzz(PDBAlarm::WorkAllDone, indexerDone);
    });

    // run the work
    storage->getWorker()->execute(myWork, indexerBuzzer);
  }

  while (indexerDone != 1) {
    indexerBuzzer->wait();
  }
  auto indexer_end = std::chrono::steady_clock::now();

  // if this is too large we need to make indexing parallel
  std::cout << "Indexing overhead was " << std::chrono::duration_cast<std::chrono::nanoseconds>(indexer_end - indexer_start).count() << "[ns]" << '\n';

  return true;
}

void TRALocalJoinStage::cleanup(const pdb::PDBPhysicalAlgorithmStatePtr &state,
                                const std::shared_ptr<pdb::PDBStorageManagerBackend> &storage) {

  // cast the state
  auto s = dynamic_pointer_cast<TRALocalJoinState>(state);

  // unpin them all just in case
  s->leftPageSet->unpinAll();
  s->rightPageSet->unpinAll();
  s->output->unpinAll();

  // remove the left page set as we are not using this
  // TODO this should check what is where leaving it like this for now
  storage->removePageSet({0, lhsPageSet});
  storage->removePageSet({0, "right_intermediate"});
  storage->removePageSet({0, "intermediate"});

  std::cout << "Cleanup\n";
}

TRALocalJoinStage::TRALocalJoinStage(const std::string &lhsPageSet, const std::string &rhsDB, const std::string &rhsSet,
                                     const std::string &sink,
                                     const pdb::Vector<int32_t> &lhsIndices, const pdb::Vector<int32_t> &rhsIndices,
                                     const std::string &firstTupleSet, const std::string &finalTupleSet) :
    PDBPhysicalAlgorithmStage(*(_sink),
                              *(_sources),
                              *(_finalTupleSet),
                              *(_secondarySources),
                              *(_setsToMaterialize)), rhsDB(rhsDB), rhsSet(rhsSet), sink(sink), lhsPageSet(lhsPageSet),
    firstTupleSet(firstTupleSet), finalTupleSet(finalTupleSet), lhsIndices(lhsIndices), rhsIndices(rhsIndices) {}

}

const pdb::PDBSinkPageSetSpec *pdb::TRALocalJoinStage::_sink = nullptr;
const pdb::Vector<pdb::PDBSourceSpec> *pdb::TRALocalJoinStage::_sources = nullptr;
const pdb::String *pdb::TRALocalJoinStage::_finalTupleSet = nullptr;
const pdb::Vector<pdb::Handle<pdb::PDBSourcePageSetSpec>> *pdb::TRALocalJoinStage::_secondarySources = nullptr;
const pdb::Vector<pdb::PDBSetObject> *pdb::TRALocalJoinStage::_setsToMaterialize = nullptr;