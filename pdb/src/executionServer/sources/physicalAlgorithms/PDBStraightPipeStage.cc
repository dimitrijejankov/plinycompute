#include <PDBStraightPipeStage.h>
#include <PDBStraightPipeState.h>
#include <ExJob.h>
#include <NullProcessor.h>
#include <PDBStorageManagerBackend.h>
#include <ComputePlan.h>
#include <GenericWork.h>

pdb::PDBStraightPipeStage::PDBStraightPipeStage(const pdb::PDBSinkPageSetSpec &sink,
                                                const pdb::Vector<pdb::PDBSourceSpec> &sources,
                                                const pdb::String &finalTupleSet,
                                                const pdb::Vector<pdb::Handle<PDBSourcePageSetSpec>> &secondarySources,
                                                const pdb::Vector<pdb::PDBSetObject> &setsToMaterialize)
    : PDBPhysicalAlgorithmStage(sink, sources, finalTupleSet, secondarySources, setsToMaterialize) {}


bool pdb::PDBStraightPipeStage::setup(const pdb::Handle<pdb::ExJob> &job,
                                      const pdb::PDBPhysicalAlgorithmStatePtr &state,
                                      const std::shared_ptr<pdb::PDBStorageManagerBackend> &storage,
                                      const std::string &error) {

  // cast the state
  auto s = dynamic_pointer_cast<PDBStraightPipeState>(state);

  // init the plan
  ComputePlan plan(std::make_shared<LogicalPlan>(job->tcap, *job->computations));
  s->logicalPlan = plan.getPlan();

  /// 0. Figure out the sink tuple set

  // get the sink page set
  auto sinkPageSet = storage->createAnonymousPageSet(std::make_pair(sink.pageSetIdentifier.first, sink.pageSetIdentifier.second));

  // did we manage to get a sink page set? if not the setup failed
  if(sinkPageSet == nullptr) {
    return false;
  }

  /// 1. Initialize the sources

  // we put them here
  std::vector<PDBAbstractPageSetPtr> sourcePageSets;
  sourcePageSets.reserve(sources.size());

  // initialize them
  for(int i = 0; i < sources.size(); i++) {
    sourcePageSets.emplace_back(getSourcePageSet(storage, i));
  }

  /// 2. Initialize all the pipelines

  // get the number of worker threads from this server's config
  int32_t numWorkers = storage->getConfiguration()->numThreads;

  // check that we have at least one worker per primary source
  if(numWorkers < sources.size()) {
    return false;
  }

  // we put all the pipelines we need to run here
  s->myPipelines = std::make_shared<std::vector<PipelinePtr>>();
  for (uint64_t pipelineIndex = 0; pipelineIndex < numWorkers; ++pipelineIndex) {

    /// 2.1. Figure out what source to use

    // figure out what pipeline
    auto pipelineSource = pipelineIndex % sources.size();

    // grab these thins from the source we need them
    bool swapLHSandRHS = sources[pipelineSource].swapLHSandRHS;
    const pdb::String &firstTupleSet = sources[pipelineSource].firstTupleSet;

    // get the source computation
    auto srcNode = s->logicalPlan->getComputations().getProducingAtomicComputation(firstTupleSet);

    // go grab the source page set
    PDBAbstractPageSetPtr sourcePageSet = sourcePageSets[pipelineSource];

    // did we manage to get a source page set? if not the setup failed
    if(sourcePageSet == nullptr) {
      return false;
    }

    /// 2.2. Figure out the parameters of the pipeline

    // figure out the join arguments
    auto joinArguments = getJoinArguments(storage);

    // if we could not create them we are out of here
    if(joinArguments == nullptr) {
      return false;
    }

    // get catalog client
    auto catalogClient = storage->getFunctionalityPtr<PDBCatalogClient>();

    // empty computations parameters
    std::map<ComputeInfoType, ComputeInfoPtr> params =  {{ComputeInfoType::PAGE_PROCESSOR, std::make_shared<NullProcessor>()},
                                                         {ComputeInfoType::JOIN_ARGS, joinArguments},
                                                         {ComputeInfoType::SHUFFLE_JOIN_ARG, std::make_shared<ShuffleJoinArg>(swapLHSandRHS)},
                                                         {ComputeInfoType::SOURCE_SET_INFO, getSourceSetArg(catalogClient, pipelineSource)}};



    /// 2.3. Build the pipeline

    auto pipeline = plan.buildPipeline(firstTupleSet, /* this is the TupleSet the pipeline starts with */
                                       finalTupleSet,     /* this is the TupleSet the pipeline ends with */
                                       sourcePageSet,
                                       sinkPageSet,
                                       params,
                                       job->thisNode,
                                       job->numberOfNodes,
                                       job->numberOfProcessingThreads,
                                       pipelineIndex);
    s->myPipelines->push_back(pipeline);
  }

  // get the key extractor
  s->keyExtractor = getKeyExtractor(finalTupleSet, plan);

  return true;
}

bool pdb::PDBStraightPipeStage::run(const pdb::Handle<pdb::ExJob> &request,
                                    const pdb::PDBPhysicalAlgorithmStatePtr &state,
                                    const std::shared_ptr<pdb::PDBStorageManagerBackend> &storage,
                                    const std::string &error) {

  // cast the state
  auto s = dynamic_pointer_cast<PDBStraightPipeState>(state);

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
  for (int i = 0; i < s->myPipelines->size(); ++i) {

    // get a worker from the server
    PDBWorkerPtr worker = storage->getWorker();

    // make the work
    PDBWorkPtr myWork = std::make_shared<pdb::GenericWork>([&counter, &success, &s, i](const PDBBuzzerPtr& callerBuzzer) {

      try {

        // run the pipeline
        (*s->myPipelines)[i]->run();
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
  while (counter < s->myPipelines->size()) {
    tempBuzzer->wait();
  }

  // if we failed finish
  if(!success) {
    return success;
  }

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

    // do we need to extract the keys too
    if(s->keyExtractor != nullptr) {
      // materialize the keys
      sinkPageSet->resetPageSet();
      success = storage->materializeKeys(sinkPageSet,std::make_pair<std::string, std::string>(setsToMaterialize[j].database, setsToMaterialize[j].set), s->keyExtractor) && success;
    }
  }

  return success;
}

void pdb::PDBStraightPipeStage::cleanup(const pdb::PDBPhysicalAlgorithmStatePtr &state, const std::shared_ptr<pdb::PDBStorageManagerBackend> &storage) {

  // cast the state
  auto s = dynamic_pointer_cast<PDBStraightPipeState>(state);

  // invalidate everything
  s->myPipelines = nullptr;
  s->logicalPlan = nullptr;
}

