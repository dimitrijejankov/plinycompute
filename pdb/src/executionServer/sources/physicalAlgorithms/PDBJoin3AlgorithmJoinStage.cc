#include <PDBJoin3AlgorithmJoinStage.h>
#include <PDBJoin3AlgorithmState.h>
#include <ExJob.h>
#include <ComputePlan.h>
#include <NullProcessor.h>
#include <GenericWork.h>

bool pdb::PDBJoin3AlgorithmJoinStage::setup(const pdb::Handle<pdb::ExJob> &job,
                                            const pdb::PDBPhysicalAlgorithmStatePtr &state,
                                            const std::shared_ptr<pdb::PDBStorageManagerBackend> &storage,
                                            const std::string &error) {

  // cast the state
  auto s = dynamic_pointer_cast<PDBJoin3AlgorithmState>(state);

  // get plan from the page
  auto* record = (Record<JoinPlannerResult>*) s->planPage->getBytes();
  auto planResult = record->getRootObject();

  // go through the join records and figure out what are on this node
  for(int i = 0; i < planResult->joinedRecords->size(); ++i) {

    // get the node
    auto node = (*planResult->join_group_mapping)[i];

    // if this is our node
    if(node == job->thisNode) {
      s->joinedRecords.emplace_back((*planResult->joinedRecords)[i]);
    }
  }

  // init the plan
  ComputePlan plan(std::make_shared<LogicalPlan>(job->tcap, *job->computations));
  s->logicalPlan = plan.getPlan();

  // make the sink set
  auto sinkPageSet = storage->createAnonymousPageSet({0, "out"});

  // get the source page set
  auto sourcePageSet = storage->getPageSet({0, "intermediate"});
  sourcePageSet->resetPageSet();

  // make the join arguments
  auto joinArguments = std::make_shared<JoinArguments>();

  // set the plan page and the inputs
  joinArguments->planPage = s->planPage;
  joinArguments->inputs = { in0, in1, in2, in3, in4, in5, in6, in7 };
  joinArguments->isJoin8 = true;
  joinArguments->joinedRecords = &s->joinedRecords;
  joinArguments->mappings = &s->TIDToRecordMapping;

  // empty computations parameters
  std::map<ComputeInfoType, ComputeInfoPtr> params =  {{ComputeInfoType::PAGE_PROCESSOR, std::make_shared<NullProcessor>()},
                                                       {ComputeInfoType::JOIN_ARGS, joinArguments}};

  // build the pipelines
  s->myPipelines = std::make_shared<std::vector<PipelinePtr>>();
  for (uint64_t pipelineIndex = 0; pipelineIndex < job->numberOfProcessingThreads; ++pipelineIndex) {

    // build the pipelines
    auto pipeline = plan.buildPipeline(in0, /* this is the TupleSet the pipeline starts with */
                                       "OutFor_0_joinRec_41JoinComp8_out",     /* this is the TupleSet the pipeline ends with */
                                       sourcePageSet,
                                       sinkPageSet,
                                       params,
                                       job->thisNode,
                                       job->numberOfNodes,
                                       job->numberOfProcessingThreads,
                                       pipelineIndex);

    // insert the pipeline
    s->myPipelines->emplace_back(pipeline);
  }

  return true;
}

bool pdb::PDBJoin3AlgorithmJoinStage::run(const pdb::Handle<pdb::ExJob> &job,
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

  // get the page set
  auto sinkPageSet = storage->getPageSet({0, "out"});

  // if the thing does not exist finish!
  if(sinkPageSet == nullptr) {
    return false;
  }

  // materialize the page set
  sinkPageSet->resetPageSet();
  success = storage->materializePageSet(sinkPageSet, { sinkSet.database, sinkSet.set });

  return success;
}

void pdb::PDBJoin3AlgorithmJoinStage::cleanup(const pdb::PDBPhysicalAlgorithmStatePtr &state) {
}
