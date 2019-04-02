//
// Created by dimitrije on 3/20/19.
//

#include "physicalAlgorithms/PDBAggregationPipeAlgorithm.h"
#include "ComputePlan.h"
#include "ExJob.h"
#include "PDBAggregationPipeAlgorithm.h"
#include "PDBStorageManagerBackend.h"
#include "GenericWork.h"

pdb::PDBAggregationPipeAlgorithm::PDBAggregationPipeAlgorithm(const std::string &firstTupleSet,
                                                              const std::string &finalTupleSet,
                                                              const pdb::Handle<pdb::PDBSourcePageSetSpec> &source,
                                                              const pdb::Handle<pdb::PDBSinkPageSetSpec> &hashedToSend,
                                                              const pdb::Handle<pdb::PDBSourcePageSetSpec> &hashedToRecv,
                                                              const pdb::Handle<pdb::PDBSinkPageSetSpec> &sink,
                                                              const pdb::Handle<pdb::Vector<pdb::PDBSourcePageSetSpec>> &secondarySources)
    : PDBPhysicalAlgorithm(firstTupleSet, finalTupleSet, source, sink, secondarySources),
      hashedToSend(hashedToSend),
      hashedToRecv(hashedToRecv) {}

bool pdb::PDBAggregationPipeAlgorithm::setup(std::shared_ptr<pdb::PDBStorageManagerBackend> &storage,
                                             Handle<pdb::ExJob> &job,
                                             const std::string &error) {

  // init the plan
  ComputePlan plan(job->tcap, *job->computations);
  LogicalPlanPtr logicalPlan = plan.getPlan();

  /// 1. Figure out the source page set

  // get the source computation
  auto srcNode = logicalPlan->getComputations().getProducingAtomicComputation(firstTupleSet);

  // if this is a scan set get the page set from a real set
  PDBAbstractPageSetPtr sourcePageSet;
  if (srcNode->getAtomicComputationTypeID() == ScanSetAtomicTypeID) {

    // cast it to a scan
    auto scanNode = std::dynamic_pointer_cast<ScanSet>(srcNode);

    // get the page set
    sourcePageSet = storage->createPageSetFromPDBSet(scanNode->getDBName(),
                                                     scanNode->getSetName(),
                                                     std::make_pair(source->pageSetIdentifier.first,
                                                                    source->pageSetIdentifier.second));
  } else {

    // we are reading from an existing page set get it
    sourcePageSet = storage->getPageSet(std::make_pair(job->computationID, firstTupleSet));
  }

  // did we manage to get a source page set? if not the setup failed
  if (sourcePageSet == nullptr) {
    return false;
  }

  /// 2. Figure out the sink tuple set

  // figure out the sink node
  auto sinkNode = logicalPlan->getComputations().getProducingAtomicComputation(finalTupleSet);

  // get the sink page set
  auto intermediatePageSet = storage->createAnonymousPageSet(hashedToSend->pageSetIdentifier);

  // did we manage to get a sink page set? if not the setup failed
  if (intermediatePageSet == nullptr) {
    return false;
  }

  /// 3. Init the preaggregation queues

  pageQueues = std::make_shared<std::vector<preaggPageQueuePtr>>();
  for(int i = 0; i < job->numberOfNodes; ++i) { pageQueues->emplace_back(std::make_shared<preaggPageQueue>()); }

  /// 4. Init the preaggregation pipelines

  // set the parameters
  auto myMgr = storage->getFunctionalityPtr<PDBBufferManagerInterface>();
  std::map<ComputeInfoType, ComputeInfoPtr> params = { { ComputeInfoType::PAGE_PROCESSOR,  std::make_shared<PreaggregationPageProcessor>(job->numberOfNodes, job->numberOfProcessingThreads, *pageQueues, myMgr) } };

  // fill uo the vector for each thread
  pipelines = std::make_shared<std::vector<PipelinePtr>>();
  for (uint64_t i = 0; i < job->numberOfProcessingThreads; ++i) {

    auto pipeline = plan.buildPipeline(firstTupleSet, /* this is the TupleSet the pipeline starts with */
                                       finalTupleSet,     /* this is the TupleSet the pipeline ends with */
                                       sourcePageSet,
                                       intermediatePageSet,
                                       params,
                                       job->numberOfNodes,
                                       job->numberOfProcessingThreads,
                                       20,
                                       i);
    pipelines->push_back(pipeline);
  }

  return true;
}

bool pdb::PDBAggregationPipeAlgorithm::run(std::shared_ptr<pdb::PDBStorageManagerBackend> &storage) {

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

  // here we get a worker per pipeline and run all the pipelines.
  for (int i = 0; i < pipelines->size(); ++i) {

    // get a worker from the server
    PDBWorkerPtr worker = storage->getWorker();

    // make the work
    PDBWorkPtr myWork = std::make_shared<pdb::GenericWork>([&counter, i, this](PDBBuzzerPtr callerBuzzer) {

      // run the pipeline
      (*pipelines)[i]->run();

      // signal that the run was successful
      callerBuzzer->buzz(PDBAlarm::WorkAllDone, counter);
    });
    // run the work
    worker->execute(myWork, tempBuzzer);
  }

  // wait until all the pipelines have completed
  while (counter < pipelines->size()) {
    tempBuzzer->wait();
  }

  return true;
}