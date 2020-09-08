#include <PipelineInterface.h>
#include <ComputePlan.h>
#include <physicalAlgorithms/TRALocalJoinState.h>
#include <processors/NullProcessor.h>
#include <AtomicComputationClasses.h>
#include "TRALocalJoinStage.h"
#include "TRALocalJoinEmmiter.h"
#include "ExJob.h"

namespace pdb {

bool TRALocalJoinStage::setup(const Handle<pdb::ExJob> &job,
                              const PDBPhysicalAlgorithmStatePtr &state,
                              const std::shared_ptr<pdb::PDBStorageManagerBackend> &storage,
                              const std::string &error) {

  // cast the state
  auto s = dynamic_pointer_cast<TRALocalJoinState>(state);

  // get the join comp
  auto joinAtomicComp =
      dynamic_pointer_cast<ApplyJoin>(s->logicalPlan->getComputations().getProducingAtomicComputation(firstTupleSet));


  // the the input page set
  s->inputPageSet = storage->createPageSetFromPDBSet(db, set, false);

  // the emmitter will put set pageser here
  s->leftPageSet = storage->createRandomAccessPageSet({0, "intermediate"});

  // get the rhs page set
  s->rightPageSet = storage->getPageSet({0, inputPageSet});

  // get the in
  s->output = storage->createAnonymousPageSet({0, sink});

  // init the plan
  ComputePlan plan(std::make_shared<LogicalPlan>(job->tcap, *job->computations));
  s->logicalPlan = plan.getPlan();

  std::map<ComputeInfoType, ComputeInfoPtr> params;

  // the join arguments
  auto joinArguments = std::make_shared<JoinArguments>(JoinArgumentsInit{{joinAtomicComp->getRightInput().getSetName(),
                                                                          std::make_shared<JoinArg>(s->rightPageSet)}});

  // mark that this is the join aggregation algorithm
  joinArguments->isTRALocalJoin = true;
  joinArguments->emitter = std::make_shared<TRALocalJoinEmmiter>();

  /// 15.1 Init the preaggregation queues

  // fill uo the vector for each thread
  s->joinPipelines = std::make_shared<std::vector<PipelinePtr>>();
  for (uint64_t pipelineIndex = 0; pipelineIndex < job->numberOfProcessingThreads; ++pipelineIndex) {

    /// 15.2. Figure out the parameters of the pipeline

    // initialize the parameters
    params = {{ComputeInfoType::PAGE_PROCESSOR, std::make_shared<NullProcessor>()},
              {ComputeInfoType::JOIN_ARGS, joinArguments},
              {ComputeInfoType::SHUFFLE_JOIN_ARG, std::make_shared<ShuffleJoinArg>(false)},
              {ComputeInfoType::SOURCE_SET_INFO, nullptr}};

    /// 15.3. Build the pipeline

    auto pipeline = plan.buildPipeline(firstTupleSet, /* this is the TupleSet the pipeline starts with */
                                       finalTupleSet,     /* this is the TupleSet the pipeline ends with */
                                       s->leftPageSet,
                                       s->output,
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

  std::cout << "run\n";
  return true;
}

void TRALocalJoinStage::cleanup(const pdb::PDBPhysicalAlgorithmStatePtr &state,
                                const std::shared_ptr<pdb::PDBStorageManagerBackend> &storage) {
  std::cout << "Cleanup\n";
}

TRALocalJoinStage::TRALocalJoinStage(const std::string &db, const std::string &set,
                                     const std::string &sink, const pdb::Vector<int32_t> &indices,
                                     const std::string &firstTupleSet, const std::string &finalTupleSet) :
    PDBPhysicalAlgorithmStage(*(_sink),
                              *(_sources),
                              *(_finalTupleSet),
                              *(_secondarySources),
                              *(_setsToMaterialize)), db(db), set(set), sink(sink),
    firstTupleSet(firstTupleSet), finalTupleSet(finalTupleSet) {}

}

const pdb::PDBSinkPageSetSpec *pdb::TRALocalJoinStage::_sink = nullptr;
const pdb::Vector<pdb::PDBSourceSpec> *pdb::TRALocalJoinStage::_sources = nullptr;
const pdb::String *pdb::TRALocalJoinStage::_finalTupleSet = nullptr;
const pdb::Vector<pdb::Handle<pdb::PDBSourcePageSetSpec>> *pdb::TRALocalJoinStage::_secondarySources = nullptr;
const pdb::Vector<pdb::PDBSetObject> *pdb::TRALocalJoinStage::_setsToMaterialize = nullptr;