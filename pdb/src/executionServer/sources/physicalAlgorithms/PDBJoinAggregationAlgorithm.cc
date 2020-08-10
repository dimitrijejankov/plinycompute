#include <utility>

#include <AtomicComputationClasses.h>
#include "PDBJoinAggregationAlgorithm.h"
#include "PDBJoinAggregationKeyStage.h"
#include <ExJob.h>
#include <physicalAlgorithms/PDBJoinAggregationComputationStage.h>
#include "GenericWork.h"
#include "PDBLabeledPageSet.h"
#include "PDBBroadcastForJoinState.h"
#include "AggregateCompBase.h"
#include "PDBJoinAggregationLocalAggregationStage.h"
#include "PDBJoinAggregationAggregationStage.h"
#include "PDBJoinAggregationLocalComputationStage.h"

namespace pdb {

PDBJoinAggregationAlgorithm::PDBJoinAggregationAlgorithm(const std::vector<PDBPrimarySource> &leftSource,
                                                         const std::vector<PDBPrimarySource> &rightSource,
                                                         const pdb::Handle<PDBSinkPageSetSpec> &sink,
                                                         const pdb::Handle<PDBSinkPageSetSpec> &leftKeySink,
                                                         const pdb::Handle<PDBSinkPageSetSpec> &rightKeySink,
                                                         const pdb::Handle<PDBSinkPageSetSpec> &joinAggKeySink,
                                                         const pdb::Handle<PDBSinkPageSetSpec> &intermediateSink,
                                                         const pdb::Handle<PDBSinkPageSetSpec> &preaggIntermediate,
                                                         const pdb::Handle<PDBSourcePageSetSpec> &leftJoinSource,
                                                         const pdb::Handle<PDBSourcePageSetSpec> &rightJoinSource,
                                                         const AtomicComputationPtr &leftInputTupleSet,
                                                         const AtomicComputationPtr &rightInputTupleSet,
                                                         const AtomicComputationPtr &joinTupleSet,
                                                         const AtomicComputationPtr &aggregationKey,
                                                         const std::vector<pdb::Handle<PDBSourcePageSetSpec>> &secondarySources,
                                                         const pdb::Handle<pdb::Vector<PDBSetObject>> &setsToMaterialize)
    : leftInputTupleSet(leftInputTupleSet->getOutputName()),
      rightInputTupleSet(rightInputTupleSet->getOutputName()),
      joinTupleSet(joinTupleSet->getOutputName()) {

  // set the sink
  this->sink = sink;

  // set the key sinks
  this->lhsKeySink = leftKeySink;
  this->rhsKeySink = rightKeySink;
  this->joinAggKeySink = joinAggKeySink;
  this->intermediateSink = intermediateSink;
  this->preaggIntermediate = preaggIntermediate;

  // set the join sources
  this->leftJoinSource = leftJoinSource;
  this->rightJoinSource = rightJoinSource;

  // set the sets to materialize
  this->setsToMaterialize = setsToMaterialize;

  // set the final tuple set
  finalTupleSet = aggregationKey->getOutputName();

  // ini the source sizes
  sources = pdb::Vector<PDBSourceSpec>(leftSource.size(), leftSource.size());
  rightSources = pdb::Vector<PDBSourceSpec>(rightSource.size(), rightSource.size());

  // copy all the primary sources
  for (int i = 0; i < leftSource.size(); ++i) {

    // grab the source
    auto &source = leftSource[i];

    // check if we are scanning a set if we are fill in sourceSet field
    if (source.startAtomicComputation->getAtomicComputationTypeID() == ScanSetAtomicTypeID) {

      // cast to a scan set
      auto scanSet = (ScanSet *) source.startAtomicComputation.get();

      // get the set info
      sources[i].sourceSet = pdb::makeObject<PDBSetObject>(scanSet->getDBName(), scanSet->getSetName());
    } else {
      sources[i].sourceSet = nullptr;
    }

    sources[i].firstTupleSet = source.startAtomicComputation->getOutputName();
    sources[i].pageSet = source.source;
    sources[i].swapLHSandRHS = source.shouldSwapLeftAndRight;
  }

  // copy all the primary sources
  for (int i = 0; i < rightSource.size(); ++i) {

    // grab the source
    auto &source = rightSource[i];

    // check if we are scanning a set if we are fill in sourceSet field
    if (source.startAtomicComputation->getAtomicComputationTypeID() == ScanSetAtomicTypeID) {

      // cast to a scan set
      auto scanSet = (ScanSet *) source.startAtomicComputation.get();

      // get the set info
      rightSources[i].sourceSet = pdb::makeObject<PDBSetObject>(scanSet->getDBName(), scanSet->getSetName());
    } else {
      rightSources[i].sourceSet = nullptr;
    }

    rightSources[i].firstTupleSet = source.startAtomicComputation->getOutputName();
    rightSources[i].pageSet = source.source;
    rightSources[i].swapLHSandRHS = source.shouldSwapLeftAndRight;
  }

  // copy all the secondary sources
  this->secondarySources = pdb::makeObject<pdb::Vector<pdb::Handle<PDBSourcePageSetSpec>>>(secondarySources.size(), 0);
  for (const auto &secondarySource : secondarySources) {
    this->secondarySources->push_back(secondarySource);
  }
}

PDBPhysicalAlgorithmStatePtr PDBJoinAggregationAlgorithm::getInitialState(const pdb::Handle<pdb::ExJob> &job) const {
  // init the state
  auto state = std::make_shared<PDBJoinAggregationState>();

  // init the logger for this algorithm
  state->logger = make_shared<PDBLogger>("PDBJoinAggregationAlgorithm_" + std::to_string(job->computationID));

  // return the state
  return state;
}

pdb::PDBPhysicalAlgorithmStagePtr PDBJoinAggregationAlgorithm::getNextStage(const PDBPhysicalAlgorithmStatePtr &state) {

  // we are done if we already served a stage
  if (currentStage == 2) {
    return nullptr;
  }

  // create the right stages
  switch (currentStage) {

    case 0: {

      // go to the next stage
      currentStage++;

      // return the key stage
      return std::make_shared<PDBJoinAggregationKeyStage>(*sink,
                                                          sources,
                                                          finalTupleSet,
                                                          *secondarySources,
                                                          *setsToMaterialize,
                                                          leftInputTupleSet,
                                                          rightInputTupleSet,
                                                          joinTupleSet,
                                                          *lhsKeySink,
                                                          *rhsKeySink,
                                                          *joinAggKeySink,
                                                          rightSources);
    }
    case 1: {

    }
      // cast the state
      auto s = dynamic_pointer_cast<PDBJoinAggregationState>(state);

      // go to the next stage
      currentStage++;

      if (s->localAggregation) {

        // return the computation
        return std::make_shared<PDBJoinAggregationLocalComputationStage>(*sink,
                                                                         sources,
                                                                         finalTupleSet,
                                                                         *secondarySources,
                                                                         *setsToMaterialize,
                                                                         joinTupleSet,
                                                                         *leftJoinSource,
                                                                         *rightJoinSource,
                                                                         *intermediateSink,
                                                                         rightSources);
      } else {

        // return the computation
        return std::make_shared<PDBJoinAggregationComputationStage>(*sink,
                                                                    *preaggIntermediate,
                                                                    sources,
                                                                    finalTupleSet,
                                                                    *secondarySources,
                                                                    *setsToMaterialize,
                                                                    joinTupleSet,
                                                                    *leftJoinSource,
                                                                    *rightJoinSource,
                                                                    *intermediateSink,
                                                                    rightSources);
      }
  }

  throw runtime_error("Unrecognized stage. How did we get here?");
}

int32_t PDBJoinAggregationAlgorithm::numStages() const {
  return 2;
}

pdb::PDBPhysicalAlgorithmType pdb::PDBJoinAggregationAlgorithm::getAlgorithmType() {
  return JoinAggregation;
}

pdb::PDBCatalogSetContainerType pdb::PDBJoinAggregationAlgorithm::getOutputContainerType() {

  // ends with an aggregation therefore it is a map
  return PDB_CATALOG_SET_MAP_CONTAINER;
}

}