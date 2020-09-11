#include <physicalAlgorithms/TRALocalAggregationState.h>
#include <GenericWork.h>
#include <TRABlock.h>
#include "TRALocalAggregationStage.h"
#include "ExJob.h"

namespace pdb {

bool TRALocalAggregationStage::setup(const Handle<pdb::ExJob> &job,
                                     const PDBPhysicalAlgorithmStatePtr &state,
                                     const std::shared_ptr<pdb::PDBStorageManagerBackend> &storage,
                                     const std::string &error) {


  // cast the state
  auto s = dynamic_pointer_cast<TRALocalAggregationState>(state);

  // get the input set and index
  s->inputSet = std::dynamic_pointer_cast<pdb::PDBRandomAccessPageSet>(storage->getPageSet({0, inputPageSet}));
  s->index = storage->getIndex({0, inputPageSet});

  // make the output set
  s->outputSet = storage->createRandomAccessPageSet({0, sink});

  return true;
}

bool TRALocalAggregationStage::run(const Handle<pdb::ExJob> &job,
                                   const PDBPhysicalAlgorithmStatePtr &state,
                                   const std::shared_ptr<pdb::PDBStorageManagerBackend> &storage,
                                   const std::string &error) {

  // cast the state
  auto s = dynamic_pointer_cast<TRALocalAggregationState>(state);

  // success indicator
  atomic_bool success;
  success = true;

  // create the buzzer
  atomic_int aggCounter;
  aggCounter = 0;
  PDBBuzzerPtr aggBuzzer = make_shared<PDBBuzzer>([&](PDBAlarm myAlarm, atomic_int &cnt) {

    // did we fail?
    if (myAlarm == PDBAlarm::GenericError) {
      success = false;
    }

    // increment the count
    cnt++;
  });

  // we use this as the hash pattern
  unordered_set<int32_t> pattern;
  for (int32_t i = 0; i < indices.size(); ++i) {
    pattern.insert(indices[i]);
  }

  // repin all the pages
  s->inputSet->repinAll();

  // grab all the vectors
  std::vector<Handle<Vector<Handle<TRABlock>>>> inputVectors;
  for(int i = 0; i < s->inputSet->getNumPages(); ++i) {

    // get the vector from the page
    auto vec = ((Record<Vector<Handle<TRABlock>>> *) (*s->inputSet)[i]->getBytes())->getRootObject();
    inputVectors.push_back(vec);
  }

  std::vector<TRAIndexNodePtr> tempIdx(job->numberOfProcessingThreads);
  for (int workerID = 0; workerID < job->numberOfProcessingThreads; ++workerID) {

    // make a temporary index we will use this for the aggregation
    tempIdx[workerID] = std::make_shared<TRAIndexNode>(false);

    // get a worker from the server
    PDBWorkerPtr worker = storage->getWorker();

    // make the work
    PDBWorkPtr myWork = std::make_shared<pdb::GenericWork>([&aggCounter, &success, &job, &inputVectors,
                                                               workerID, &s, this, &pattern](const PDBBuzzerPtr &callerBuzzer) {

      // get all the record that belong to this node
      std::vector<std::pair<int32_t, int32_t>> out;
      s->index->getWithHash(out, pattern, workerID, job->numberOfProcessingThreads);

      // on now we need to aggregate them
      for(int i = 0; i < out.size();) {

        // grab the record index
        auto &recordIndex = out[i];

        auto record = (*inputVectors[recordIndex.first])[recordIndex.second];
      }
      // signal that the run was successful
      callerBuzzer->buzz(PDBAlarm::WorkAllDone, aggCounter);
    });

    // run the work
    worker->execute(myWork, aggBuzzer);
  }

  return true;
}

void TRALocalAggregationStage::cleanup(const pdb::PDBPhysicalAlgorithmStatePtr &state,
                                       const std::shared_ptr<pdb::PDBStorageManagerBackend> &storage) {

}

TRALocalAggregationStage::TRALocalAggregationStage(const pdb::String &inputPageSet,
                                                   const pdb::Vector<int32_t> &indices,
                                                   const pdb::String &sink) :
    PDBPhysicalAlgorithmStage(*(_sink),
                              *(_sources),
                              *(_finalTupleSet),
                              *(_secondarySources),
                              *(_setsToMaterialize)), inputPageSet(inputPageSet), indices(indices), sink(sink) {}

}

const pdb::PDBSinkPageSetSpec *pdb::TRALocalAggregationStage::_sink = nullptr;
const pdb::Vector<pdb::PDBSourceSpec> *pdb::TRALocalAggregationStage::_sources = nullptr;
const pdb::String *pdb::TRALocalAggregationStage::_finalTupleSet = nullptr;
const pdb::Vector<pdb::Handle<pdb::PDBSourcePageSetSpec>> *pdb::TRALocalAggregationStage::_secondarySources = nullptr;
const pdb::Vector<pdb::PDBSetObject> *pdb::TRALocalAggregationStage::_setsToMaterialize = nullptr;