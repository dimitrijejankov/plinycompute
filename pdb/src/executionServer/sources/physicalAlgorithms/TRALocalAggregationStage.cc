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
  s->outputIndex = storage->createIndex({0, sink});

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

  std::vector<AggregationIndex> tempIdx(job->numberOfProcessingThreads);
  for (int workerID = 0; workerID < job->numberOfProcessingThreads; ++workerID) {

    // get a worker from the server
    PDBWorkerPtr worker = storage->getWorker();

    // make the work
    PDBWorkPtr myWork = std::make_shared<pdb::GenericWork>([&aggCounter, &success, &job, &inputVectors,
                                                                        workerID, &s, this, &pattern, &tempIdx ](const PDBBuzzerPtr &callerBuzzer) {

      // the aggregation index
      auto &aggIdx = tempIdx[workerID];

      // we put the key here
      std::vector<uint32_t> key;

      // get all the record that belong to this node
      std::vector<std::pair<int32_t, int32_t>> out;
      s->index->getWithHash(out, pattern, workerID, job->numberOfProcessingThreads);

      // do we have something to process...
      if(out.empty()){

        // signal that the run was successful
        callerBuzzer->buzz(PDBAlarm::WorkAllDone, aggCounter);
        return;
      }

      // get a new page
      auto currentPage = s->outputSet->getNewPage();
      makeObjectAllocatorBlock(currentPage->getBytes(), currentPage->getSize(), true);

      // make the vector we write to
      std::vector<Handle<Vector<Handle<pdb::TRABlock>>>> vectors;
      vectors.emplace_back(makeObject<Vector<Handle<pdb::TRABlock>>>());

      // is there stuff on the page
      bool stuffOnPage = false;

      // on now we need to aggregate them
      for(int i = 0; i < out.size();) {

        // grab the record index
        auto &recordIndex = out[i];

        // get the record
        auto record = (*inputVectors[recordIndex.first])[recordIndex.second];

        // form the key
        key.clear();
        for (int32_t k = 0; k < indices.size(); ++k) {
          key.emplace_back(record->metaData->indices[indices[k]]);
        }

        // get the location
        auto &loc = aggIdx.get(key);

        // if we did not find it insert it
        if(loc.first == -1) {

          try {

            // store it
            vectors.back()->push_back(record);

            // reduce the index size this will not reallocate
            while((*vectors.back())[vectors.back()->size() - 1]->metaData->indices.size() != key.size()) {
              (*vectors.back())[vectors.back()->size() - 1]->metaData->indices.pop_back();
            }

            // copy the key
            for(int j = 0; j < key.size(); ++j) {
              (*vectors.back())[vectors.back()->size() - 1]->metaData->indices[j] = key[j];
            }

            aggIdx.insert(key, { vectors.size() - 1, vectors.back()->size() - 1 });
            stuffOnPage = true;

            // go to the next record
            i++;

          } catch (pdb::NotEnoughSpace &n) {

            // make this the root object
            getRecord(vectors.back());

            // grab a new page
            stuffOnPage = false;
            currentPage = s->outputSet->getNewPage();
            makeObjectAllocatorBlock(currentPage->getBytes(), currentPage->getSize(), true);

            // make a new vector!
            vectors.emplace_back(makeObject<Vector<Handle<pdb::TRABlock>>>());
          }
        }
        else {

          // get the object
          TRABlockData &lhs = (*(*vectors[loc.first])[loc.second]->data);
          TRABlockData &rhs = *record->data;

          // aggregate it
          for(int32_t g = 0; lhs.data->size(); ++g) {
            (*lhs.data)[g] += (*rhs.data)[g];
          }

          // go to the next record
          i++;
        }
      }

      // is there some stuff on the page
      if(stuffOnPage) {

        // make this the root object
        getRecord(vectors.back());
      }

      // invalidate the block
      makeObjectAllocatorBlock(1024, true);

      // signal that the run was successful
      callerBuzzer->buzz(PDBAlarm::WorkAllDone, aggCounter);
    });

    // run the work
    worker->execute(myWork, aggBuzzer);
  }

  // wait for the aggregation to finish
  while (aggCounter != job->numberOfProcessingThreads) {
    aggBuzzer->wait();
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
    // make the work
    PDBWorkPtr myWork = std::make_shared<pdb::GenericWork>([&indexerDone, s](const PDBBuzzerPtr& callerBuzzer) {

      PDBPageHandle page;
      for(int loc = 0; loc < s->outputSet->getNumPages(); ++loc) {

        // grab a page
        page = (*s->outputSet)[loc];
        page->repin();

        // get the vector from the page
        auto &vec = *(((Record<Vector<Handle<TRABlock>>> *) page->getBytes())->getRootObject());

        // generate the index
        for(int i = 0; i < vec.size(); ++i) {
          vec[i]->print();
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

void TRALocalAggregationStage::cleanup(const pdb::PDBPhysicalAlgorithmStatePtr &state,
                                       const std::shared_ptr<pdb::PDBStorageManagerBackend> &storage) {

  // cast the state
  auto s = dynamic_pointer_cast<TRALocalAggregationState>(state);

  // unpin all
  s->outputSet->unpinAll();
  s->inputSet->unpinAll();
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