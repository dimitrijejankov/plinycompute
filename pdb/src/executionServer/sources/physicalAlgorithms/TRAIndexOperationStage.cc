#include <physicalAlgorithms/TRAIndexOperationState.h>
#include <GenericWork.h>
#include <TRABlock.h>
#include "TRAIndexOperationStage.h"
#include "ExJob.h"

pdb::TRAIndexOperationStage::TRAIndexOperationStage(const std::string &db, const std::string &set) :
    PDBPhysicalAlgorithmStage(*(_sink),
                              *(_sources),
                              *(_finalTupleSet),
                              *(_secondarySources),
                              *(_setsToMaterialize)), db(db), set(set) {}

bool pdb::TRAIndexOperationStage::setup(const pdb::Handle<pdb::ExJob> &job,
                                   const pdb::PDBPhysicalAlgorithmStatePtr &state,
                                   const std::shared_ptr<pdb::PDBStorageManagerBackend> &storage,
                                   const std::string &error) {

  // cast the state
  auto s = dynamic_pointer_cast<TRAIndexOperationState>(state);

  // input page set
  s->inputSet = storage->createPageSetFromPDBSet((std::string) db, (std::string) set, false)->asRandomAccessPageSet();

  // create an empty index for this page set
  s->index = storage->createIndex({0, ((std::string) db + ":" + (std::string) set)});

  // make the logger
  s->logger = make_shared<PDBLogger>("TRAIndexOperationStage_" + std::to_string(job->computationID));

  return true;
}

bool pdb::TRAIndexOperationStage::run(const pdb::Handle<pdb::ExJob> &job,
                                 const pdb::PDBPhysicalAlgorithmStatePtr &state,
                                 const std::shared_ptr<pdb::PDBStorageManagerBackend> &storage,
                                 const std::string &error) {

  // cast the state
  auto s = dynamic_pointer_cast<TRAIndexOperationState>(state);

  // success indicator
  atomic_bool success;
  success = true;

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

  {

    // make the work
    PDBWorkPtr myWork = std::make_shared<pdb::GenericWork>([&indexerDone, s](const PDBBuzzerPtr& callerBuzzer) {

      PDBPageHandle page;
      for(int loc = 0; loc < s->inputSet->getNumPages(); ++loc) {

        // get the page
        page = (*s->inputSet)[loc];

        // get the vector from the page
        auto &vec = *(((Record<Vector<Handle<TRABlock>>> *) page->getBytes())->getRootObject());

        // generate the index
        for(int i = 0; i < vec.size(); ++i) {
          s->index->insert(*vec[i]->metaData, { loc,  i});
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

  auto indexer_start = std::chrono::steady_clock::now();
  while (indexerDone != 1) {
    indexerBuzzer->wait();
  }
  auto indexer_end = std::chrono::steady_clock::now();

  // if this is too large we need to make indexing parallel
  std::cout << "Indexing overhead was " << std::chrono::duration_cast<std::chrono::nanoseconds>(indexer_end - indexer_start).count() << "[ns]" << '\n';

  return true;
}

void pdb::TRAIndexOperationStage::cleanup(const pdb::PDBPhysicalAlgorithmStatePtr &state,
                                     const std::shared_ptr<pdb::PDBStorageManagerBackend> &storage) {

  std::cout << "cleanup\n";
}

const pdb::PDBSinkPageSetSpec *pdb::TRAIndexOperationStage::_sink = nullptr;
const pdb::Vector<pdb::PDBSourceSpec> *pdb::TRAIndexOperationStage::_sources = nullptr;
const pdb::String *pdb::TRAIndexOperationStage::_finalTupleSet = nullptr;
const pdb::Vector<pdb::Handle<pdb::PDBSourcePageSetSpec>> *pdb::TRAIndexOperationStage::_secondarySources = nullptr;
const pdb::Vector<pdb::PDBSetObject> *pdb::TRAIndexOperationStage::_setsToMaterialize = nullptr;