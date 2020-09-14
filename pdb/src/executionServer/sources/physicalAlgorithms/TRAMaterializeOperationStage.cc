#include <physicalAlgorithms/TRAMaterializeOperationState.h>
#include <GenericWork.h>
#include <TRABlock.h>
#include "TRAMaterializeOperationStage.h"
#include "ExJob.h"

pdb::TRAMaterializeOperationStage::TRAMaterializeOperationStage(const std::string &db, const std::string &set,
                                                                const std::string &pageSet) :
    PDBPhysicalAlgorithmStage(*(_sink),
                              *(_sources),
                              *(_finalTupleSet),
                              *(_secondarySources),
                              *(_setsToMaterialize)), db(db), set(set), inputPageSet(pageSet) {}

bool pdb::TRAMaterializeOperationStage::setup(const pdb::Handle<pdb::ExJob> &job,
                                   const pdb::PDBPhysicalAlgorithmStatePtr &state,
                                   const std::shared_ptr<pdb::PDBStorageManagerBackend> &storage,
                                   const std::string &error) {

  // cast the state
  auto s = dynamic_pointer_cast<TRAMaterializeOperationState>(state);

  // input page set
  s->inputSet = std::dynamic_pointer_cast<pdb::PDBRandomAccessPageSet>(storage->getPageSet({0, inputPageSet}));

  // make the logger
  s->logger = make_shared<PDBLogger>("TRAMaterializeOperationStage_" + std::to_string(job->computationID));

  return true;
}

bool pdb::TRAMaterializeOperationStage::run(const pdb::Handle<pdb::ExJob> &job,
                                 const pdb::PDBPhysicalAlgorithmStatePtr &state,
                                 const std::shared_ptr<pdb::PDBStorageManagerBackend> &storage,
                                 const std::string &error) {

  // cast the state
  auto s = dynamic_pointer_cast<TRAMaterializeOperationState>(state);

  // materialize the page set
  s->inputSet->resetPageSet();
  storage->materializePageSet(s->inputSet, std::make_pair<std::string, std::string>(db, set));

  return true;
}

void pdb::TRAMaterializeOperationStage::cleanup(const pdb::PDBPhysicalAlgorithmStatePtr &state,
                                                const std::shared_ptr<pdb::PDBStorageManagerBackend> &storage) {

  storage->removePageSet({0, inputPageSet});
  std::cout << "cleanup\n";
}

const pdb::PDBSinkPageSetSpec *pdb::TRAMaterializeOperationStage::_sink = nullptr;
const pdb::Vector<pdb::PDBSourceSpec> *pdb::TRAMaterializeOperationStage::_sources = nullptr;
const pdb::String *pdb::TRAMaterializeOperationStage::_finalTupleSet = nullptr;
const pdb::Vector<pdb::Handle<pdb::PDBSourcePageSetSpec>> *pdb::TRAMaterializeOperationStage::_secondarySources = nullptr;
const pdb::Vector<pdb::PDBSetObject> *pdb::TRAMaterializeOperationStage::_setsToMaterialize = nullptr;