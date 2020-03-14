#include <PDBJoin8AlgorithmKeyStage.h>
#include <ExJob.h>
#include <physicalAlgorithms/PDBJoin8AlgorithmState.h>
#include <GenericWork.h>

bool pdb::PDBJoin8AlgorithmKeyStage::setup(const pdb::Handle<pdb::ExJob> &job,
                                           const pdb::PDBPhysicalAlgorithmStatePtr &state,
                                           const std::shared_ptr<pdb::PDBStorageManagerBackend> &storage,
                                           const std::string &error) {

  // cast the state
  auto s = dynamic_pointer_cast<PDBJoin8AlgorithmState>(state);

  // make a logical plan
  s->logicalPlan = std::make_shared<LogicalPlan>(job->tcap, *job->computations);

  // get catalog client
  auto catalogClient = storage->getFunctionalityPtr<PDBCatalogClient>();

  // the current node
  int32_t currNode = job->thisNode;

  // go through each node
  for(int n = 0; n < job->numberOfNodes; n++) {

      // make the parameters for the first set
      std::map<ComputeInfoType, ComputeInfoPtr> params = {{ ComputeInfoType::SOURCE_SET_INFO,
                                                            getKeySourceSetArg(catalogClient)} };

      // go grab the source page set
      bool isThisNode = job->nodes[currNode]->address == job->nodes[n]->address && job->nodes[currNode]->port == job->nodes[n]->port;
      PDBAbstractPageSetPtr sourcePageSet = isThisNode ? getKeySourcePageSet(storage, sources) :
                                                         getFetchingPageSet(storage, sources, job->nodes[n]->address, job->nodes[n]->port);

      // store the pipeline
      s->keySourcePageSets[n] = sourcePageSet;
  }

  s->keyPipeline = std::make_shared<EightWayJoinPipeline>(s->keySourcePageSets);

  return true;
}

pdb::SourceSetArgPtr pdb::PDBJoin8AlgorithmKeyStage::getKeySourceSetArg(std::shared_ptr<pdb::PDBCatalogClient> &catalogClient) {

  // grab the set
  std::string error;
  auto set = catalogClient->getSet(sourceSet.database, sourceSet.set, error);
  if(set == nullptr || !set->isStoringKeys) {
    return nullptr;
  }

  // update the set so it is keyed
  set->name = PDBCatalog::toKeySetName(sourceSet.set);
  set->containerType = PDB_CATALOG_SET_VECTOR_CONTAINER;

  // return the argument
  return std::make_shared<pdb::SourceSetArg>(set);
}

bool pdb::PDBJoin8AlgorithmKeyStage::run(const pdb::Handle<pdb::ExJob> &job,
                                         const pdb::PDBPhysicalAlgorithmStatePtr &state,
                                         const std::shared_ptr<pdb::PDBStorageManagerBackend> &storage,
                                         const std::string &error) {
  // check if it is the lead node
  if(job->isLeadNode) {

    // run the lead node
    return runLead(job, state, storage, error);
  }

  return runFollower(job, state, storage, error);
}

void pdb::PDBJoin8AlgorithmKeyStage::cleanup(const pdb::PDBPhysicalAlgorithmStatePtr &state) {

}

bool pdb::PDBJoin8AlgorithmKeyStage::runLead(const pdb::Handle<pdb::ExJob> &job,
                                             const pdb::PDBPhysicalAlgorithmStatePtr &state,
                                             const std::shared_ptr<pdb::PDBStorageManagerBackend> &storage,
                                             const std::string &error) {
  // cast the state
  auto s = dynamic_pointer_cast<PDBJoin8AlgorithmState>(state);

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

  // go through the nodes and execute the key pipelines
  for(int n = 0; n < job->numberOfNodes; n++) {

    // get a worker from the server
    PDBWorkerPtr worker = storage->getWorker();

    // make the work
    PDBWorkPtr myWork = std::make_shared<pdb::GenericWork>([&s, n, &success, &counter] (const PDBBuzzerPtr& callerBuzzer) {

      try {

        // run the pipeline
        s->keyPipeline->runSide(n);
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

  // wait for it to finish
  while (counter < job->numberOfNodes) {
    tempBuzzer->wait();
  }

  return true;
}

bool pdb::PDBJoin8AlgorithmKeyStage::runFollower(const pdb::Handle<pdb::ExJob> &job,
                                                 const pdb::PDBPhysicalAlgorithmStatePtr &state,
                                                 const std::shared_ptr<pdb::PDBStorageManagerBackend> &storage,
                                                 const std::string &error) {
  return true;
}

pdb::PDBAbstractPageSetPtr pdb::PDBJoin8AlgorithmKeyStage::getKeySourcePageSet(const std::shared_ptr<pdb::PDBStorageManagerBackend> &storage,
                                                                               const pdb::Vector<PDBSourceSpec> &srcs) {

  // if this is a scan set get the page set from a real set
  PDBAbstractPageSetPtr sourcePageSet = storage->createPageSetFromPDBSet(sourceSet.database, sourceSet.set, true);
  sourcePageSet->resetPageSet();

  // return the page set
  return sourcePageSet;
}

pdb::PDBAbstractPageSetPtr pdb::PDBJoin8AlgorithmKeyStage::getFetchingPageSet(const std::shared_ptr<pdb::PDBStorageManagerBackend> &storage,
                                                                              const pdb::Vector<PDBSourceSpec> &srcs,
                                                                              const std::string &ip,
                                                                              int32_t port) {
  // get the page set
  PDBAbstractPageSetPtr sourcePageSet = storage->fetchPDBSet(sourceSet.database, sourceSet.set, true, ip, port);
  sourcePageSet->resetPageSet();

  // return the page set
  return sourcePageSet;
}
