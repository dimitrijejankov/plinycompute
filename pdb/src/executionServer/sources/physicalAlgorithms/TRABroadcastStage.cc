#include <physicalAlgorithms/TRABroadcastState.h>
#include <GenericWork.h>
#include "TRABroadcastStage.h"
#include "ExJob.h"

pdb::TRABroadcastStage::TRABroadcastStage(const std::string &db, const std::string &set, const std::string &sink) :
  PDBPhysicalAlgorithmStage(*(_sink),
                            *(_sources),
                            *(_finalTupleSet),
                            *(_secondarySources),
                            *(_setsToMaterialize)), db(db), set(set), sink(sink) {}

bool pdb::TRABroadcastStage::setup(const pdb::Handle<pdb::ExJob> &job,
                                   const pdb::PDBPhysicalAlgorithmStatePtr &state,
                                   const std::shared_ptr<pdb::PDBStorageManagerBackend> &storage,
                                   const std::string &error) {

  // cast the state
  auto s = dynamic_pointer_cast<TRABroadcastState>(state);

  s->logger = make_shared<PDBLogger>("BHBroadcastStage_" + std::to_string(job->computationID));

  /// 0. Init the shuffle queues

  s->pageQueues = std::make_shared<std::vector<PDBPageQueuePtr>>();
  for(int i = 0; i < job->numberOfNodes; ++i) { s->pageQueues->emplace_back(std::make_shared<PDBPageQueue>()); }

  // input page set
  auto tmp = storage->createPageSetFromPDBSet((std::string) db, (std::string) set, false);
  tmp->resetPageSet();
  s->inputSet = tmp;

  // get the receive page set
  auto recvPageSet = storage->createFeedingAnonymousPageSet({0, ((std::string) sink) },
                                                            job->numberOfProcessingThreads,
                                                            job->numberOfNodes);

  // make sure we can use them all at the same time
  recvPageSet->setUsagePolicy(PDBFeedingPageSetUsagePolicy::KEEP_AFTER_USED);

  /// 1. Create the self receiver to forward pages that are created on this node and the network senders to forward pages for the other nodes

  // get the buffer manager
  auto myMgr = storage->getFunctionalityPtr<PDBBufferManagerInterface>();

  int32_t currNode = job->thisNode;
  s->senders = std::make_shared<std::vector<PDBPageNetworkSenderPtr>>();
  for(unsigned i = 0; i < job->nodes.size(); ++i) {

    // check if it is this node or another node
    if(job->nodes[i]->port == job->nodes[currNode]->port && job->nodes[i]->address == job->nodes[currNode]->address) {

      // make the self receiver
      s->selfReceiver = std::make_shared<pdb::PDBPageSelfReceiver>(s->pageQueues->at(i), recvPageSet, myMgr);
    }
    else {

      // make the sender
      auto sender = std::make_shared<PDBPageNetworkSender>(job->nodes[i]->address,
                                                           job->nodes[i]->port,
                                                           job->numberOfProcessingThreads,
                                                           job->numberOfNodes,
                                                           storage->getConfiguration()->maxRetries,
                                                           s->logger,
                                                           std::make_pair(0, ((std::string) sink)),
                                                           s->pageQueues->at(i));

      // setup the sender, if we fail return false
      if(!sender->setup()) {
        return false;
      }

      // make the sender
      s->senders->emplace_back(sender);
    }
  }

  return true;
}

bool pdb::TRABroadcastStage::run(const pdb::Handle<pdb::ExJob> &job,
                                 const pdb::PDBPhysicalAlgorithmStatePtr &state,
                                 const std::shared_ptr<pdb::PDBStorageManagerBackend> &storage,
                                 const std::string &error) {

  // cast the state
  auto s = dynamic_pointer_cast<TRABroadcastState>(state);

  // success indicator
  atomic_bool success;
  success = true;

  /// 1. Run the self receiver,

  // create the buzzer
  atomic_int selfRecDone;
  selfRecDone = 0;
  PDBBuzzerPtr selfRefBuzzer = make_shared<PDBBuzzer>([&](PDBAlarm myAlarm, atomic_int &cnt) {

    // did we fail?
    if (myAlarm == PDBAlarm::GenericError) {
      success = false;
    }

    // we are done here
    cnt = 1;
  });

  // run the work
  {

    // make the work
    PDBWorkPtr myWork = std::make_shared<pdb::GenericWork>([&selfRecDone, s](const PDBBuzzerPtr& callerBuzzer) {

      // run the receiver
      if(s->selfReceiver->run()) {

        // signal that the run was successful
        callerBuzzer->buzz(PDBAlarm::WorkAllDone, selfRecDone);
      }
      else {

        // signal that the run was unsuccessful
        callerBuzzer->buzz(PDBAlarm::GenericError, selfRecDone);
      }
    });

    // run the work
    storage->getWorker()->execute(myWork, selfRefBuzzer);
  }

  /// 2. Run the senders

  // create the buzzer
  atomic_int sendersDone;
  sendersDone = 0;
  PDBBuzzerPtr sendersBuzzer = make_shared<PDBBuzzer>([&](PDBAlarm myAlarm, atomic_int &cnt) {

    // did we fail?
    if (myAlarm == PDBAlarm::GenericError) {
      success = false;
    }

    // we are done here
    cnt++;
  });

  // go through each sender and run them
  for(auto &sender : *s->senders) {

    // make the work
    PDBWorkPtr myWork = std::make_shared<pdb::GenericWork>([&sendersDone, sender, this](const PDBBuzzerPtr& callerBuzzer) {

      // run the sender
      if(sender->run()) {

        // signal that the run was successful
        callerBuzzer->buzz(PDBAlarm::WorkAllDone, sendersDone);
      }
      else {

        // signal that the run was unsuccessful
        callerBuzzer->buzz(PDBAlarm::GenericError, sendersDone);
      }
    });

    // run the work
    storage->getWorker()->execute(myWork, sendersBuzzer);
  }

  /// 3. Run the self receiver,

  // create the buzzer
  atomic_int queueFeederDone;
  queueFeederDone = 0;
  PDBBuzzerPtr queueFeederBuzzer = make_shared<PDBBuzzer>([&](PDBAlarm myAlarm, atomic_int &cnt) {

    // did we fail?
    if (myAlarm == PDBAlarm::GenericError) {
      success = false;
    }

    // we are done here
    cnt = 1;
  });

  // run the work
  {

    // make the work
    PDBWorkPtr myWork = std::make_shared<pdb::GenericWork>([&queueFeederDone, s](const PDBBuzzerPtr& callerBuzzer) {

      PDBPageHandle page;
      while((page = s->inputSet->getNextPage(0)) != nullptr) {
        for(auto &q : *s->pageQueues) {
          q->enqueue(page);
        }
      }
      for(auto &q : *s->pageQueues) {
        q->enqueue(nullptr);
      }

      // signal that the run was successful
      callerBuzzer->buzz(PDBAlarm::WorkAllDone, queueFeederDone);
    });

    // run the work
    storage->getWorker()->execute(myWork, selfRefBuzzer);
  }

  while (selfRecDone != 1) {
    selfRefBuzzer->wait();
  }

  while (sendersDone != s->senders->size()) {
    sendersBuzzer->wait();
  }

  while (queueFeederDone != 1) {
    queueFeederBuzzer->wait();
  }

  return true;
}

void pdb::TRABroadcastStage::cleanup(const pdb::PDBPhysicalAlgorithmStatePtr &state,
                                     const std::shared_ptr<pdb::PDBStorageManagerBackend> &storage) {
  std::cout << "cleanup\n";
}

const pdb::PDBSinkPageSetSpec *pdb::TRABroadcastStage::_sink = nullptr;
const pdb::Vector<pdb::PDBSourceSpec> *pdb::TRABroadcastStage::_sources = nullptr;
const pdb::String *pdb::TRABroadcastStage::_finalTupleSet = nullptr;
const pdb::Vector<pdb::Handle<pdb::PDBSourcePageSetSpec>> *pdb::TRABroadcastStage::_secondarySources = nullptr;
const pdb::Vector<pdb::PDBSetObject> *pdb::TRABroadcastStage::_setsToMaterialize = nullptr;